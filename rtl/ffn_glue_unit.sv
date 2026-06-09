// ffn_glue_unit — the FFN inter-projection glue (relu^2 * up * w + int8 requant) on the fabric.
//
// The last big HOST glue term: per channel f the soft CPU computes H=relu(gate_int)^2*up_int
// (int64), N=H*w (w = ffn_sub_norm fixed-point), and the per-token int8 requant
// h_q=round(N*127/max|N|) — every dequant scale + the RMSNorm normalizer cancel (the identity in
// models/ffn_glue_ref.py). On a cacheless rv32 those int64 mults dominate (~2.58M cyc/layer).
// This unit does it in hardware: gate_int/up_int/w_q resident in BRAM, two passes —
//   pass1: N[f] = relu(gate)^2 * up * w_q  ; track amaxN = max|N|
//   (one reciprocal)  R = msb(amaxN)+25 ; recip = (127<<R) / amaxN   (one restoring divide)
//   pass2: h_q[f] = clip( round_half_up(|N|*recip >> R) * sign(N), -128, 127 )  -> BRAM
// Outputs h_q[f] (int8) + amaxN (host applies it as the down_proj output dequant). Divider-free
// per channel (one divide per call). Bit-exact vs models/ffn_glue_unit_ref.py. Inputs obey the
// model invariant |gate_int|,|up_int| < 2^19 (GEMV over hidden=2560 of int8 x ternary). ALL memory
// reads in a clock-only block (+ ram_style="block") so gate/up/w infer BRAM (async-reset -> LUTRAM).
`default_nettype none

module ffn_glue_unit #(
    parameter int unsigned F_MAX = 6912,
    parameter int unsigned FW    = (F_MAX <= 1) ? 1 : $clog2(F_MAX),
    parameter int unsigned NW    = 88,        // width of N (signed): relu(g)^2 * up * w_q
    parameter int unsigned RW    = 128        // divider working width (numerator 127<<R)
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire [FW:0]           f_count,
    input  wire                  we,
    input  wire [FW-1:0]         waddr,
    input  wire signed [31:0]    gate_wdata,
    input  wire signed [31:0]    up_wdata,
    input  wire signed [15:0]    w_wdata,
    input  wire                  start,
    output reg                   done,
    input  wire [FW-1:0]         rd_addr,
    output reg  signed [7:0]     rd_data,      // h_q[rd_addr]
    output reg  [95:0]           amaxN,        // max|N|
    output reg  [31:0]           cycle_count
);
    (* ram_style = "block" *) reg signed [31:0] gmem [0:F_MAX-1];
    (* ram_style = "block" *) reg signed [31:0] umem [0:F_MAX-1];
    (* ram_style = "block" *) reg signed [15:0] wmem [0:F_MAX-1];
    (* ram_style = "block" *) reg signed [7:0]  hqmem[0:F_MAX-1];

    localparam [2:0] IDLE=0, P1=1, DIV=2, P2=3, FIN=4;
    reg [2:0]  st;
    reg [FW:0] fc, f;
    reg [33:0] recip;

    wire running = (st == P1) || (st == P2);

    // ---- write port + clock-only reads; index/valid travel with the read data ----
    reg signed [31:0] g_rd, u_rd;
    reg signed [15:0] w_rd;
    reg [FW-1:0]      f_rd;
    reg               rv;
    always_ff @(posedge clk) begin
        if (we) begin gmem[waddr] <= gate_wdata; umem[waddr] <= up_wdata; wmem[waddr] <= w_wdata; end
        g_rd <= gmem[f[FW-1:0]];
        u_rd <= umem[f[FW-1:0]];
        w_rd <= wmem[f[FW-1:0]];
        f_rd <= f[FW-1:0];
        rv   <= running && (f < fc);
        rd_data <= hqmem[rd_addr];      // single-stage read (2-edge latency in the tb)
    end

    // ---- combinational glue math (1 channel/cycle off the registered reads) -------
    wire [20:0]          g  = (g_rd > 0) ? g_rd[20:0] : 21'd0;   // relu, |gate_int|<2^19
    wire signed [21:0]   u  = $signed(u_rd[21:0]);               // |up_int|<2^19
    wire [41:0]          gg = g * g;
    wire signed [65:0]   H  = $signed({1'b0, gg}) * u;
    wire signed [NW-1:0] N  = H * w_rd;
    wire [NW-1:0]        absN = N[NW-1] ? (~N + 1'b1) : N;

    // requant (pass 2): h_q = round_half_up(|N|*recip >> Rsh) with sign
    reg  [6:0]   msb;
    integer i;
    always_comb begin msb = 0; for (i = 0; i < 96; i = i + 1) if (amaxN[i]) msb = i[6:0]; end
    wire [7:0]   Rsh   = msb + 8'd25;
    wire [127:0] prodq = absN * recip;
    wire [127:0] half  = 128'd1 << (Rsh - 8'd1);
    wire [127:0] hqmag = (prodq + half) >> Rsh;
    wire [7:0]   hqclip = (hqmag > 128'd127) ? 8'd127 : hqmag[7:0];
    wire signed [7:0] hq_signed = N[NW-1] ? -$signed({1'b0, hqclip}) : $signed({1'b0, hqclip});

    // ---- restoring divider state: recip = (127<<Rsh) / amaxN --------------------
    reg [RW-1:0] dnum, drem, dquo;
    reg [7:0]    dcnt;
    wire [RW-1:0] rem2 = (drem << 1) | dnum[RW-1];
    wire [RW-1:0] amaxN_ext = {{(RW-96){1'b0}}, amaxN};

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= IDLE; done <= 0; cycle_count <= 0; f <= 0;
        end else begin
            done <= 0;
            if (st != IDLE) cycle_count <= cycle_count + 1;
            case (st)
            IDLE: if (start) begin
                fc <= f_count; f <= 0; amaxN <= 0; cycle_count <= 0; st <= P1;
            end
            P1: begin
                if (f < fc) f <= f + 1;
                if (rv && (absN > amaxN)) amaxN <= absN;
                if (f >= fc && !rv) begin
                    dnum <= ({{(RW-8){1'b0}}, 8'd127}) << (msb + 8'd25);
                    drem <= 0; dquo <= 0; dcnt <= RW[7:0]; st <= DIV;
                end
            end
            DIV: begin
                if (dcnt != 0) begin
                    dnum <= dnum << 1;
                    if (rem2 >= amaxN_ext) begin drem <= rem2 - amaxN_ext; dquo <= (dquo << 1) | 1'b1; end
                    else                   begin drem <= rem2;             dquo <= (dquo << 1);        end
                    dcnt <= dcnt - 1;
                end else begin
                    recip <= dquo[33:0]; f <= 0; st <= P2;
                end
            end
            P2: begin
                if (f < fc) f <= f + 1;
                if (rv) hqmem[f_rd] <= hq_signed;
                if (f >= fc && !rv) st <= FIN;
            end
            FIN: begin done <= 1; st <= IDLE; end
            default: st <= IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
