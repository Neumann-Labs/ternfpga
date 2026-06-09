// ffn_glue_unit — the FFN inter-projection glue (relu^2 * up * w + int8 requant) on the fabric.
//
// The last big HOST glue term: per channel f the soft CPU computes H=relu(gate_int)^2*up_int
// (int64), N=H*w (w = ffn_sub_norm fixed-point), and the per-token int8 requant
// h_q=round(N*127/max|N|) — every dequant scale + the RMSNorm normalizer cancel (the identity in
// models/ffn_glue_ref.py). On a cacheless rv32 those int64 mults dominate (~2.58M cyc/layer).
// This unit does it in hardware: gate_int/up_int/w_q in BRAM, two passes —
//   pass1: N[f] = relu(gate)^2 * up * w_q  ; track amaxN = max|N|
//   (one reciprocal)  R = msb(amaxN)+25 ; recip = (127<<R) / amaxN   (one restoring divide)
//   pass2: h_q[f] = clip( round_half_up(|N|*recip >> R) * sign(N), -128, 127 )  -> BRAM
// The compute is a 7-stage pipeline (one multiply/shift per stage) so it closes 100 MHz; the
// per-stage valid + channel index + operands travel together (the read produces g_rd/f_rd/rv
// aligned, every stage delays them by one). Outputs h_q (int8) + max|N|; divider-free per channel.
// Bit-exact vs models/ffn_glue_unit_ref.py. ALL memory reads/writes in a clock-only block (+
// ram_style="block") so gate/up/w/hq infer BRAM (async-reset read, or hqmem write in a separate
// block, forces a FF array + decoder -> 115% LUT/134% FF).
`default_nettype none

module ffn_glue_unit #(
    parameter int unsigned F_MAX = 6912,
    parameter int unsigned FW    = (F_MAX <= 1) ? 1 : $clog2(F_MAX),
    parameter int unsigned NW    = 88,
    parameter int unsigned RW    = 128
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
    output reg  signed [7:0]     rd_data,
    output reg  [95:0]           amaxN,
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

    // requant scale (stable during P2): R = msb(amaxN)+25. amaxN is frozen after P1, so Rsh/half
    // are REGISTERED once at the P1->DIV transition — keeping the msb priority encoder off the
    // per-cycle critical path (it was amaxN -> msb -> Rsh -> barrel-shift, the WNS path).
    reg [6:0] msb;
    integer i;
    always_comb begin msb = 0; for (i = 0; i < 96; i = i + 1) if (amaxN[i]) msb = i[6:0]; end
    reg [7:0]   Rsh_r;
    reg [127:0] half_r;

    // ---- block A: clock-only memory writes + reads (BRAM); index/valid aligned with reads -----
    reg signed [31:0] g_rd, u_rd;
    reg signed [15:0] w_rd;
    reg [FW-1:0]      f_rd;
    reg               rv;
    // stage-7 outputs (computed in block B) feed the hqmem write here
    wire              v7;
    wire [FW-1:0]     f7;
    wire signed [7:0] hq7;
    wire              hq_we = (st == P2) && v7;
    always_ff @(posedge clk) begin
        if (we) begin gmem[waddr] <= gate_wdata; umem[waddr] <= up_wdata; wmem[waddr] <= w_wdata; end
        g_rd <= gmem[f[FW-1:0]];
        u_rd <= umem[f[FW-1:0]];
        w_rd <= wmem[f[FW-1:0]];
        f_rd <= f[FW-1:0];
        rv   <= running && (f < fc);
        if (hq_we) hqmem[f7] <= hq7;
        rd_data <= hqmem[rd_addr];
    end

    // ---- block B: 7-stage compute pipeline (one multiply/shift per stage) ----------------------
    reg [20:0]          gp1; reg signed [21:0] u1; reg signed [15:0] w1; reg [FW-1:0] f1; reg v1;
    reg [41:0]          gg2; reg signed [21:0] u2; reg signed [15:0] w2; reg [FW-1:0] f2; reg v2;
    reg signed [63:0]   H3;  reg signed [15:0] w3; reg [FW-1:0] f3; reg v3;
    reg signed [NW-1:0] N4;  reg [FW-1:0] f4; reg v4;
    reg [NW-1:0]        aN5; reg sgn5; reg [FW-1:0] f5; reg v5;
    reg [NW+34-1:0]     pq6;  reg sgn6; reg [FW-1:0] f6;  reg v6;
    reg [NW+34-1:0]     sum7; reg sgn7; reg [FW-1:0] f7s; reg v7s;     // add stage (split off the shift)
    reg signed [7:0]    hq8r; reg [FW-1:0] f8r; reg v8r;

    wire [NW-1:0]      absN4   = N4[NW-1] ? (~N4 + 1'b1) : N4;       // |N| at stage 4 (for max)
    wire [NW+34-1:0]   shifted = sum7 >> Rsh_r;                       // round_half_up >> R (regd scale)
    wire [7:0]         hqmag   = (shifted > {{(NW+34-8){1'b0}}, 8'd127}) ? 8'd127 : shifted[7:0];
    wire signed [7:0]  hqsig   = sgn7 ? -$signed({1'b0, hqmag}) : $signed({1'b0, hqmag});

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin v1<=0; v2<=0; v3<=0; v4<=0; v5<=0; v6<=0; v7s<=0; v8r<=0; end
        else begin
            gp1 <= (g_rd > 0) ? g_rd[20:0] : 21'd0; u1 <= u_rd[21:0]; w1 <= w_rd; f1 <= f_rd; v1 <= rv;
            gg2 <= gp1 * gp1;                       u2 <= u1; w2 <= w1; f2 <= f1; v2 <= v1;
            H3  <= $signed({1'b0, gg2}) * u2;        w3 <= w2; f3 <= f2; v3 <= v2;
            N4  <= H3 * w3;                          f4 <= f3; v4 <= v3;
            aN5 <= absN4; sgn5 <= N4[NW-1];          f5 <= f4; v5 <= v4;
            pq6 <= aN5 * recip; sgn6 <= sgn5;        f6 <= f5; v6 <= v5;
            sum7 <= pq6 + half_r[NW+34-1:0]; sgn7 <= sgn6; f7s <= f6; v7s <= v6;
            hq8r <= hqsig;                           f8r <= f7s; v8r <= v7s;
        end
    end
    assign v7 = v8r; assign f7 = f8r; assign hq7 = hq8r;
    wire pipe_busy = v1 | v2 | v3 | v4 | v5 | v6 | v7s | v8r;

    // ---- restoring divider: recip = (127<<Rsh) / amaxN -----------------------------------------
    reg [RW-1:0] dnum, drem, dquo;
    reg [7:0]    dcnt;
    wire [RW-1:0] rem2 = (drem << 1) | dnum[RW-1];
    wire [RW-1:0] amaxN_ext = {{(RW-96){1'b0}}, amaxN};

    // ---- control FSM ---------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= IDLE; done <= 0; cycle_count <= 0; f <= 0;
        end else begin
            done <= 0;
            if (st != IDLE) cycle_count <= cycle_count + 1;
            case (st)
            IDLE: if (start) begin fc <= f_count; f <= 0; amaxN <= 0; cycle_count <= 0; st <= P1; end
            P1: begin
                if (f < fc) f <= f + 1;
                if (v4 && (absN4 > amaxN)) amaxN <= absN4;        // track max|N| at stage 4
                if (f >= fc && !pipe_busy) begin
                    Rsh_r  <= msb + 8'd25;                          // freeze the requant scale
                    half_r <= 128'd1 << (msb + 8'd24);              // 1 << (R-1)
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
                if (f < fc) f <= f + 1;                            // hqmem write is in block A (hq_we)
                if (f >= fc && !pipe_busy) st <= FIN;
            end
            FIN: begin done <= 1; st <= IDLE; end
            default: st <= IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
