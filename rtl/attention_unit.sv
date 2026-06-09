// attention_unit — one query (head) vs a BRAM-resident KV cache, on the fabric.
//
// The measured glue is GLUE-BOUND: 83% is host-side attention over DRAM-resident KV on the
// cacheless VexRiscv (16.2M cyc/layer). This unit does that attention in hardware with KV in
// BRAM at ~1 MAC/cycle, collapsing it. For one query q[D] and T cached keys/values (D-wide):
//   scores[j] = sum_d q[d]*k[j][d]                                   (int16xint16 MAC)
//   idx[j]    = clamp((max_s - scores[j]) >> score_shift, 0, EXP_N-1)
//   e[j]      = EXP_LUT[idx[j]]              (Q15, BRAM)              ;  sum_e = sum_j e[j]
//   num[d]    = sum_j e[j]*v[j][d]                                   (raw weighted sum)
// Outputs num[d] (int) + sum_e; ctx[d]=num[d]/sum_e is a trivial host step (D divides), so the
// unit stays divider-free (clean synth). Bit-exact vs models/attn_unit_ref.py.
`default_nettype none

module attention_unit #(
    parameter int unsigned D      = 128,
    parameter int unsigned T_MAX  = 128,
    parameter int unsigned EXP_N  = 4096,
    parameter int unsigned DW = (D     <= 1) ? 1 : $clog2(D),
    parameter int unsigned TW = (T_MAX <= 1) ? 1 : $clog2(T_MAX),
    parameter int unsigned EW = $clog2(EXP_N)
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire [TW:0]           t_keys,
    input  wire [5:0]            score_shift,
    input  wire                  q_we,
    input  wire [DW-1:0]         q_waddr,
    input  wire signed [15:0]    q_wdata,
    input  wire                  kv_we,                    // write k AND v at addr j*D+d
    input  wire [TW+DW-1:0]      kv_waddr,
    input  wire signed [15:0]    k_wdata,
    input  wire signed [15:0]    v_wdata,
    input  wire                  lut_we,
    input  wire [EW-1:0]         lut_waddr,
    input  wire [15:0]           lut_wdata,
    input  wire                  start,
    output reg                   done,
    input  wire [DW-1:0]         rd_addr,
    output reg  signed [47:0]    rd_data,                  // num[rd_addr]
    output reg  [31:0]           sum_e,
    output reg  [31:0]           cycle_count
);
    reg signed [15:0] qmem  [0:D-1];
    reg signed [15:0] kmem  [0:T_MAX*D-1];
    reg signed [15:0] vmem  [0:T_MAX*D-1];
    reg        [15:0] elut  [0:EXP_N-1];
    reg signed [47:0] smem  [0:T_MAX-1];
    reg        [15:0] emem  [0:T_MAX-1];                   // exp weights (Q15, >=0)
    reg signed [47:0] nummem[0:D-1];

    always_ff @(posedge clk) begin
        if (q_we)  qmem[q_waddr] <= q_wdata;
        if (kv_we) begin kmem[kv_waddr] <= k_wdata; vmem[kv_waddr] <= v_wdata; end
        if (lut_we) elut[lut_waddr] <= lut_wdata;
    end

    localparam [2:0] IDLE=0, SCORE=1, EXP=2, AV=3, FIN=4;
    reg [2:0]        st;
    reg [TW:0]       tk;
    reg [5:0]        shift_r;

    reg [TW:0]       jc;            // key counter
    reg [DW:0]       dc;            // dim counter
    reg [TW+DW:0]    voff;          // running v/k address = jc*D + dc
    reg              gen, gen_q;
    reg [TW:0]       jc_q;
    reg [DW-1:0]     dc_q;
    reg signed [15:0] a_rd, b_rd;
    reg signed [47:0] acc, max_s;

    // EXP 2-stage pipeline
    reg [TW:0]        ec;
    reg               e_v1, e_v2;
    reg [TW:0]        ec1, ec2;
    reg signed [47:0] score_rd;
    reg        [15:0] elut_rd;
    wire signed [47:0] drop_raw = max_s - score_rd;
    wire signed [47:0] drop_sh  = (drop_raw[47]) ? 48'sd0 : (drop_raw >>> shift_r);
    wire [EW-1:0]      idx_c    = (drop_sh >= EXP_N) ? (EXP_N-1) : drop_sh[EW-1:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= IDLE; done <= 0; gen <= 0; gen_q <= 0; cycle_count <= 0;
        end else begin
            done <= 0;
            if (st != IDLE) cycle_count <= cycle_count + 1;
            case (st)
            IDLE: if (start) begin
                tk <= t_keys; shift_r <= score_shift;
                jc <= 0; dc <= 0; voff <= 0; gen <= 1; gen_q <= 0;
                acc <= 0; max_s <= {1'b1, 47'b0}; cycle_count <= 0;
                st <= SCORE;
            end

            // scores[j] = sum_d q[d]*k[j][d], track max -------------------------
            SCORE: begin
                a_rd  <= qmem[dc[DW-1:0]];
                b_rd  <= kmem[voff];
                gen_q <= gen; jc_q <= jc; dc_q <= dc[DW-1:0];
                if (gen) begin
                    if (dc == D-1) begin
                        dc <= 0; voff <= voff + 1;
                        if (jc == tk-1) gen <= 0; else jc <= jc + 1;
                    end else begin dc <= dc + 1; voff <= voff + 1; end
                end
                if (gen_q) begin
                    if (dc_q == 0)            acc <= a_rd * b_rd;
                    else if (dc_q == D-1) begin
                        smem[jc_q] <= acc + a_rd * b_rd;
                        if ((acc + a_rd * b_rd) > max_s) max_s <= acc + a_rd * b_rd;
                    end else                 acc <= acc + a_rd * b_rd;
                end
                if (!gen && !gen_q) begin ec <= 0; e_v1 <= 0; e_v2 <= 0; sum_e <= 0; st <= EXP; end
            end

            // e[j] = LUT[clamp((max-score)>>shift)] ; sum_e = sum e --------------
            EXP: begin
                score_rd <= smem[ec];                       // stage0 read
                e_v1 <= (ec < tk); ec1 <= ec;
                if (ec < tk) ec <= ec + 1;
                elut_rd <= elut[idx_c];                     // stage1: score_rd -> idx -> LUT
                e_v2 <= e_v1; ec2 <= ec1;
                if (e_v2) begin emem[ec2] <= elut_rd; sum_e <= sum_e + elut_rd; end  // stage2
                if (ec >= tk && !e_v1 && !e_v2) begin
                    jc <= 0; dc <= 0; voff <= 0; gen <= 1; gen_q <= 0; acc <= 0; st <= AV;
                end
            end

            // num[d] = sum_j e[j]*v[j][d] ; d outer, j inner --------------------
            AV: begin
                a_rd  <= $signed({1'b0, emem[jc]});         // e >= 0
                b_rd  <= vmem[voff];
                gen_q <= gen; jc_q <= jc; dc_q <= dc[DW-1:0];
                if (gen) begin
                    if (jc == tk-1) begin
                        jc <= 0; voff <= dc + 1;            // next d, j=0 -> addr = d+1
                        if (dc == D-1) gen <= 0; else dc <= dc + 1;
                    end else begin jc <= jc + 1; voff <= voff + D; end
                end
                if (gen_q) begin
                    if (jc_q == 0)            acc <= a_rd * b_rd;
                    else if (jc_q == tk-1)    nummem[dc_q] <= acc + a_rd * b_rd;
                    else                     acc <= acc + a_rd * b_rd;
                end
                if (!gen && !gen_q) st <= FIN;
            end

            FIN: begin done <= 1; st <= IDLE; end
            default: st <= IDLE;
            endcase
        end
    end

    always_ff @(posedge clk) rd_data <= nummem[rd_addr];
endmodule

`default_nettype wire
