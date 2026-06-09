// ternary_gemv_stream — BRAM-centric, pipelined ternary GEMV over real row widths.
//
//   y[m] = sum_{t=0..NT-1} dot( x_tile[t] , w_tile[m][t] ),   row width KT = K*NT
//
// The fit sweep (bench/results/fit_sweep.md) showed the register-resident,
// single-cycle, flat-part-select `ternary_gemv_tiled` blows flip-flops to 79% of
// the 35T and fails timing (63 MHz) at real width — the NT:1 activation mux is the
// critical path. This is the scalable replacement the sweep mandated:
//   * the activation lives in a BRAM, read one K-wide tile per cycle by a
//     SEQUENTIAL address (no mux),
//   * the K-wide multiply-free dot is the 3-stage pipelined lane (ternary_dot_pipe),
//   * partials stream out in order and are accumulated NT-at-a-time into a y BRAM.
// Weight tiles stream row-major (row 0 tiles 0..NT-1, row 1 tiles 0..NT-1, ...);
// the activation is loaded once and reused for every row. Still ZERO DSP.
//
// Runtime dims: `nt` (tiles/row, KT = K*nt) and `m_rows` (output rows); K is fixed.
// Activation read is registered (1-cycle); weight tile + valid are delayed one cycle
// to match, so x_tile[t] and w_tile[t] enter the dot together.

`default_nettype none

module ternary_gemv_stream #(
    parameter int unsigned K      = 16,                          // lanes per tile (fixed)
    parameter int unsigned NT_MAX = 512,                         // max tiles/row (KT_MAX=K*NT_MAX)
    parameter int unsigned M_MAX  = 8192,                        // max output rows
    parameter int unsigned AW = (NT_MAX <= 1) ? 1 : $clog2(NT_MAX),
    parameter int unsigned MW = (M_MAX  <= 1) ? 1 : $clog2(M_MAX)
) (
    input  wire                  clk,
    input  wire                  rst_n,
    // dims (sampled at start)
    input  wire [AW-1:0]         nt,                             // tiles per row (>=1)
    input  wire [MW-1:0]         m_rows,                         // output rows (>=1)
    // activation load: write `nt` words of K int8 each (do this before start)
    input  wire                  x_we,
    input  wire [AW-1:0]         x_waddr,
    input  wire signed [8*K-1:0] x_wdata,
    // compute
    input  wire                  start,                          // reset counters/accum, latch dims
    input  wire                  w_valid,                        // one K-wide weight tile this cycle
    input  wire [2*K-1:0]        w_tile,
    // output (registered read: present rd_addr, read rd_data next cycle)
    input  wire [MW-1:0]         rd_addr,
    output reg  signed [31:0]    rd_data,
    output reg                   done
);
    // ---- operand BRAMs ----
    reg signed [8*K-1:0] x_mem [0:NT_MAX-1];     // activation, K int8 per word
    reg signed [31:0]    y_mem [0:M_MAX-1];       // results, int32 per row

    // latched dims
    reg [AW-1:0] nt_r;
    reg [MW-1:0] m_r;

    // ---- activation write port ----
    always_ff @(posedge clk) begin
        if (x_we) x_mem[x_waddr] <= x_wdata;
    end

    // ---- input side: tile counter drives the sequential activation read; weight
    // tile + valid are delayed one cycle to align with the registered BRAM read ----
    reg [AW-1:0]         ti;                       // tile within row, 0..nt-1
    reg signed [8*K-1:0] x_tile;                   // registered read data
    reg [2*K-1:0]        w_tile_d;
    reg                  w_valid_d;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ti <= '0; x_tile <= '0; w_tile_d <= '0; w_valid_d <= 1'b0;
        end else begin
            x_tile    <= x_mem[ti];                // x_tile[T+1] = x_mem[ti(T)]
            w_tile_d  <= w_tile;
            w_valid_d <= w_valid & ~start;
            if (start)              ti <= '0;
            else if (w_valid)       ti <= (ti == nt_r - 1) ? '0 : ti + 1'b1;
        end
    end

    // ---- the multiply-free pipelined lane (3-stage, ~280 MHz, 0 DSP) ----
    wire               p_valid;
    wire signed [31:0] p_dot;
    ternary_dot_pipe #(.K(K)) u_dot (
        .clk(clk), .rst_n(rst_n),
        .valid_in(w_valid_d), .a_flat(x_tile), .w_flat(w_tile_d),
        .valid_out(p_valid), .dot(p_dot)
    );

    // ---- output side: accumulate NT consecutive partials into one row ----
    // Control (counters/done) carries the async reset; the y_mem WRITE is kept in
    // a clock-only block below so Vivado can infer it as BRAM (a RAM write inside an
    // async-reset block cannot be a BRAM — that splays it into registers and fails).
    reg signed [31:0] acc;
    reg [AW-1:0]      oti;                          // output tile counter
    reg [MW-1:0]      omi;                          // output row index

    wire signed [31:0] acc_next = (oti == '0) ? p_dot : (acc + p_dot);
    wire               last_tile = (oti == nt_r - 1);
    wire               row_done  = p_valid & last_tile;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0; acc <= '0; oti <= '0; omi <= '0; nt_r <= '0; m_r <= '0;
        end else if (start) begin
            done <= 1'b0; acc <= '0; oti <= '0; omi <= '0; nt_r <= nt; m_r <= m_rows;
        end else if (p_valid) begin
            if (last_tile) begin                   // last tile of this row
                acc <= '0;
                oti <= '0;
                if (omi == m_r - 1) done <= 1'b1;
                else                omi <= omi + 1'b1;
            end else begin
                acc <= acc_next;
                oti <= oti + 1'b1;
            end
        end
    end

    // ---- y BRAM: clock-only write + registered read (both reset-free -> infers BRAM) ----
    always_ff @(posedge clk) begin
        if (row_done) y_mem[omi] <= acc_next;
        rd_data <= y_mem[rd_addr];
    end

endmodule

`default_nettype wire
