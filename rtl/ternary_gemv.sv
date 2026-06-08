// ternary_gemv — ternary matrix-vector multiply  y = W * x
//
//   y[m] = sum_{k} ternary(W[m][k]) * x[k]      m in [0,M), k in [0,K)
//
// W is an M x K ternary matrix; x is a K-vector of signed int8 activations.
// The weight matrix is streamed ONE ROW PER CYCLE (the bandwidth-bound regime
// the real engine lives in: weights come from DDR3, activations stay on-chip).
// Each row is reduced by the multiplier-free `ternary_dot` lane (sign-select +
// adder tree, zero DSP). Results land in an on-chip result memory and are read
// back through a narrow address port `rd_addr -> rd_data` (no wide result bus).
//
// This is the v0 correctness model: one row/cycle, one dot lane. Throughput
// parallelism (P lanes) and DDR3 streaming come later; the contract proved here
// is bit-exact integer GEMV.

`default_nettype none

module ternary_gemv #(
    parameter int unsigned K  = 8,                         // input dimension
    parameter int unsigned M  = 16,                        // output dimension (rows)
    parameter int unsigned RW = (M <= 1) ? 1 : $clog2(M)   // result-address width (derived)
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   start,        // pulse 1 cycle: latch x, reset counter
    input  wire signed [8*K-1:0]  x_flat,       // activation vector (K signed int8)
    input  wire        [2*K-1:0]  w_row,         // one weight row: K ternary 2-bit codes
    input  wire                   w_row_valid,   // assert while streaming rows
    input  wire        [RW-1:0]   rd_addr,       // read back result row rd_addr ...
    output wire signed [31:0]     rd_data,       // ... as signed int32 (combinational)
    output reg                    done           // high once all M rows are written
);

    reg  signed [8*K-1:0] x_reg;            // stationary activations
    reg         [31:0]    row_idx;          // 0..M
    reg  signed [31:0]    y_mem [0:M-1];    // result memory (per-row int32, no wide bus)
    integer i;

    // Combinational multiply-free dot of the current row against x_reg.
    wire signed [31:0] row_dot;
    ternary_dot #(.K(K)) u_dot (
        .a_flat (x_reg),
        .w_flat (w_row),
        .dot    (row_dot)
    );

    assign rd_data = y_mem[rd_addr];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_reg   <= '0;
            row_idx <= 32'd0;
            done    <= 1'b0;
            for (i = 0; i < M; i = i + 1) y_mem[i] <= 32'sd0;
        end else if (start) begin
            x_reg   <= x_flat;
            row_idx <= 32'd0;
            done    <= 1'b0;
        end else if (w_row_valid && (row_idx < M)) begin
            y_mem[row_idx[RW-1:0]] <= row_dot;
            row_idx <= row_idx + 32'd1;
            if (row_idx == (M - 1)) done <= 1'b1;
        end
    end

endmodule

`default_nettype wire
