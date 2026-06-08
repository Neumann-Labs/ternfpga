// ternary_gemv_pipe — row-streamed matrix-vector using the PIPELINED dot lane.
//
// Same y = W·x as ternary_gemv, but the lane is ternary_dot_pipe (3-stage, ~280
// MHz) instead of the combinational dot (~104 MHz). Rows stream in 1/cycle; the
// pipeline's latency is absorbed by following `valid_out` — the k-th result is
// row k because the pipe preserves order — so no cycle-counting. This is the
// high-clock streaming shape the on-board datapath will use. Still ZERO DSP.

`default_nettype none

module ternary_gemv_pipe #(
    parameter int unsigned K  = 8,
    parameter int unsigned M  = 16,
    parameter int unsigned RW = (M <= 1) ? 1 : $clog2(M)
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   start,        // latch x, reset counters
    input  wire signed [8*K-1:0]  x_flat,
    input  wire        [2*K-1:0]  w_row,
    input  wire                   w_row_valid,  // one weight row this cycle
    input  wire        [RW-1:0]   rd_addr,
    output wire signed [31:0]     rd_data,
    output reg                    done
);
    reg signed [8*K-1:0] x_reg;
    reg signed [31:0]    y_mem [0:M-1];
    reg [RW-1:0]         out_idx;     // result row index (in order)
    reg [31:0]           out_count;   // results written so far
    integer i;

    // Pipelined multiply-free lane: x stationary, a new w_row each valid cycle.
    wire               p_valid;
    wire signed [31:0] p_dot;
    ternary_dot_pipe #(.K(K)) u_pipe (
        .clk(clk), .rst_n(rst_n),
        .valid_in(w_row_valid), .a_flat(x_reg), .w_flat(w_row),
        .valid_out(p_valid), .dot(p_dot)
    );

    assign rd_data = y_mem[rd_addr];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0; out_idx <= '0; out_count <= 32'd0; x_reg <= '0;
            for (i = 0; i < M; i = i + 1) y_mem[i] <= 32'sd0;
        end else if (start) begin
            x_reg <= x_flat; out_idx <= '0; out_count <= 32'd0; done <= 1'b0;
            for (i = 0; i < M; i = i + 1) y_mem[i] <= 32'sd0;
        end else if (p_valid) begin
            y_mem[out_idx] <= p_dot;          // results arrive in row order
            out_idx        <= out_idx + 1'b1;
            out_count      <= out_count + 32'd1;
            if (out_count == M - 1) done <= 1'b1;
        end
    end
endmodule

`default_nettype wire
