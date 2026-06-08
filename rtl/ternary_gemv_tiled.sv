// ternary_gemv_tiled — ternary GEMV over an arbitrary row width via K-tiling.
//
// Real layers have hidden dims of hundreds, far wider than one dot lane. This
// accumulates each output row's dot over NT tiles of K lanes (row width KT=K*NT)
// using a single ternary_dot, so a fixed lane computes full-width rows. Weight
// tiles stream row-major (row 0: tile 0..NT-1; row 1: ...); the activation is
// stationary (latched on start). Still ZERO DSP — the foundation for the FFN.

`default_nettype none

module ternary_gemv_tiled #(
    parameter int unsigned K  = 8,                 // lanes per tile
    parameter int unsigned NT = 4,                 // tiles per row  (KT = K*NT)
    parameter int unsigned M  = 16,                // output rows
    parameter int unsigned RW = (M <= 1) ? 1 : $clog2(M)
) (
    input  wire                       clk,
    input  wire                       rst_n,
    input  wire                       start,        // latch x, reset
    input  wire signed [8*K*NT-1:0]   x_flat,       // full activation: K*NT int8
    input  wire        [2*K-1:0]      w_tile,        // one K-wide ternary weight tile
    input  wire                       w_tile_valid,
    input  wire        [RW-1:0]       rd_addr,
    output wire signed [31:0]         rd_data,
    output reg                        done
);
    reg signed [8*K*NT-1:0] x_reg;
    reg [31:0]              m_idx, t_idx;
    reg signed [31:0]       acc;
    reg signed [31:0]       y_mem [0:M-1];
    integer i;

    // tile t of the stationary activation, dotted with the incoming weight tile
    wire signed [8*K-1:0] x_tile = x_reg[8*K*t_idx +: 8*K];
    wire signed [31:0]    partial;
    ternary_dot #(.K(K)) u_dot (.a_flat(x_tile), .w_flat(w_tile), .dot(partial));

    assign rd_data = y_mem[rd_addr];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0; m_idx <= 32'd0; t_idx <= 32'd0; acc <= 32'sd0; x_reg <= '0;
            for (i = 0; i < M; i = i + 1) y_mem[i] <= 32'sd0;
        end else if (start) begin
            x_reg <= x_flat; m_idx <= 32'd0; t_idx <= 32'd0; acc <= 32'sd0; done <= 1'b0;
            for (i = 0; i < M; i = i + 1) y_mem[i] <= 32'sd0;
        end else if (w_tile_valid) begin
            logic signed [31:0] acc_next;
            acc_next = (t_idx == 32'd0) ? partial : (acc + partial);
            if (t_idx == NT - 1) begin                 // last tile of this row
                y_mem[m_idx[RW-1:0]] <= acc_next;
                t_idx <= 32'd0;
                acc   <= 32'sd0;
                if (m_idx == M - 1) begin done <= 1'b1; m_idx <= 32'd0; end
                else                       m_idx <= m_idx + 32'd1;
            end else begin
                acc   <= acc_next;
                t_idx <= t_idx + 32'd1;
            end
        end
    end
endmodule

`default_nettype wire
