// ternary_gemv_packed — matrix-vector over DENSE base-3 packed weight rows.
//
// The first integration module: weight rows arrive in the on-DDR3 dense layout
// (base-3, 5 ternary weights/byte = 1.6 bits/weight), are unpacked by an array
// of ternary_unpack5 into the 2-bit lane codes, and fed to the multiply-free
// ternary_dot — y = W·x with ZERO DSP, end to end from packed bytes. This is
// the shape of the real decode datapath (memory burst -> unpack -> MAC), proven
// bit-exact in sim before the on-board DDR3 path is built.
//
// One row per w_row_valid cycle; results read back via rd_addr/rd_data.

`default_nettype none

module ternary_gemv_packed #(
    parameter int unsigned K   = 10,                 // weights/row (multiple of 5 = no pad waste)
    parameter int unsigned M   = 16,                 // rows
    parameter int unsigned BPR = (K + 4) / 5,        // packed bytes per row
    parameter int unsigned RW  = (M <= 1) ? 1 : $clog2(M)
) (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    start,            // latch x, reset row counter
    input  wire signed [8*K-1:0]   x_flat,
    input  wire        [8*BPR-1:0] w_row_packed,     // BPR dense base-3 bytes (one row)
    input  wire                    w_row_valid,
    input  wire        [RW-1:0]    rd_addr,
    output wire signed [31:0]      rd_data,
    output reg                     done
);
    // ---- Unpack BPR bytes -> BPR*5 lane codes; use the first K ----
    localparam int unsigned NCODE = BPR * 5;
    wire [2*NCODE-1:0] codes_all;
    genvar b;
    generate
        for (b = 0; b < BPR; b++) begin : g_unpack
            ternary_unpack5 u_unp (
                .byte_in  (w_row_packed[8*b +: 8]),
                .codes_out(codes_all[10*b +: 10])
            );
        end
    endgenerate
    wire [2*K-1:0] w_flat = codes_all[2*K-1:0];

    // ---- Multiply-free dot against the stationary activation vector ----
    reg signed [8*K-1:0] x_reg;
    reg [31:0]           row_idx;
    reg signed [31:0]    y_mem [0:M-1];
    integer i;

    wire signed [31:0] row_dot;
    ternary_dot #(.K(K)) u_dot (.a_flat(x_reg), .w_flat(w_flat), .dot(row_dot));

    assign rd_data = y_mem[rd_addr];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0; row_idx <= 32'd0; x_reg <= '0;
            for (i = 0; i < M; i = i + 1) y_mem[i] <= 32'sd0;
        end else if (start) begin
            x_reg <= x_flat; row_idx <= 32'd0; done <= 1'b0;
            for (i = 0; i < M; i = i + 1) y_mem[i] <= 32'sd0;
        end else if (w_row_valid) begin
            y_mem[row_idx[RW-1:0]] <= row_dot;
            if (row_idx == M - 1) begin done <= 1'b1; row_idx <= 32'd0; end
            else                        row_idx <= row_idx + 32'd1;
        end
    end
endmodule

`default_nettype wire
