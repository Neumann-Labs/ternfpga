// ternary_tile — a full GEMV computed directly from a dense base-3 byte burst.
//
// The in-sim memory->compute->result datapath, composed from the verified pieces:
//   wbyte stream --(weight_feed)--> {w_row,valid} --(ternary_gemv_pipe)--> y = W·x
// i.e. DDR3-style dense weight bytes in, ternary matrix-vector out, multiply-free
// (0 DSP), at the pipelined ~280 MHz clock. This is the unit a LiteDRAM/MIG front
// end will drive on-board: point it at a weight tile in DRAM, stream the burst,
// read back y. Activations are stationary (latched on `start`).

`default_nettype none

module ternary_tile #(
    parameter int unsigned K  = 10,
    parameter int unsigned M  = 16,
    parameter int unsigned RW = (M <= 1) ? 1 : $clog2(M)
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   start,        // latch x, reset the GEMV
    input  wire signed [8*K-1:0]  x_flat,
    input  wire        [7:0]      wbyte,        // dense base-3 weight byte stream
    input  wire                   wbyte_valid,
    input  wire        [RW-1:0]   rd_addr,
    output wire signed [31:0]     rd_data,
    output wire                   done
);
    wire [2*K-1:0] w_row;
    wire           row_valid;

    weight_feed #(.K(K)) u_feed (
        .clk(clk), .rst_n(rst_n),
        .byte_in(wbyte), .byte_valid(wbyte_valid),
        .w_row(w_row), .row_valid(row_valid)
    );

    ternary_gemv_pipe #(.K(K), .M(M)) u_gemv (
        .clk(clk), .rst_n(rst_n), .start(start), .x_flat(x_flat),
        .w_row(w_row), .w_row_valid(row_valid),
        .rd_addr(rd_addr), .rd_data(rd_data), .done(done)
    );
endmodule

`default_nettype wire
