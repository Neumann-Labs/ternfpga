// ternary_pe_array — P parallel multiply-free ternary dot lanes.
//
// Computes P output rows per cycle (y[p] = W[p]·x), all sharing one stationary
// activation vector. This is the throughput knob: P should be sized to the DDR3
// weight-bandwidth roofline (~0.6-0.8 GB/s here), NOT to peak FLOPs — more lanes
// than memory can feed is wasted area. Still ZERO DSP (P instances of ternary_dot).

`default_nettype none

module ternary_pe_array #(
    parameter int unsigned K = 8,           // lanes per dot
    parameter int unsigned P = 4            // parallel rows/cycle
) (
    input  wire signed [8*K-1:0]    a_flat, // shared activation vector
    input  wire        [2*K*P-1:0]  w_rows, // P packed ternary weight rows
    output wire signed [32*P-1:0]   dots    // P int32 dot results
);
    genvar p;
    generate
        for (p = 0; p < P; p++) begin : lane
            ternary_dot #(.K(K)) u_dot (
                .a_flat(a_flat),
                .w_flat(w_rows[2*K*p +: 2*K]),
                .dot   (dots[32*p +: 32])
            );
        end
    endgenerate
endmodule

`default_nettype wire
