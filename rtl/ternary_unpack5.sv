// ternary_unpack5 — dense base-3 weight unpacker (5 ternary weights per byte).
//
// Ternary has 3 states, and 3^5 = 243 < 256, so FIVE ternary weights pack into
// ONE byte: 1.6 bits/weight (the log2(3)=1.585-bit optimum), 20% tighter than the
// 2-bit-code packing and 5x tighter than INT8. Since batch-1 decode is DDR3-
// bandwidth-bound, that is a direct 20% cut in weight traffic vs the simple codes.
//
// This combinational decoder takes one packed byte and emits the 5 ternary 2-bit
// codes (01=+1, 10=-1, 00=0) that ternary_dot/ternary_dot_pipe consume — so a
// DDR3 burst of dense bytes can feed the multiply-free lanes directly.
//
//   byte = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4,  trit t in {0,1,2}
//   trit 0 -> 0 (00),  trit 1 -> +1 (01),  trit 2 -> -1 (10)

`default_nettype none

module ternary_unpack5 (
    input  wire  [7:0] byte_in,
    output logic [9:0] codes_out   // 5 x 2-bit codes; lane 0 in bits [1:0]
);
    // trit value {0,1,2} -> ternary 2-bit code
    function automatic logic [1:0] trit_to_code(input logic [7:0] t);
        trit_to_code = (t == 8'd1) ? 2'b01 :
                       (t == 8'd2) ? 2'b10 : 2'b00;
    endfunction

    always_comb begin
        logic [7:0] b, t0, t1, t2, t3, t4;
        b  = byte_in;
        t0 = b % 8'd3; b = b / 8'd3;
        t1 = b % 8'd3; b = b / 8'd3;
        t2 = b % 8'd3; b = b / 8'd3;
        t3 = b % 8'd3; b = b / 8'd3;
        t4 = b % 8'd3;
        codes_out = {trit_to_code(t4), trit_to_code(t3), trit_to_code(t2),
                     trit_to_code(t1), trit_to_code(t0)};
    end
endmodule

`default_nettype wire
