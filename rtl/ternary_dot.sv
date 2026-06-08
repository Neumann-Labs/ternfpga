// ternary_dot — a multiplier-free ternary dot product (the core of the engine).
//
// Computes  dot = sum_i  ternary(w_i) * a_i  , where each "multiply" is a
// sign/zero select (NO DSP, NO multiplier): w=+1 -> +a, w=-1 -> -a, w=0 -> 0.
// This is the literal datapath that lets the FPGA do ternary arithmetic the
// GPU can only emulate by dequantizing.
//
// Encoding: weights are 2-bit codes  01=+1, 10=-1, 00/11=0 ; activations int8.
// Combinational v0 (correctness-first); pipelining comes once it's bit-exact.

`default_nettype none

module ternary_dot #(
    parameter int unsigned K = 8           // lanes (ternary weights / activations)
) (
    input  wire signed [8*K-1:0] a_flat,   // K signed int8 activations, little-endian
    input  wire        [2*K-1:0] w_flat,   // K ternary 2-bit codes, little-endian
    output reg  signed [31:0]    dot
);

    always_comb begin
        dot = 32'sd0;
        for (int unsigned i = 0; i < K; i++) begin
            logic signed [7:0] a;
            logic        [1:0] code;
            a    = $signed(a_flat[8*i +: 8]);
            code = w_flat[2*i +: 2];
            unique case (code)
                2'b01:   dot = dot + 32'(a);   // +1 * a   (sign-extended)
                2'b10:   dot = dot - 32'(a);   // -1 * a
                default: dot = dot;            //  0 * a
            endcase
        end
    end

endmodule

`default_nettype wire
