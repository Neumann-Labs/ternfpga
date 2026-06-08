// ternary_dot_pipe — pipelined multiply-free ternary dot product.
//
// Same arithmetic as ternary_dot (sign-select + add, ZERO DSP), but the K-wide
// reduction is split into registered stages so no single combinational path
// runs the whole adder tree. This trades 3 cycles of latency for a much higher
// Fmax — the standard move once correctness is proven. Throughput is 1 dot/cycle
// (a new valid_in every cycle); results emerge `valid_out` 3 cycles later, in order.
//
//   Stage 1: sign-select each lane -> 9-bit signed term   (reg)
//   Stage 2: two half-sums of the terms                   (reg)
//   Stage 3: final sum                                    (reg) -> dot
//
// `valid_in` is pipelined to `valid_out` so a streaming consumer can track which
// outputs are live without counting cycles.

`default_nettype none

module ternary_dot_pipe #(
    parameter int unsigned K = 8
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  valid_in,
    input  wire signed [8*K-1:0] a_flat,    // K signed int8 activations
    input  wire        [2*K-1:0] w_flat,    // K ternary 2-bit codes (01=+1,10=-1,00=0)
    output reg                   valid_out,
    output reg  signed [31:0]    dot
);

    // ---- Stage 1: sign-select each lane into a 9-bit signed term (registered) ----
    reg signed [8:0] term [0:K-1];
    reg              v1;
    integer i;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            v1 <= 1'b0;
            for (i = 0; i < K; i = i + 1) term[i] <= '0;
        end else begin
            v1 <= valid_in;
            for (i = 0; i < K; i = i + 1) begin
                logic signed [7:0] a;
                logic        [1:0] c;
                logic signed [8:0] ext;
                a   = $signed(a_flat[8*i +: 8]);
                c   = w_flat[2*i +: 2];
                ext = $signed({a[7], a});               // sign-extend int8 -> 9 bits
                term[i] <= (c == 2'b01) ?  ext :
                           (c == 2'b10) ? -ext : 9'sd0;  // +a / -a / 0
            end
        end
    end

    // ---- Stage 2: two half-sums of the registered terms (registered) ----
    logic signed [31:0] sumL_c, sumR_c;
    always_comb begin
        sumL_c = 32'sd0;
        sumR_c = 32'sd0;
        for (int j = 0;   j < K/2; j++) sumL_c = sumL_c + term[j];
        for (int j = K/2; j < K;   j++) sumR_c = sumR_c + term[j];
    end
    reg signed [31:0] sumL, sumR;
    reg               v2;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin v2 <= 1'b0; sumL <= '0; sumR <= '0; end
        else        begin v2 <= v1;   sumL <= sumL_c; sumR <= sumR_c; end
    end

    // ---- Stage 3: final sum (registered) ----
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin valid_out <= 1'b0; dot <= '0; end
        else        begin valid_out <= v2;   dot <= sumL + sumR; end
    end

endmodule

`default_nettype wire
