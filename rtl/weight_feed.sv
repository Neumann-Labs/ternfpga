// weight_feed — DDR3-style byte burst -> ternary w_row stream.
//
// Bridges memory to the engine: accepts one dense base-3 weight byte per cycle
// (the on-DDR3 layout, 5 ternary weights/byte), accumulates BPR = ceil(K/5) bytes
// into a row, unpacks them through a ternary_unpack5 array, and emits the row's
// 2-bit lane codes with a one-cycle `row_valid`. That `{w_row, row_valid}` is
// exactly what ternary_gemv / ternary_gemv_pipe consume — so the streaming engine
// can be driven straight from a DDR3 burst. The first concrete piece of the
// Phase-1 memory datapath. (Parameterized for BPR >= 2, i.e. K >= 6.)

`default_nettype none

module weight_feed #(
    parameter int unsigned K   = 10,
    parameter int unsigned BPR = (K + 4) / 5
) (
    input  wire           clk,
    input  wire           rst_n,
    input  wire [7:0]     byte_in,
    input  wire           byte_valid,
    output wire [2*K-1:0] w_row,
    output reg            row_valid
);
    localparam int unsigned NCODE = BPR * 5;

    reg [8*BPR-1:0] row_bytes;   // bytes shift in; byte 0 ends up at the LSB
    reg [7:0]       cnt;

    // Unpack the BPR buffered bytes -> NCODE lane codes; the row uses the first K.
    wire [2*NCODE-1:0] codes;
    genvar b;
    generate
        for (b = 0; b < BPR; b++) begin : g_unp
            ternary_unpack5 u (.byte_in(row_bytes[8*b +: 8]), .codes_out(codes[10*b +: 10]));
        end
    endgenerate
    assign w_row = codes[2*K-1:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            row_bytes <= '0; cnt <= 8'd0; row_valid <= 1'b0;
        end else begin
            row_valid <= 1'b0;
            if (byte_valid) begin
                row_bytes <= {byte_in, row_bytes[8*BPR-1:8]};   // byte 0 -> LSB after BPR shifts
                if (cnt == BPR - 1) begin
                    cnt       <= 8'd0;
                    row_valid <= 1'b1;                          // full row available next cycle
                end else begin
                    cnt <= cnt + 8'd1;
                end
            end
        end
    end
endmodule

`default_nettype wire
