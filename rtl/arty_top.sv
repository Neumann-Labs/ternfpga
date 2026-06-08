// arty_top — on-board demo: the ternary engine computing on real silicon,
// reporting over UART so the result is verifiable from the host.
//
// A runtime counter `c` (0..99) drives all 8 lanes of a ternary_dot whose
// weights sum to +2 (w = 16'hAA55: five +1, three -1), so y = 2*c. Because `c`
// is a runtime value, the dot is NOT constant-folded — the sign-select + adder
// actually run in fabric. Each result is streamed as an ASCII line
//   "TN" <c:2 hex> <y:4 hex> "\n"
// at 115200 8N1, and the 4 LEDs show a heartbeat. The host reads the serial
// port and checks y == 2*c, closing the author->synth->flash->observe loop.

`default_nettype none

module arty_top (
    input  wire       CLK100MHZ,
    output wire       uart_tx_o,
    output wire [3:0] led
);
    wire clk = CLK100MHZ;

    // ---- Power-on reset: hold reset for 256 clocks after configuration ----
    reg [7:0] por = 8'd0;
    always_ff @(posedge clk) if (!(&por)) por <= por + 8'd1;
    wire rst_n = &por;

    // ---- Runtime input + multiply-free compute: y = dot({8{c}}, w) = 2*c ----
    reg [7:0] c = 8'd0;                       // 0..99, advanced per message
    localparam [15:0] W_CONST = 16'hAA55;     // lanes: +1 +1 +1 +1 +1 -1 -1 -1 -> sum +2
    wire signed [31:0] y;
    ternary_dot #(.K(8)) u_dot (.a_flat({8{c}}), .w_flat(W_CONST), .dot(y));
    wire [15:0] y16 = y[15:0];

    // ---- UART transmitter ----
    reg        u_start;
    reg  [7:0] u_data;
    wire       u_busy;
    uart_tx #(.CLKS_PER_BIT(868)) u_uart (    // 100 MHz / 115200
        .clk(clk), .rst_n(rst_n), .start(u_start), .data(u_data),
        .tx(uart_tx_o), .busy(u_busy)
    );

    function automatic [7:0] hex(input [3:0] n);
        hex = (n < 4'd10) ? (8'd48 + {4'd0, n}) : (8'd55 + {4'd0, n});  // '0'+n / 'A'+(n-10)
    endfunction

    // 9-byte message: 'T' 'N' c[hi] c[lo] y[15:12] y[11:8] y[7:4] y[3:0] '\n'
    reg  [3:0] idx;
    reg  [7:0] cur_byte;
    always_comb begin
        case (idx)
            4'd0:    cur_byte = 8'h54;            // 'T'
            4'd1:    cur_byte = 8'h4E;            // 'N'
            4'd2:    cur_byte = hex(c[7:4]);
            4'd3:    cur_byte = hex(c[3:0]);
            4'd4:    cur_byte = hex(y16[15:12]);
            4'd5:    cur_byte = hex(y16[11:8]);
            4'd6:    cur_byte = hex(y16[7:4]);
            4'd7:    cur_byte = hex(y16[3:0]);
            default: cur_byte = 8'h0A;            // '\n'
        endcase
    end

    // Sender FSM: issue a byte, wait for the transmitter to take it, wait for done.
    localparam [1:0] S_ISSUE = 2'd0, S_ACK = 2'd1, S_DONE = 2'd2;
    reg [1:0] st;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st <= S_ISSUE; idx <= 4'd0; c <= 8'd0; u_start <= 1'b0; u_data <= 8'd0;
        end else begin
            u_start <= 1'b0;
            case (st)
                S_ISSUE: if (!u_busy) begin
                    u_data  <= cur_byte; u_start <= 1'b1; st <= S_ACK;
                end
                S_ACK:  if (u_busy) st <= S_DONE;
                S_DONE: if (!u_busy) begin
                    if (idx == 4'd8) begin
                        idx <= 4'd0;
                        c   <= (c == 8'd99) ? 8'd0 : c + 8'd1;
                    end else begin
                        idx <= idx + 4'd1;
                    end
                    st <= S_ISSUE;
                end
                default: st <= S_ISSUE;
            endcase
        end
    end

    // ---- LEDs: heartbeat + status ----
    reg [25:0] hb;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) hb <= 26'd0; else hb <= hb + 26'd1;
    end
    assign led = {rst_n, u_busy, hb[25], hb[24]};
endmodule

`default_nettype wire
