// uart_tx — minimal 8N1 UART transmitter (idle-high, LSB-first, 1 stop bit).
//
// The project's first observable I/O: it lets a bitstream on the Arty report a
// computed result over the USB-UART so we can verify on-silicon behaviour from
// the host (no logic analyzer needed). Also the seed of the eventual token I/O.
//
// CLKS_PER_BIT = clk_hz / baud (e.g. 100 MHz / 115200 = 868). Pulse `start` with
// `data` valid while `busy` is low; the frame is start(0) + 8 data + stop(1).

`default_nettype none

module uart_tx #(
    parameter int unsigned CLKS_PER_BIT = 868
) (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       start,     // pulse high (1 clk) to begin a byte
    input  wire [7:0] data,
    output reg        tx,        // serial line, idle high
    output reg        busy
);
    localparam int unsigned CW = (CLKS_PER_BIT <= 1) ? 1 : $clog2(CLKS_PER_BIT);

    reg [CW-1:0] clk_cnt;
    reg [3:0]    bit_idx;   // 0=start, 1..8=data LSB-first, 9=stop
    reg [9:0]    frame;     // {stop=1, data[7:0], start=0}; shifted out LSB first

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tx <= 1'b1; busy <= 1'b0; clk_cnt <= '0; bit_idx <= 4'd0; frame <= 10'h3FF;
        end else if (!busy) begin
            tx <= 1'b1;                       // idle high
            if (start) begin
                frame   <= {1'b1, data, 1'b0};
                busy    <= 1'b1;
                clk_cnt <= '0;
                bit_idx <= 4'd0;
            end
        end else begin
            tx <= frame[bit_idx];             // hold the current bit
            if (clk_cnt == CW'(CLKS_PER_BIT - 1)) begin
                clk_cnt <= '0;
                if (bit_idx == 4'd9) begin    // stop bit just finished
                    busy <= 1'b0;
                    tx   <= 1'b1;
                end else begin
                    bit_idx <= bit_idx + 4'd1;
                end
            end else begin
                clk_cnt <= clk_cnt + 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
