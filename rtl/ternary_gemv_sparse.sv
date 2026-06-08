// ternary_gemv_sparse — activation-sparse ternary matrix-vector multiply.
//
// This is Direction D's lever: in a decode FFN, 85-95% of intermediate neurons
// are zero *this token*, so their weight rows contribute nothing and need never
// be fetched. Given a per-row `active_mask`, this engine GATHERS only the active
// rows from weight memory (skipping inactive ones entirely — no address issued,
// no data fetched), computes their ternary dot with x, and reports how many rows
// it actually fetched. Because batch-1 decode is DDR3-bandwidth-bound, *not
// fetching* the inactive rows is the win — the metric is `rows_fetched`, which a
// dense engine (or a GPU) cannot reduce for unstructured, per-token sparsity.
//
// Memory model: a synchronous-read weight port. Assert mem_addr + mem_ren in
// cycle T; mem_rdata is valid in cycle T+1 (like a BRAM / a DDR3 burst return).
// Inactive rows leave y[m] = 0. Results read back through rd_addr/rd_data.

`default_nettype none

module ternary_gemv_sparse #(
    parameter int unsigned K  = 8,
    parameter int unsigned M  = 16,
    parameter int unsigned RW = (M <= 1) ? 1 : $clog2(M)
) (
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   start,        // pulse: latch x + active_mask, begin
    input  wire signed [8*K-1:0]  x_flat,
    input  wire        [M-1:0]    active_mask,  // bit m = 1 -> fetch+compute row m

    // synchronous weight-memory read port (data valid 1 cycle after ren)
    output logic       [RW-1:0]   mem_addr,
    output logic                  mem_ren,
    input  wire        [2*K-1:0]  mem_rdata,

    // result read-back
    input  wire        [RW-1:0]   rd_addr,
    output wire signed [31:0]     rd_data,

    output reg                    done,         // high when the scan is complete
    output reg         [31:0]     rows_fetched  // # active rows fetched (the saving metric)
);

    localparam logic [1:0] S_IDLE = 2'd0, S_SCAN = 2'd1, S_DATA = 2'd2, S_DONE = 2'd3;

    reg [1:0]            state;
    reg signed [8*K-1:0] x_reg;
    reg [M-1:0]          mask_reg;
    reg [31:0]           scan_idx;     // current row under inspection
    reg [RW-1:0]         pend_addr;    // row whose data is in flight
    reg signed [31:0]    y_mem [0:M-1];
    integer i;

    // Multiply-free dot of x_reg against the just-fetched weight row.
    wire signed [31:0] row_dot;
    ternary_dot #(.K(K)) u_dot (.a_flat(x_reg), .w_flat(mem_rdata), .dot(row_dot));

    assign rd_data = y_mem[rd_addr];

    // Combinational memory request: only issued for active rows during S_SCAN.
    always_comb begin
        mem_addr = scan_idx[RW-1:0];
        mem_ren  = (state == S_SCAN) && (scan_idx < M) && mask_reg[scan_idx[RW-1:0]];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; done <= 1'b0; scan_idx <= 32'd0; rows_fetched <= 32'd0;
            x_reg <= '0; mask_reg <= '0; pend_addr <= '0;
            for (i = 0; i < M; i = i + 1) y_mem[i] <= 32'sd0;
        end else if (start) begin
            // synchronous (re)start from any state
            x_reg <= x_flat; mask_reg <= active_mask;
            scan_idx <= 32'd0; rows_fetched <= 32'd0; done <= 1'b0;
            for (i = 0; i < M; i = i + 1) y_mem[i] <= 32'sd0;
            state <= S_SCAN;
        end else begin
            case (state)
                S_SCAN: begin
                    if (scan_idx >= M) begin
                        state <= S_DONE;
                    end else if (mask_reg[scan_idx[RW-1:0]]) begin
                        pend_addr <= scan_idx[RW-1:0];   // request issued combinationally this cycle
                        state     <= S_DATA;
                    end else begin
                        scan_idx <= scan_idx + 32'd1;    // inactive: skip, no fetch
                    end
                end
                S_DATA: begin
                    y_mem[pend_addr] <= row_dot;          // mem_rdata valid now
                    rows_fetched     <= rows_fetched + 32'd1;
                    scan_idx         <= scan_idx + 32'd1;
                    state            <= S_SCAN;
                end
                S_DONE:  done <= 1'b1;
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule

`default_nettype wire
