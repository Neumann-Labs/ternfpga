// ternary_gemv_bench — on-silicon throughput harness for ternary_gemv_stream.
//
// Wraps the engine with a weight BRAM (w_mem) + a replay FSM + a hardware cycle
// counter, so the engine can run at its true 1-tile/cycle rate from resident
// weights (no CPU/CSR feed bottleneck) and the EXACT compute-cycle count is
// captured on silicon. The host loads w_mem (+ x) once, pulses `run`, and reads
// `cycle_count` back — measured engine throughput. Energy/token then follows from
// cycles/freq x report_power.
//
// (At K<=16 the engine consumes < DDR3 bandwidth, so it is compute-bound: this
// measured engine rate IS the on-board throughput; a DDR3-streaming feed would
// not change it. The bandwidth-bound regime needs K>~28 — future work.)

`default_nettype none

module ternary_gemv_bench #(
    parameter int unsigned K      = 8,
    parameter int unsigned NT_MAX = 64,
    parameter int unsigned M_MAX  = 64,
    parameter int unsigned WDEPTH = 4096,                       // max M*NT tiles
    parameter int unsigned AW = (NT_MAX <= 1) ? 1 : $clog2(NT_MAX),
    parameter int unsigned MW = (M_MAX  <= 1) ? 1 : $clog2(M_MAX),
    parameter int unsigned WW = (WDEPTH <= 1) ? 1 : $clog2(WDEPTH)
) (
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire [AW-1:0]         nt,
    input  wire [MW-1:0]         m_rows,
    // activation load (passthrough to the engine)
    input  wire                  x_we,
    input  wire [AW-1:0]         x_waddr,
    input  wire signed [8*K-1:0] x_wdata,
    // weight-memory load (M*NT tiles, row-major)
    input  wire                  wm_we,
    input  wire [WW-1:0]         wm_waddr,
    input  wire [2*K-1:0]        wm_wdata,
    // run + readback
    input  wire                  run,            // pulse: start engine + replay w_mem at 1 tile/cycle
    input  wire [MW-1:0]         rd_addr,
    output wire signed [31:0]    rd_data,
    output reg                   done,
    output reg  [31:0]           cycle_count     // measured compute cycles (run -> engine done)
);
    // ---- weight memory (clock-only write -> BRAM) ----
    reg [2*K-1:0] w_mem [0:WDEPTH-1];
    always_ff @(posedge clk) begin
        if (wm_we) w_mem[wm_waddr] <= wm_wdata;
    end

    // ---- replay FSM state ----
    reg          replaying;
    reg [WW-1:0] ridx;          // next tile to read
    reg [31:0]   total;         // nt*m_rows tiles
    reg [31:0]   cyc;
    reg [2*K-1:0] w_tile_r;     // registered w_mem read -> engine
    reg          w_valid_r;

    wire eng_done;

    ternary_gemv_stream #(.K(K), .NT_MAX(NT_MAX), .M_MAX(M_MAX)) u_eng (
        .clk(clk), .rst_n(rst_n),
        .nt(nt), .m_rows(m_rows),
        .x_we(x_we), .x_waddr(x_waddr), .x_wdata(x_wdata),
        .start(run),
        .w_valid(w_valid_r), .w_tile(w_tile_r),
        .rd_addr(rd_addr), .rd_data(rd_data),
        .done(eng_done)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            replaying <= 1'b0; ridx <= '0; total <= 32'd0; cyc <= 32'd0;
            w_tile_r <= '0; w_valid_r <= 1'b0; done <= 1'b0; cycle_count <= 32'd0;
        end else if (run) begin
            replaying <= 1'b1; ridx <= '0; total <= nt * m_rows;
            cyc <= 32'd0; done <= 1'b0; w_valid_r <= 1'b0;
        end else begin
            // sequential w_mem read with a 1-cycle-aligned valid (matches the BRAM read latency)
            w_tile_r  <= w_mem[ridx];
            w_valid_r <= replaying && (ridx < total);
            if (replaying && (ridx < total)) ridx <= ridx + 1'b1;
            if (replaying) cyc <= cyc + 32'd1;
            if (replaying && eng_done) begin
                replaying   <= 1'b0;
                done        <= 1'b1;
                cycle_count <= cyc;       // run -> done, the measured compute latency
            end
        end
    end
endmodule

`default_nettype wire
