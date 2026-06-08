# Contributing & Engineering Practices

This repo follows a few hard rules. They exist because the whole value of the project is **rigor + honest measurement**, not a flashy demo.

## 1. Test-Driven Development (non-negotiable)
- **The cocotb test lands before (or with) the RTL it tests.** A PR that adds RTL without a test that exercises it does not merge.
- Every hardware block has a **bit-exact golden reference** in Python (NumPy). The test asserts the RTL matches the golden model exactly (ternary/integer math → exact equality, not tolerance).
- Run the suite locally before pushing: `make -C sim` (runs verilator + cocotb). CI runs the same on every PR; **red CI does not merge.**

## 2. Benchmark early and often
- A number you haven't measured is a guess. We commit a **benchmark result the moment a block runs** — even a partial one — into `bench/results/` with the date, commit, and setup.
- The CPU baseline (`bitnet.cpp`) and the energy-measurement harness exist **before** the FPGA does, so we always have something to compare against.
- Headline metric: **energy-per-token (J/tok)**, measured board-to-board. Throughput is reported honestly (we expect to lose it).

## 3. Git hygiene
- **Conventional Commits**: `type(scope): summary` — e.g. `feat(rtl): ternary PE sign-select array`, `test(sim): bit-exact GEMV vs numpy`, `bench: first DDR3 sustained-BW number`.
- **Short-lived feature branches → PR → squash-merge to `main`.** `main` stays green and releasable.
- Small, atomic commits with a clear "why" in the body. No "wip"/"fix" soup on `main`.
- Commits are authored by the maintainers (human-attributed); no tooling co-authors.

## 4. The build log
- [`BUILDLOG.md`](BUILDLOG.md) is a **dated, append-only narrative**: decisions, measured numbers, dead-ends, surprises. It is the source material for the eventual write-up — keep it honest and specific.

## Dev environment
- **Author** code here; **build/sim/flash on `worker4`** (Vivado 2025.2 + verilator + cocotb + the Arty A7-35T).
- `tools/sync_worker4.sh` rsyncs the repo to `worker4:/srv/fpga/ternfpga` and runs the requested target (sim / synth / flash) there, streaming results back.
- Pure-simulation tests also run in CI (GitHub Actions: verilator + cocotb). Synthesis and on-board runs are worker4-only.

## What "done" means for a block
RTL written ✅ · cocotb test bit-exact vs golden ✅ · Vivado synth (utilization + Fmax, DSP-free multiply path confirmed) ✅ · a benchmark/measurement logged ✅ · `BUILDLOG.md` updated ✅.
