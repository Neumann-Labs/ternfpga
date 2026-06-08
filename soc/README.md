# soc/ — ternary engine in a LiteX RISC-V SoC

The ternary engine integrated as a CPU-controlled CSR peripheral in a LiteX SoC
(VexRiscv + LiteDRAM DDR3) on the Arty A7-35T. The CPU drives a GEMV on the engine
and reads the result back — verified bit-exact on silicon
([`../bench/results/onboard_soc_gemv.md`](../bench/results/onboard_soc_gemv.md)).

- **`ternary_tile_csr.py`** — LiteX/Migen wrapper exposing `ternary_tile` via CSRs
  (`x`, `wbyte`, `ctl`/start, `status`/done, `rd_addr`/`rd_data`).
- **`ternary_arty.py`** — custom Arty A7-35T SoC target (engine + LiteDRAM + BIOS).
  `K=8` so `x` is a clean 64-bit CSR.
- **`firmware/main.c`** — VexRiscv firmware: drive a GEMV, verify vs the golden,
  print `TERNARY_ONBOARD_PASS`.
- **`firmware/gen_testvec.py`** — emit the C test vector (same golden + base-3
  packing as the RTL tests).

## Build + run (worker4)
Prereqs: LiteX + `riscv64-unknown-elf-gcc` + `meson`/`ninja` (see `../syn/litex_arty.sh`).
```bash
cd /srv/fpga/litex-build
RTL_DIR=<repo>/rtl python <repo>/soc/ternary_arty.py --run        # build SoC bitstream (~30 min)
openFPGALoader -b arty_a7_35t build_ternary/gateware/ternary_arty.bit

litex_bare_metal_demo --build-path=$PWD/build_ternary             # firmware skeleton
python <repo>/soc/firmware/gen_testvec.py > demo/testvec.h
cp <repo>/soc/firmware/main.c demo/main.c
make -C demo BUILD_DIR=$PWD/build_ternary
script -qfc "litex_term /dev/ttyUSB1 --kernel=demo/demo.bin" /tmp/t.log   # serialboot (needs a pty)
# -> TERNARY_ONBOARD_PASS (16 rows bit-exact vs golden)
```

Note: `litex_term` needs a real TTY — wrap it in `script` when running non-interactively.
