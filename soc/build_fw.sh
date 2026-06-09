#!/usr/bin/env bash
# Build a ternfpga firmware against an existing LiteX build, linking the soft-float
# helpers (libgcc) + math (libm) that the bare-metal demo omits — our transformer glue
# uses double-precision soft-float + cos/sin/exp/sqrt/lrint.
#
#   soc/build_fw.sh <build_dir> <main.c> [extra_header ...]
#
# Produces <build_dir>/../demo/demo.bin (the litex_term --kernel image).
set -euo pipefail
BUILD="$1"; MAINC="$2"; shift 2
cd /srv/fpga/litex-build
rm -rf demo
litex_bare_metal_demo --build-path="$BUILD" >/dev/null 2>&1
cp "$MAINC" demo/main.c
for h in "$@"; do cp "$h" demo/; done

# Append the correct-multilib libgcc.a (soft-float __truncdfsf2/__extendsfdf2 etc.) to the link
# group — resolved as a LITERAL absolute path in bash (the recipe's $(CPUFLAGS) is empty, so a
# make-time $(shell) picks the wrong-arch rv64 libgcc). Math comes from fw_mathf.h (no libm here).
CPUFLAGS=$(grep -oP '(?<=^CPUFLAGS=).*' "$BUILD/software/include/generated/variables.mak")
LIBGCC=$(riscv64-unknown-elf-gcc $CPUFLAGS -print-file-name=libgcc.a)
echo "link libgcc: $LIBGCC"
python3 - "$LIBGCC" <<'PY'
import re, sys
libgcc = sys.argv[1]
p = "demo/Makefile"
s = open(p).read()
s2 = re.sub(r"(\$\(REGULAR_LIBS:lib%=-l%\)) \\", r"\1 " + libgcc + r" \\", s, count=1)
assert s2 != s, "link line not found / already patched"
open(p, "w").write(s2)
PY

make -C demo BUILD_DIR="$BUILD" 2>&1 | tail -5
ls -la demo/demo.bin
