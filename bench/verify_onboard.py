#!/usr/bin/env python3
"""Verify arty_top on real silicon: read its UART stream and check y == 2*c.

arty_top streams ASCII lines "TN<c:2hex><y:4hex>\\n" where y is the ternary
engine's dot product of a runtime counter c against weights summing to +2.
Reading these from the board's USB-UART and confirming y == 2*c proves the
multiply-free engine computes correctly in fabric.

  python bench/verify_onboard.py --port /dev/ttyUSB1 --n 20
"""
import argparse
import re
import sys
import time

try:
    import serial  # pyserial
except ImportError:
    sys.exit("pyserial not installed: pip install pyserial")

PAT = re.compile(rb"TN([0-9A-Fa-f]{2})([0-9A-Fa-f]{4})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB1")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--n", type=int, default=20, help="lines to verify")
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)
    ok = checked = 0
    buf = b""
    deadline = time.time() + args.timeout
    while checked < args.n and time.time() < deadline:
        buf += ser.read(64)
        for m in PAT.finditer(buf):
            c = int(m.group(1), 16)
            y = int(m.group(2), 16)
            exp = 2 * c
            good = (y == exp)
            ok += good
            checked += 1
            print(f"  c={c:3d}  y={y:5d}  expected={exp:5d}  {'ok' if good else 'MISMATCH'}")
            if checked >= args.n:
                break
        buf = buf[-8:]  # keep a tail for partial matches across reads
    ser.close()

    print(f"verified {ok}/{checked} lines (y == 2*c)")
    sys.exit(0 if checked > 0 and ok == checked else 1)


if __name__ == "__main__":
    main()
