#!/bin/bash
# Build libanima_rt.so for x86_64 Linux (WSL host) for quick verification.
# Usage: bash build_host.sh
set -e

SRC="src/anima_tensor.cpp src/cpu_backend.cpp src/anima_rt.cpp"
OUT="libanima_rt_host.so"

echo "Building $OUT for x86_64 (host verification)..."
g++ -O2 -std=c++17 -fPIC -shared \
    -Iinclude \
    -o "$OUT" \
    $SRC \
    -lm

echo "[OK] $OUT built: $(du -h $OUT | cut -f1)"
