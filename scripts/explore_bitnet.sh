#!/bin/bash
set -e
git clone --recursive --depth 1 https://github.com/microsoft/BitNet.git /tmp/bitnet_explore
echo "--- ALL HEADERS ---"
find /tmp/bitnet_explore -name "*.h" | grep "bitnet-lut-kernels.h"
echo "--- ALL DIRECTORIES ---"
find /tmp/bitnet_explore -maxdepth 5 -type d | grep "llama.cpp"
rm -rf /tmp/bitnet_explore
