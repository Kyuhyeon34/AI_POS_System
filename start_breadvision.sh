#!/bin/bash
set -e
cd "$(dirname "$0")"

export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

python3 main.py
