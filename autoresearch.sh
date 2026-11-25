#!/usr/bin/env bash
set -euo pipefail
cd /home/han/pyx/HMI_clean
exec bash scripts_run/train_loaded_4d_vxh.sh "$@"
