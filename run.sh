#!/usr/bin/env bash
set -euo pipefail

make
python3 generate_input.py --width 1024 --height 768 --output input.ppm
./cuda_image_capstone --input input.ppm --output result --block 16
python3 generate_input.py --width 1920 --height 1080 --output input_large.ppm
./cuda_image_capstone --input input_large.ppm --output result_large --block 16
