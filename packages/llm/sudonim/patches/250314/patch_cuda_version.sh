#!/usr/bin/env bash
set -ex

CUDA_VERSION=$(nvcc -V | grep -oP 'release \K[0-9]+\.[0-9]+')
sed -i \
  "s/version *= *nim.getenv('CUDA_VERSION')/version = nim.getenv('CUDA_VERSION') or '${CUDA_VERSION}'/" \
  /opt/sudonim/sudonim/utils/cuda.py
