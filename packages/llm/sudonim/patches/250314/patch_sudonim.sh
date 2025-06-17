#!/usr/bin/env bash
set -ex

CUDA_VERSION=$(nvcc -V | grep -oP 'release \K[0-9]+\.[0-9]+')
sed -i \
  "s/version *= *nim.getenv('CUDA_VERSION')/version = nim.getenv('CUDA_VERSION') or '${CUDA_VERSION}'/" \
  /opt/sudonim/sudonim/utils/cuda.py

sed -i \
  "s/env.NUM_GPU = smi.get('attached_gpus', 0)/env.NUM_GPU = smi.get('attached_gpus', 1)/" \
  /opt/sudonim/sudonim/env.py

sed -i \
  's|env.GPU_ARCH = f"sm{CUDA_DEVICES\[0\].cc}" if CUDA_DEVICES else None|env.GPU_ARCH = f"sm{CUDA_DEVICES[0].cc}" if CUDA_DEVICES else "sm72"|' \
  /opt/sudonim/sudonim/env.py