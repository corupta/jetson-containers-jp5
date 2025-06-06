#!/usr/bin/env bash
set -ex

echo "Building TENSORRT ${TENSORRT_VERSION} (commit=${TENSORRT_COMMIT})"

git clone https://github.com/nvidia/TensorRT ${SOURCE_DIR}
cd ${SOURCE_DIR}
git checkout ${TENSORRT_BRANCH}
git submodule update --init --recursive

mkdir -p build && cd build
cmake .. -DTRT_LIB_DIR=/usr/lib/aarch64-linux-gnu \
  -DTRT_OUT_DIR=`pwd`/out \
  -DTRT_PLATFORM_ID=aarch64 \
  -DCUDA_VERSION=${USE_CUDA_VERSION} \
  -DCUDNN_VERSION=${USE_CUDNN_VERSION} \
  -DCMAKE_C_COMPILER=$(which gcc) \
  -DCMAKE_CXX_COMPILER=$(which g++) \
  -DGPU_ARCHS="72 87"
make -j$(nproc)
