#!/usr/bin/env bash
set -ex

echo "Building MLX ${MLX_VERSION} (commit=${MLX_COMMIT})"

apt update
apt install -y liblapacke-dev

git clone https://github.com/${MLX_FORK} ${SOURCE_DIR}
cd ${SOURCE_DIR}
git checkout ${MLX_COMMIT}
# git submodule update --init --recursive

# apply patches to the source
if [ -s /tmp/mlx/patch.diff ]; then 
	git apply /tmp/mlx/patch.diff 
fi

CMAKE_BUILD_PARALLEL_LEVEL=10 CMAKE_ARGS="-DMLX_BUILD_CUDA=on -DMLX_BUILD_EXAMPLES=ON -DMLX_FAST_COMPILE=ON -DMLX_CUDA_ARCHITECTURES=native" pip install .

python setup.py generate_stubs

pip install mlx-lm
