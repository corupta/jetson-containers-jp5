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

mkdir -p build && cd build
cmake .. -DMLX_BUILD_CUDA=ON -DMLX_BUILD_EXAMPLES=ON
make -j$(nproc)
make test
make install

CMAKE_BUILD_PARALLEL_LEVEL=8 pip install -e .

python setup.py generate_stubs

pip install mlx-lm