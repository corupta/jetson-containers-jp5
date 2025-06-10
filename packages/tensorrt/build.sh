#!/usr/bin/env bash
set -ex

# Let's update nvdla first
# apt update && apt install -y automake autoconf libtool
# ln -s $(which aclocal) /usr/local/bin/aclocal-1.14
# ln -s $(which automake) /usr/local/bin/automake-1.14
# git clone https://github.com/nvdla/sw.git /opt/nvdla
# cd /opt/nvdla/umd/external/protobuf-2.6
# ./configure
# make -j$(nproc)
# cp src/.libs/libprotobuf.a /opt/nvdla/umd/apps/compiler/libprotobuf.a
# cp src/.libs/libprotobuf.a /opt/nvdla/umd/core/src/compiler/libprotobuf.a
# cd /opt/nvdla/umd
# make -j$(nproc)

echo "Building TENSORRT ${TENSORRT_VERSION} (commit=${TENSORRT_COMMIT})"

# cd /usr/lib/aarch64-linux-gnu
# ln -s libnvinfer.so libnvinfer.so.9
# ln -s /usr/lib/aarch64-linux-gnu/libnvinfer.so /usr/lib/aarch64-linux-gnu/libnvinfer.so.10
# ln -s libnvinfer_dispatch.so libnvinfer_dispatch.so.9
# ln -s /usr/lib/aarch64-linux-gnu/libnvinfer_dispatch.so /usr/lib/aarch64-linux-gnu/libnvinfer_dispatch.so.10
# ln -s libnvinfer_lean.so libnvinfer_lean.so.9
# ln -s /usr/lib/aarch64-linux-gnu/libnvinfer_lean.so /usr/lib/aarch64-linux-gnu/libnvinfer_lean.so.10

git clone https://github.com/nvidia/TensorRT ${SOURCE_DIR}
cd ${SOURCE_DIR}
git checkout ${TENSORRT_BRANCH}
git submodule update --init --recursive

# fix nvcc fatal   : Unsupported gpu architecture 'compute_SM'
sed -i 's/list(APPEND CMAKE_CUDA_ARCHITECTURES SM)/list(APPEND CMAKE_CUDA_ARCHITECTURES ${SM})/' CMakeLists.txt

if [ "$TENSORRT_VERSION" = "9.3" ]; then
  # Manually mark it as 9.3.0.1 :)
  sed -i 's/#define NV_TENSORRT_MINOR 2/#define NV_TENSORRT_MINOR 3/' include/NvInferVersion.h
  sed -i 's/#define NV_TENSORRT_BUILD 5/#define NV_TENSORRT_BUILD 1/' include/NvInferVersion.h
  sed -i 's/#define NV_TENSORRT_SONAME_MINOR 2/#define NV_TENSORRT_SONAME_MINOR 3/' include/NvInferVersion.h
  # fix error: identifier "FLT_MAX" is undefined
  sed -i '/#include <cub\/cub.cuh>/a #include <cfloat>' plugin/common/common.cuh
fi

mkdir -p build && cd build
cmake .. -DTRT_LIB_DIR=/usr/lib/aarch64-linux-gnu \
  -DTRT_OUT_DIR=`pwd`/out \
  -DTRT_PLATFORM_ID=aarch64 \
  -DCUDA_VERSION=${USE_CUDA_VERSION} \
  -DCUDNN_VERSION=${USE_CUDNN_VERSION} \
  -DCMAKE_C_COMPILER=$(which gcc) \
  -DCMAKE_CXX_COMPILER=$(which g++) \
  -DGPU_ARCHS="72 87" \
  -DCUDNN_ROOT_DIR=${CUDNN_LIB_PATH} \
  -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_HOME} \
  -DTRT_STATIC=1
  # -D_GLIBCXX_USE_CXX11_ABI=1 \
# -DCMAKE_BUILD_TYPE=Debug
# -DTRT_STATIC=1 is needed despite what cmake tells :)

make TRT_STATIC=1 -j$(nproc)
make install
mv /usr/lib/lib/lib* /usr/lib/aarch64-linux-gnu/
mkdir -p /usr/src/tensorrt/bin/
mv /usr/lib/bin/* /usr/src/tensorrt/bin/
rmdir /usr/lib/lib
rmdir /usr/lib/bin

export EXT_PATH=${SOURCE_DIR}/external
export TRT_OSSPATH=${SOURCE_DIR}
mkdir -p $EXT_PATH && cd $EXT_PATH
git clone https://github.com/pybind/pybind11.git
cd pybind11
git checkout v2.13
cd ..

TRT_PY=$(python3 --version | awk '{print $2}')
TRT_PY_MAJOR=$(echo $TRT_PY | awk -F '.' '{print $1}')
TRT_PY_MINOR=$(echo $TRT_PY | awk -F '.' '{print $2}')
TRT_PY_SHORT=${TRT_PY_MAJOR}.${TRT_PY_MINOR}
wget https://www.python.org/ftp/python/${TRT_PY}/Python-${TRT_PY}.tgz
tar -xvf Python-${TRT_PY}.tgz
mkdir -p $EXT_PATH/python${TRT_PY_SHORT}/include
cp -r Python-${TRT_PY}/Include/* $EXT_PATH/python${TRT_PY_SHORT}/include


cp /usr/include/aarch64-linux-gnu/python${TRT_PY_SHORT}/pyconfig.h $EXT_PATH/python${TRT_PY_SHORT}/include/

cd ${SOURCE_DIR}/python
TENSORRT_MODULE=tensorrt PYTHON_MAJOR_VERSION=${TRT_PY_MAJOR} PYTHON_MINOR_VERSION=${TRT_PY_MINOR} TARGET_ARCHITECTURE=aarch64 ./build.sh
python3 -m pip install ./build/bindings_wheel/dist/tensorrt-*.whl --force-reinstall
