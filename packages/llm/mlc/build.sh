#!/usr/bin/env bash
set -ex

echo "Building MLC ${MLC_VERSION} (commit=${MLC_COMMIT})"

# could NOT find zstd (missing: zstd_LIBRARY zstd_INCLUDE_DIR)
apt-get update
apt-get install -y --no-install-recommends libzstd-dev ccache
rm -rf /var/lib/apt/lists/*
apt-get clean
	
# clone the sources
git clone https://github.com/mlc-ai/mlc-llm ${SOURCE_DIR}
cd ${SOURCE_DIR}
git checkout ${MLC_COMMIT}
git submodule update --init --recursive
    
# apply patches to the source
if [ -s /tmp/mlc/patch.diff ]; then 
  cuda_archs=$(echo "$CUDA_ARCHITECTURES" | sed "s|;| |g")
	# sed -i "s|SUPPORTED 87|SUPPORTED ${cuda_archs}|g" /tmp/mlc/patch.diff 

	git apply /tmp/mlc/patch.diff 
fi
sed -i 's|flashinfer-python==0.2.5|flashinfer-python|' ${SOURCE_DIR}/python/setup.py

# cd ${SOURCE_DIR}/3rdparty/tvm/3rdparty/flashinfer
# git restore .
# git checkout 1605eaab
# git submodule update --init --recursive
# git config user.email "jetpack@containers.com"
# git config user.name "Jetpack Containers"
# git cherry-pick 3a69560
# # aka 0.2.1.post2
# git apply /tmp/flashinfer/patch.diff
# sed -i 's|options={.*| |g' setup.py
# sed -i 's|if arch < 75|if arch < 72|' setup.py
# perl -0777 -i -pe 's/-DDMLC_USE_LOGGING_LIBRARY=\s*/-DDMLC_USE_LOGGING_LIBRARY=/g' CMakeLists.txt
# cat CMakeLists.txt | grep DDMLC_USE_LOGGING_LIBRARY
# cd ${SOURCE_DIR}

git status
# git diff --submodule=diff

# add extras to the source
cp /tmp/mlc/benchmark.py ${SOURCE_DIR}/

# flashinfer build references 'python'
ln -sf /usr/bin/python3 /usr/bin/python

# disable pytorch: https://github.com/apache/tvm/issues/9362
# -DUSE_LIBTORCH=$(pip3 show torch | grep Location: | cut -d' ' -f2)/torch
mkdir build
cd build

cmake -G Ninja \
	-DCMAKE_CXX_STANDARD=17 \
	-DCMAKE_CUDA_STANDARD=17 \
	-DCMAKE_CUDA_ARCHITECTURES=${CUDAARCHS} \
	-DUSE_CUDA=ON \
	-DUSE_CUDNN=ON \
	-DUSE_CUBLAS=ON \
	-DUSE_CURAND=ON \
	-DUSE_CUTLASS=ON \
	-DUSE_FLASHINFER=ON \
	-DUSE_THRUST=ON \
	-DUSE_GRAPH_EXECUTOR=ON \
	-DUSE_GRAPH_EXECUTOR_CUDA_GRAPH=ON \
	-DUSE_STACKVM_RUNTIME=ON \
	-DUSE_CCACHE=ON \
	-DUSE_LLVM="/usr/bin/llvm-config --link-static" \
	-DHIDE_PRIVATE_SYMBOLS=ON \
	-DSUMMARIZE=ON \
	../
	
ninja

#rm -rf CMakeFiles tvm/CMakeFiles
#rm -rf tokenizers/CMakeFiles tokenizers/release

# build TVM python module
cd ${SOURCE_DIR}/3rdparty/tvm/python

TVM_LIBRARY_PATH=${SOURCE_DIR}/build/tvm \
python3 setup.py --verbose bdist_wheel --dist-dir /opt

pip3 install /opt/tvm*.whl
#pip3 show tvm && python3 -c 'import tvm'
#rm -rf dist build

# build mlc-llm python module
cd ${SOURCE_DIR}

if [ -f setup.py ]; then
	python3 setup.py --verbose bdist_wheel --dist-dir /opt
fi

cd python
python3 setup.py --verbose bdist_wheel --dist-dir /opt

pip3 install /opt/mlc*.whl

# make sure it loads
#cd /
#pip3 show mlc_llm
#python3 -m mlc_llm.build --help
#python3 -c "from mlc_chat import ChatModule; print(ChatModule)"
    
# make the CUTLASS sources available for model builder
ln -s ${SOURCE_DIR}/3rdparty/tvm/3rdparty "$(pip3 show tvm | awk '/Location:/ {print $2}')/tvm/3rdparty"

# upload wheels
twine upload --verbose /opt/tvm*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/mlc_llm*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
twine upload --verbose /opt/mlc_chat*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
