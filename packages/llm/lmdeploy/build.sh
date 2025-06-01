#!/usr/bin/env bash
set -ex

echo "Building LMDEPLOY ${LMDEPLOY_VERSION} (commit=${LMDEPLOY_COMMIT})"

git clone https://github.com/${LMDEPLOY_FORK}/lmdeploy ${SOURCE_DIR}
cd ${SOURCE_DIR}
git checkout ${LMDEPLOY_COMMIT}
# git submodule update --init --recursive

# apply patches to the source
if [ -s /tmp/lmdeploy/patch.diff ]; then 
	git apply /tmp/lmdeploy/patch.diff 
fi

sed -i "s/^__version__ *= *['\"].*['\"]/__version__ = '${LMDEPLOY_VERSION}'/" lmdeploy/version.py

mkdir -p build && cd build
bash ../generate.sh make
make -j$(nproc) && make install
cd ..
pip install -e .