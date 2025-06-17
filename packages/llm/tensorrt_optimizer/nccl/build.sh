#!/usr/bin/env bash
set -ex

echo "cloning nccl sources from git (branch=${NCCL_BRANCH})"
git clone https://github.com/NVIDIA/nccl.git ${SOURCE_DIR}
cd ${SOURCE_DIR}
git checkout ${NCCL_BRANCH}
git status

if [ -s ${SOURCE_DIR}/patch.diff ]; then 
  echo "applying git patches from ${NCCL_PATCH}"
  git apply ${SOURCE_DIR}/patch.diff
fi

git status
make -j src.build NVCC_GENCODE="${NVCC_GENCODE}"

# Deb Build
# apt-get update && \
#     apt-get install -y --no-install-recommends \
#         build-essential \
#         devscripts \
#         debhelper \
#         fakeroot && \
#     rm -rf /var/lib/apt/lists/* && \
#     apt-get clean
# make pkg.debian.build
# ls build/pkg/deb/

# Tar Build
# make pkg.txz.build
# ls build/pkg/txz/

make install

# echo "Installed NCCL, Building Tests"
# git clone https://github.com/NVIDIA/nccl-tests.git
# cd nccl-tests
# make -j

