#---
# name: nccl
# group: llm
# config: config.py
# depends: [cuda]
# test: [test.sh]
# requires: '>=35'
# notes: Building it for tensorrt-llm
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NCCL_VERSION \
    NCCL_BRANCH \
    NCCL_SOURCE \
    NCCL_PATCH \
    NVCC_GENCODE \
    FORCE_BUILD="on" \
    SOURCE_DIR="/opt/nccl" \
    TMP_DIR="/tmp/nccl"

COPY build.sh install.sh ${TMP_DIR}/
COPY ${NCCL_PATCH} ${TMP_DIR}/patch.diff

RUN ${TMP_DIR}/install.sh || ${TMP_DIR}/build.sh
