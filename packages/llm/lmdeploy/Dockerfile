#---
# name: lmdeploy
# group: llm
# config: config.py
# depends: [transformers, pybind11, flash-attention]
# requires: '>=36'
# test: [test.py, test.sh]
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TVM_VERSION \
    LMDEPLOY_VERSION \
    LMDEPLOY_COMMIT \
    LMDEPLOY_FORK \
    LMDEPLOY_PATCH \
    FORCE_BUILD=on \
    SOURCE_DIR=/opt/lmdeploy \
    TMP_DIR=/tmp/lmdeploy

ENV LMDEPLOY_SOURCE_DIR=${SOURCE_DIR}
    
COPY build.sh install.sh ${TMP_DIR}/
COPY ${LMDEPLOY_PATCH} ${TMP_DIR}/patch.diff

RUN ${TMP_DIR}/install.sh || ${TMP_DIR}/build.sh || echo "BUILD FAILED (lmdeploy ${LMDEPLOY_FORK} ${LMDEPLOY_VERSION} ${LMDEPLOY_COMMIT})"
