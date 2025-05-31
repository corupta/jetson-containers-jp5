#---
# name: lmdeploy
# group: llm
# config: config.py
# depends: [transformers, pybind11]
# requires: '>=35'
# test: [test.py, test.sh]
# docs: docs.md
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TVM_VERSION \
    LMDEPLOY_VERSION \
    LMDEPLOY_COMMIT \
    LMDEPLOY_PATCH \
    FORCE_BUILD=on \
    SOURCE_DIR=/opt/lmdeploy \
    TMP_DIR=/tmp/lmdeploy

ENV LMDEPLOY_SOURCE_DIR=${SOURCE_DIR}
    
COPY build.sh install.sh ${TMP_DIR}/
COPY ${LMDEPLOY_PATCH} ${TMP_DIR}/patch.diff

RUN ${TMP_DIR}/install.sh || ${TMP_DIR}/build.sh || echo "BUILD FAILED (lmdeploy ${lmdeploy_VERSION} ${lmdeploy_COMMIT})"
