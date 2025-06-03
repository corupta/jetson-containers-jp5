#---
# name: mlx
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
    MLX_VERSION \
    MLX_COMMIT \
    MLX_FORK \
    MLX_PATCH \
    FORCE_BUILD=on \
    SOURCE_DIR=/opt/mlx \
    TMP_DIR=/tmp/mlx

ENV MLX_SOURCE_DIR=${SOURCE_DIR}
    
COPY build.sh install.sh ${TMP_DIR}/
COPY ${MLX_PATCH} ${TMP_DIR}/patch.diff

RUN ${TMP_DIR}/install.sh || ${TMP_DIR}/build.sh || echo "BUILD FAILED (mlx ${MLX_FORK} ${MLX_VERSION} ${MLX_COMMIT})"
