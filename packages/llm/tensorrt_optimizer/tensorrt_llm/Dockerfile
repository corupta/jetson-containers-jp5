#---
# name: tensorrt_llm
# group: llm
# config: config.py
# depends: [tensorrt, pytorch, transformers, cuda-python, nvidia_modelopt, flashinfer, nccl]
# test: [test.py, test.sh]
# requires: '>=35'
# notes: The `tensorrt-llm:builder` container includes the C++ binaries under `/opt`
#---

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TRT_LLM_VERSION \
    TRT_LLM_BRANCH \
    TRT_LLM_SOURCE \
    TRT_LLM_PATCH \
    CUDA_ARCHS \
    FORCE_BUILD="on" \
    BUILD_DIR="/opt/TensorRT-LLM/cpp/build" \
    SOURCE_DIR="/opt/TensorRT-LLM" \
    SOURCE_TAR="/tmp/TensorRT-LLM/source.tar.gz" \
    GIT_PATCHES="/tmp/TensorRT-LLM/patch.diff" \
    TMP_DIR="/tmp/TensorRT-LLM/"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openmpi-bin \
        libopenmpi-dev \
        git-lfs \
        ccache && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

RUN pip3 install polygraphy mpi4py uvicorn fastapi && \
    pip3 install --upgrade six

COPY sources ${TMP_DIR}
RUN bash ${TMP_DIR}/install_cusparselt.sh

COPY ${TRT_LLM_PATCH} ${GIT_PATCHES}
COPY build.sh install.sh /tmp/setup/

RUN /tmp/setup/install.sh 

RUN /tmp/setup/build.sh

COPY llama.sh ${SOURCE_DIR}/

# check python version and save variable PYTHON_VERSION
RUN PYTHON_TAG=$(python3 -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))") \
    && echo "PYTHON_TAG=${PYTHON_TAG}"
ARG PYTHON_VERSION=${PYTHON_TAG}

# name '_device_get_memory_info_fn' is not defined
COPY patches/profiler.py /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorrt_llm/profiler.py
COPY patches/convert_utils.py /usr/local/lib/python${PYTHON_VERSION}/dist-packages/tensorrt_llm/models/convert_utils.py
