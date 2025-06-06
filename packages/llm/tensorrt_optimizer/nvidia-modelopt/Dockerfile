#---
# name: nvidia_modelopt
# group: llm
# config: config.py
# depends: [tensorrt, pytorch, torchvision, torchaudio, transformers, triton, cuda-python]
# requires: '>=34.1.0'
# test: test.py
# notes: https://github.com/NVIDIA/TensorRT-Model-Optimizer
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG NVIDIA_MODELOPT_VERSION \
    FORCE_BUILD=on \
    CUDAARCHS

COPY build.sh install.sh /tmp/NVIDIA_MODELOPT/

RUN /tmp/NVIDIA_MODELOPT/install.sh || /tmp/NVIDIA_MODELOPT/build.sh
