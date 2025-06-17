#!/usr/bin/env bash
set -ex

# Clone the repository if it doesn't exist
git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer /opt/nvidia_modelopt
cd /opt/nvidia_modelopt
git checkout ${NVIDIA_MODELOPT_VERSION}
git submodule update --init --recursive

pip3 install .[all]
