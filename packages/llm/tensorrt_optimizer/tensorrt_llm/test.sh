#!/usr/bin/env bash
set -ex

export FORCE_BUILD=on
/opt/TensorRT-LLM/llama.sh
echo "TensorRT-LLM OK"