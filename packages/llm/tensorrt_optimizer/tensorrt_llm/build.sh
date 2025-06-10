#!/usr/bin/env bash
set -ex

cd ${SOURCE_DIR}/3rdparty/pybind11
git checkout v2.13 # Build won't work, current 3rdparty is pointing to a weird version.

echo "Building TensorRT-LLM ${TRT_LLM_VERSION}"

cd ${SOURCE_DIR}
sed -i 's|venv_python = venv_prefix / sys\.executable\.removeprefix(sys\.prefix)\[1:\]|venv_python = venv_prefix / "bin" / Path(sys.executable).name|' ${SOURCE_DIR}/scripts/build_wheel.py
sed -i 's|^flashinfer-python\W.*|flashinfer-python|' ${SOURCE_DIR}/requirements.txt
sed -i 's/find_package(TensorRT 10 REQUIRED COMPONENTS OnnxParser)/find_package(TensorRT REQUIRED COMPONENTS OnnxParser)/' ${SOURCE_DIR}/cpp/CMakeLists.txt

# TODO 1. Try to build tensorrt 10.10 (10.11 won't build)
        # Test if its trt exec works
        # Test without copying lib, then with copying lib.
# TODO 2. Try to disable kINT64, kBF16, kINT4 kFP4 and build
python3 ${SOURCE_DIR}/scripts/build_wheel.py \
        --clean \
        --build_type Release \
        --cuda_architectures "${CUDA_ARCHS}" \
        --build_dir ${BUILD_DIR} \
        --dist_dir $PIP_WHEEL_DIR \
        --extra-cmake-vars "ENABLE_MULTI_DEVICE=0" \
        --benchmarks \
        --use_ccache \
        --python_bindings

# note that this approach installed nccl from wheel, we might need to build it ourselves if it's a broken one.

pip3 install $PIP_WHEEL_DIR/tensorrt_llm*.whl

#pip3 show tensorrt_llm
#python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

twine upload --verbose $PIP_WHEEL_DIR/tensorrt_llm*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
