#!/usr/bin/env bash
set -ex

# cd ${SOURCE_DIR}/3rdparty/pybind11
# git checkout v2.13 # Build won't work, current 3rdparty is pointing to a weird version.

echo "Building TensorRT-LLM ${TRT_LLM_VERSION}"

cd ${SOURCE_DIR}
if [ -s ${GIT_PATCHES} ]; then 
        echo "applying git patches from ${TRT_LLM_PATCH}"
        git apply ${GIT_PATCHES}
fi

sed -i '/^--extra-index-url/d' requirements.txt
sed -i 's|^tensorrt\W.*|tensorrt|' requirements.txt
sed -i 's|^torch\W.*|torch|' requirements.txt
#sed -i 's|nvidia-cudnn.*||' requirements.txt

git status
# git diff --submodule=diff


sed -i 's|venv_python = venv_prefix / sys\.executable\.removeprefix(sys\.prefix)\[1:\]|venv_python = venv_prefix / "bin" / Path(sys.executable).name|' ${SOURCE_DIR}/scripts/build_wheel.py
sed -i 's|^flashinfer-python\W.*|flashinfer-python|' ${SOURCE_DIR}/requirements.txt
# sed -i 's/find_package(TensorRT 10 REQUIRED COMPONENTS OnnxParser)/find_package(TensorRT REQUIRED COMPONENTS OnnxParser)/' ${SOURCE_DIR}/cpp/CMakeLists.txt
# sed -i '/add_definitions("-DENABLE_BF16")/d' ${SOURCE_DIR}/cpp/CMakeLists.txt
# sed -i '/add_definitions("-DENABLE_FP8")/d' ${SOURCE_DIR}/cpp/CMakeLists.txt
# sed -i '/add_definitions("-DENABLE_FP4")/d' ${SOURCE_DIR}/cpp/CMakeLists.txt

# patch tensorrt header for fp4 (in TensorRT 10.7.0)
sed -i 's/^.*kINT4 = 9,.*/    kINT4 = 9,  kFP4=10,/' /usr/include/aarch64-linux-gnu/NvInferRuntimeBase.h


# BUILD CUBINS FOR JETSON ORIN. (SM 87)
cd ${SOURCE_DIR}/cpp/kernels/xqa
python3 gen_cubins.py
cp -r cubin/*sm_87*.cpp ${SOURCE_DIR}/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/cubin/
rm ${SOURCE_DIR}/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/cubin/*sm_86*.cpp
# TODO WE MIGHT FIX MEDUSA BUILD AS WELL (ie: python3 gen_cubins.py medusa)
#  to do so, compare mha.cu, mha_sm90.cu and mha_sm120.cu and fix
#       mha.cu(1485): error: identifier "inputSeqLen" is undefined => static_assert(inputSeqLen == 1);

# │/usr/bin/ld: ../tensorrt_llm/libtensorrt_llm.so: undefined reference to `tensorrt_llm::kernels::cubin_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_64_sm87_cu_cubin'                                                                                                                             │
# │/usr/bin/ld: ../tensorrt_llm/libtensorrt_llm.so: undefined reference to `tensorrt_llm::kernels::cubin_fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_64_sm87_cu_cubin_len'    

# fmha now uses cu instead of cubins. (but only if we delete them)
rm ${SOURCE_DIR}/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/cubin/*.cubin.cpp

python3 ${SOURCE_DIR}/scripts/build_wheel.py \
        --clean \
        --build_type Release \
        --cuda_architectures "${CUDA_ARCHS}" \
        --build_dir ${BUILD_DIR} \
        --dist_dir $PIP_WHEEL_DIR \
        --benchmarks \
        --use_ccache \
        --generate_fmha \
        --python_bindings # --extra-cmake-vars "ENABLE_MULTI_DEVICE=0"

# --extra-cmake-vars "ENABLE_MULTI_DEVICE=0" \  # requires us to use with TLLM_WORKER_USE_SINGLE_PROCESS=1 then.

# note that this approach installed nccl from wheel, we might need to build it ourselves if it's a broken one.

pip3 install $PIP_WHEEL_DIR/tensorrt_llm*.whl

#pip3 show tensorrt_llm
#python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

twine upload --verbose $PIP_WHEEL_DIR/tensorrt_llm*.whl || echo "failed to upload wheel to ${TWINE_REPOSITORY_URL}"
