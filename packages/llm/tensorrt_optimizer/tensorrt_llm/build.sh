#!/usr/bin/env bash
set -ex

# cd ${SOURCE_DIR}/3rdparty/pybind11
# git checkout v2.13 # Build won't work, current 3rdparty is pointing to a weird version.

echo "Building TensorRT-LLM ${TRT_LLM_VERSION}"

cd ${SOURCE_DIR}
sed -i 's|venv_python = venv_prefix / sys\.executable\.removeprefix(sys\.prefix)\[1:\]|venv_python = venv_prefix / "bin" / Path(sys.executable).name|' ${SOURCE_DIR}/scripts/build_wheel.py
sed -i 's|^flashinfer-python\W.*|flashinfer-python|' ${SOURCE_DIR}/requirements.txt
# sed -i 's/find_package(TensorRT 10 REQUIRED COMPONENTS OnnxParser)/find_package(TensorRT REQUIRED COMPONENTS OnnxParser)/' ${SOURCE_DIR}/cpp/CMakeLists.txt
# sed -i '/add_definitions("-DENABLE_BF16")/d' ${SOURCE_DIR}/cpp/CMakeLists.txt
# sed -i '/add_definitions("-DENABLE_FP8")/d' ${SOURCE_DIR}/cpp/CMakeLists.txt
# sed -i '/add_definitions("-DENABLE_FP4")/d' ${SOURCE_DIR}/cpp/CMakeLists.txt

# patch tensorrt header for fp4 (in TensorRT 10.7.0)
sed -i 's/^.*kINT4 = 9,.*/    kINT4 = 9,  kFP4=10,/' /usr/include/aarch64-linux-gnu/NvInferRuntimeBase.h

# mv cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/cubin/fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_64_sm86.cubin.cpp \
#         cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/cubin/fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_64_sm87.cubin.cpp
# sed -i 's|_sm86_|_sm87_|g' \
#         cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/cubin/fmha_v2_flash_attention_bf16_64_32_S_q_paged_kv_64_sm87.cubin.cpp

for f in $(find . -name '*sm_86*.cpp'); do
        f2=$(echo $f | sed 's|sm_86|sm_87|g')
        echo "patch sm86->sm87 $f2"
        mv $f $f2
        sed -i 's|_sm_86_|_sm_87_|g' $f2
        sed -i 's|_sm86_|_sm87_|g' $f2
done
for f in $(find . -name '*sm86*.cpp'); do
        f2=$(echo $f | sed 's|sm86|sm87|g')
        echo "patch sm86->sm87 $f2"
        mv $f $f2
        sed -i 's|_sm_86_|_sm_87_|g' $f2
        sed -i 's|_sm86_|_sm87_|g' $f2
done

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
