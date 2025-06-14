#!/usr/bin/env bash
set -ex

PYTORCH_VERSION=$(python3 -c 'import torch; print(torch.__version__)')

if [ -s ${SOURCE_TAR} ]; then
	echo "extracting TensorRT-LLM sources from ${TRT_LLM_SOURCE}"
	mkdir -p ${SOURCE_DIR}
	tar -xzf ${SOURCE_TAR} -C ${SOURCE_DIR}
else
	echo "cloning TensorRT-LLM sources from git (branch=${TRT_LLM_BRANCH})"
	git clone https://github.com/NVIDIA/TensorRT-LLM.git ${SOURCE_DIR}
	cd ${SOURCE_DIR}
	git checkout ${TRT_LLM_BRANCH}
	git status
	git submodule update --init --recursive
	git lfs pull
	
	if [ -s ${GIT_PATCHES} ]; then 
		echo "applying git patches from ${TRT_LLM_PATCH}"
		git apply ${GIT_PATCHES}
	fi
	
	sed -i '/^--extra-index-url/d' requirements.txt
	sed -i 's|^tensorrt\W.*|tensorrt|' requirements.txt
	sed -i 's|^torch\W.*|torch|' requirements.txt
	#sed -i 's|nvidia-cudnn.*||' requirements.txt
	
	git status
	git diff --submodule=diff
fi	

# if [ "$FORCE_BUILD" == "on" ]; then
# 	echo "Forcing build of TensorRT-LLM ${TRT_LLM_VERSION}"
# 	exit 1
# fi

# apt update && apt install python3.12-venv
# pip3 install -r ${SOURCE_DIR}/requirements.txt
# pip3 install tensorrt_llm==${TRT_LLM_VERSION}

#pip3 uninstall -y torch && pip3 install torch==${PYTORCH_VERSION}

#pip3 show tensorrt_llm
#python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
