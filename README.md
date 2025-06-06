Hi y'all, 
So, I've got an AGX Xavier 32GB and an AGX Orin 64GB, and can't afford much.
I wanted to make the most of my Xavier, so in this repo, I'll try to build stuff that can run in it with the latest versions.
* Note that it has sm72, it supports tensor matrix operations with size same as sm70 (m8n8k4) but additionally supports int operation. (sm75 supports higher length matrix op m16n8k8, sm80+ even higher m16n8k16)
* Most open-source software forget about sm72, but if one supports sm70 or sm75 we probably can run it here as well :) Otherwise need to write custom kernels as polyfills.

* Base: Jetpack5 Cuda 12.2 CuDNN 9.10 Python 3.12 Ubuntu 20.04 (as per the latest supported for Xavier)
* Successfully built triton 3.2, pytorch 2.7, so far.
* Flashattention/flashinfer raised some issue: sm80 or higher required. Will investigate further..

### Jetpack 5.1.5 (r35.6.1)
Latest jetpack docker image from nvidia is 35.4.1, so we pull it update apt repo and upgrade to 35.6.1. You can either build it via `cd l4t-jetpack/r35.6.1 && ./build.sh` or use what I've uploaded to dockerhub, aka: `corupta/l4t-jetpack:r35.6.1`

### LMDeploy 0.8.0
* Built it for Jetpack6 and tried it in Orin as well. Turbomind seemed to ran somewhat faster than pytorch backend. Turbomind seemed to ran much slower than MLC.
* `CUDA_VERSION=12.2 PYTHON_VERSION=3.12 PYTORCH_VERSION=2.7 NUMPY_VERSION=1 CUDNN_VERSION=9.10 jetson-containers build lmdeploy:0.8.1`
* Successfully built lmdeploy version 0.8.0 (dubbed 0.8.1 due to my patches) without flash attention package, both its pytorch engine and turbomind engine works but will fail in models requiring operations/dtypes of sm80+.
* Below is a sample docker compose, that worked both much better and faster than my expectation in Jetson AGX Xavier 32GB. It might the best model to run as of today. Not all models work due to dtype, quantization, etc.
* When experimenting if you get `[TM][ERROR] CUDA runtime error: out of memory /opt/lmdeploy/src/turbomind/core/allocator.cc:49` error, if the error includes `weights` try a smaller model, otherwise lower `cache-max-entry-count` or `quant-policy`.
```
services:
  llm-server:
    stdin_open: true
    tty: true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities:
                - gpu
    ports:
      - 9000:9000
    environment:
      - DOCKER_PULL=always
      - HUGGING_FACE_HUB_TOKEN=${HUGGINGFACE_TOKEN}
      - HF_HUB_CACHE=/root/.cache/huggingface
    pull_policy: always
    volumes:
      - /mnt/nvme/cache:/root/.cache
    image: corupta/lmdeploy:0.8.1-r35.6.1-cp312-cu122-20.04
    command: lmdeploy serve api_server Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound-awq-inc
      --reasoning-parser deepseek-r1
      --backend turbomind
      --cache-max-entry-count 0.5
      --quant-policy 8
      --model-format awq
      --server-port 9000
```
https://github.com/user-attachments/assets/e2515eda-365a-4d66-a54f-bc0b5aee1a64

* Surprisingly, I got the best result for the following query in that same model, Intel AutoRound is much better than plain DWQ/AWQ or any MLC/MLX quantization in its response.

https://github.com/user-attachments/assets/e262474d-6da5-4b24-994a-16607f21ea34

### MLX 0.26.0
* It seems they are porting MLX to use CUDA unified memory. I see that the repo is under active development, I wanted to head up and start building it for jetson.
* Ok, it won't work, yet, will continue once most development is done for CUDA, tracked in [[WIP] CUDA backend](https://github.com/ml-explore/mlx/pull/1983#issuecomment-2942722126)

### TensorRT LLM 0.21
* Now, I'm trying this one :)
* Ok, horrendous experience. Get this, both NCCL and TensorRT 10 heavily depends on libc2.35. Requiring Ubuntu 22.04.
* NCCL might be built as it looks like open source but tensorrt depends on compiled `libnvinfer.so` which also use libc2.35. 
* We can either build NCCL and try to make TensortRT LLM work with tensorrt 8.5 or take a drastic approach.
* What if we upgrade ubuntu to 22 in jetpack 5 and install all the CUDA etc. packages :) Both tensorrt and nccl might work painlessly easy then.

## Jetpack 5 Ubuntu 22.04 Roadmap
- [x] Build corupta/l4t-jetpack:r35.6.1-22.04
- [x] ~~Build Cuda 12.9 just to try,~~ if it won't work build Cuda 12.2 package (Prolly, latest supported by the driver) (To my great surprise, it built but gave segfault in Cuda sample)
- [x] Build Cuda samples with maching version to see if it works. (12.2 works)
  - [x] Test Cuda 12.2 => OK
  > [x] Test Cuda 12.9 => Bad
    ```
      cuda:12.9-samples-r35.6.1-cp312-cu129-22.04-cuda_12.9-samples \
      /bin/bash -c '/bin/bash test-samples.sh                                                        
      + : /opt/cuda-samples
      ++ uname -m
      + cd /opt/cuda-samples/bin/aarch64/linux/release
      + ./deviceQuery
      ./deviceQuery Starting...
      CUDA Device Query (Runtime API) version (CUDART static linking)
      test-samples.sh: line 8:    21 Segmentation fault      (core dumped) ./deviceQuery
      [14:46:14] Failed building:  cuda:12.9-samples
    ```
  > [ ] Test Cuda 12.8 => Pending
    ```
      cuda:12.8-samples-r35.6.1-cp312-cu128-22.04-cuda_12.8-samples \
      /bin/bash -c '/bin/bash test-samples.sh

      + : /opt/cuda-samples
      ++ uname -m
      + cd /opt/cuda-samples/bin/aarch64/linux/release
      + ./deviceQuery
      ./deviceQuery Starting...

      CUDA Device Query (Runtime API) version (CUDART static linking)

      cudaGetDeviceCount returned 803
      -> system has unsupported display driver / cuda driver combination
    ```
  > [x] Test Cuda 12.4 => OK
    ```
      cuda:12.4-samples-r35.6.1-cp312-cu124-22.04 \
      /bin/bash -c '/bin/bash test-samples.sh


      + : /opt/cuda-samples
      ++ uname -m
      + cd /opt/cuda-samples/bin/aarch64/linux/release
      + ./deviceQuery
      ./deviceQuery Starting...

      CUDA Device Query (Runtime API) version (CUDART static linking)

      Detected 1 CUDA Capable device(s)

      Device 0: "Xavier"
        CUDA Driver Version / Runtime Version          12.4 / 12.4
        CUDA Capability Major/Minor version number:    7.2
        Total amount of global memory:                 30991 MBytes (32496205824 bytes)
        (008) Multiprocessors, (064) CUDA Cores/MP:    512 CUDA Cores
        GPU Max Clock rate:                            1377 MHz (1.38 GHz)
        Memory Clock rate:                             1377 Mhz
        Memory Bus Width:                              256-bit
        L2 Cache Size:                                 524288 bytes
        Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
        Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
        Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
        Total amount of constant memory:               65536 bytes
        Total amount of shared memory per block:       49152 bytes
        Total shared memory per multiprocessor:        98304 bytes
        Total number of registers available per block: 65536
        Warp size:                                     32
        Maximum number of threads per multiprocessor:  2048
        Maximum number of threads per block:           1024
        Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
        Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
        Maximum memory pitch:                          2147483647 bytes
        Texture alignment:                             512 bytes
        Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
        Run time limit on kernels:                     No
        Integrated GPU sharing Host Memory:            Yes
        Support host page-locked memory mapping:       Yes
        Alignment requirement for Surfaces:            Yes
        Device has ECC support:                        Disabled
        Device supports Unified Addressing (UVA):      Yes
        Device supports Managed Memory:                Yes
        Device supports Compute Preemption:            Yes
        Supports Cooperative Kernel Launch:            Yes
        Supports MultiDevice Co-op Kernel Launch:      Yes
        Device PCI Domain ID / Bus ID / location ID:   0 / 0 / 0
        Compute Mode:
          < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

      deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.4, CUDA Runtime Version = 12.4, NumDevs = 1
      Result = PASS
      + ./bandwidthTest
      [CUDA Bandwidth Test] - Starting...
      Running on...

      Device 0: Xavier
      Quick Mode

      Host to Device Bandwidth, 1 Device(s)
      PINNED Memory Transfers
        Transfer Size (Bytes)        Bandwidth(GB/s)
        32000000                     15.6

      Device to Host Bandwidth, 1 Device(s)
      PINNED Memory Transfers
        Transfer Size (Bytes)        Bandwidth(GB/s)
        32000000                     28.9

      Device to Device Bandwidth, 1 Device(s)
      PINNED Memory Transfers
        Transfer Size (Bytes)        Bandwidth(GB/s)
        32000000                     110.6

      Result = PASS

      NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
      + ./vectorAdd
      [Vector addition of 50000 elements]
      Copy input data from the host memory to the CUDA device
      CUDA kernel launch with 196 blocks of 256 threads
      Copy output data from the CUDA device to the host memory
      Test PASSED
      Done
      + ./matrixMul
      [Matrix Multiply Using CUDA] - Starting...
      GPU Device 0: "Xavier" with compute capability 7.2

      MatrixA(320,320), MatrixB(640,320)
      Computing result using CUDA Kernel...
      done
      Performance= 358.26 GFlop/s, Time= 0.366 msec, Size= 131072000 Ops, WorkgroupSize= 1024 threads/block
      Checking computed result for correctness: Result = PASS

      NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.

      [18:57:28] ✅ Built cuda:12.4-samples (cuda:12.4-samples-r35.6.1-cp312-cu124-22.04) 
    ```
  - [ ] Test Cuda 12.6 =>
- [ ] Build TensorRT 10.10
- [ ] Build triton
- [ ] Build pytorch
- [ ] Build torchvision
- [ ] Build torchaudio
- [ ] Build transformers
- [ ] Build other stuff.


# Below is the original readme from [dusty-nv/jetson-containers](https://github.com/dusty-nv/jetson-containers)
Special thanks to original contributers, I really loved the project structure.

[![a header for a software project about building containers for AI and machine learning](https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/header_blueprint_rainbow.jpg)](https://www.jetson-ai-lab.com)

# CUDA Containers for Edge AI & Robotics

Modular container build system that provides the latest [**AI/ML packages**](https://pypi.jetson-ai-lab.dev/) for [NVIDIA Jetson](https://jetson-ai-lab.com) :rocket::robot:

> [!NOTE]
> Ubuntu 24.04 containers for JetPack 6 are now available (with CUDA support)
>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`LSB_RELEASE=24.04 jetson-containers build pytorch:2.7`  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`jetson-containers run dustynv/pytorch:2.7-r36.4-cu128-24.04`
>
> ARM SBSA (Server Base System Architecture) is supported for GH200 / GB200.  
> To install CUDA 12.8 SBSA wheels for Python 3.10 / 22.04 or Python 3.12 / 24.04:  
>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`pip3 install torch torchvision torchaudio \`  
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`--index-url https://pypi.jetson-ai-lab.dev/sbsa/cu128`
>
> Thanks to all our contributors from **[`Discord`](https://discord.gg/BmqNSK4886)** and AI community for their support 🤗

![Jetson PyPI Health](https://img.shields.io/endpoint?url=https://tokk-nv.github.io/jetson-containers/health.json)

| |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
|---|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **ML** | [`pytorch`](packages/pytorch) [`tensorflow`](packages/ml/tensorflow) [`jax`](packages/ml/jax) [`onnxruntime`](packages/ml/onnxruntime) [`deepstream`](packages/cv/deepstream) [`holoscan`](packages/cv/holoscan) [`CTranslate2`](packages/ml/ctranslate2) [`JupyterLab`](packages/ml/jupyterlab)                                                                                                                                                                                                                                                                               |
| **LLM** | [`SGLang`](packages/llm/sglang) [`vLLM`](packages/llm/vllm) [`MLC`](packages/llm/mlc) [`AWQ`](packages/llm/awq) [`transformers`](packages/llm/transformers) [`text-generation-webui`](packages/llm/text-generation-webui) [`ollama`](packages/llm/ollama) [`llama.cpp`](packages/llm/llama_cpp) [`llama-factory`](packages/llm/llama-factory) [`exllama`](packages/llm/exllama) [`AutoGPTQ`](packages/llm/auto_gptq) [`FlashAttention`](packages/attention/flash-attention) [`DeepSpeed`](packages/llm/deepspeed) [`bitsandbytes`](packages/llm/bitsandbytes) [`xformers`](packages/llm/xformers) |
| **VLM** | [`llava`](packages/vlm/llava) [`llama-vision`](packages/vlm/llama-vision) [`VILA`](packages/vlm/vila) [`LITA`](packages/vlm/lita) [`NanoLLM`](packages/llm/nano_llm) [`ShapeLLM`](packages/vlm/shape-llm) [`Prismatic`](packages/vlm/prismatic) [`xtuner`](packages/vlm/xtuner)                                                                                                                                                                                                                                                                                                                |
| **VIT** | [`NanoOWL`](packages/vit/nanoowl) [`NanoSAM`](packages/vit/nanosam) [`Segment Anything (SAM)`](packages/vit/sam) [`Track Anything (TAM)`](packages/vit/tam) [`clip_trt`](packages/vit/clip_trt)                                                                                                                                                                                                                                                                                                                                                                                                |
| **RAG** | [`llama-index`](packages/rag/llama-index) [`langchain`](packages/rag/langchain) [`jetson-copilot`](packages/rag/jetson-copilot) [`NanoDB`](packages/vectordb/nanodb) [`FAISS`](packages/vectordb/faiss) [`RAFT`](packages/ml/rapids/raft)                                                                                                                                                                                                                                                                                                                                                      |
| **L4T** | [`l4t-pytorch`](packages/ml/l4t/l4t-pytorch) [`l4t-tensorflow`](packages/ml/l4t/l4t-tensorflow) [`l4t-ml`](packages/ml/l4t/l4t-ml) [`l4t-diffusion`](packages/ml/l4t/l4t-diffusion) [`l4t-text-generation`](packages/ml/l4t/l4t-text-generation)                                                                                                                                                                                                                                                                                                                                                              |
| **CUDA** | [`cupy`](packages/numeric/cupy) [`cuda-python`](packages/cuda/cuda-python) [`pycuda`](packages/cuda/pycuda) [`cv-cuda`](packages/cv/cv-cuda) [`opencv:cuda`](packages/cv/opencv) [`numba`](packages/numeric/numba)                                                                                                                                                                                                                                                                                                                                          |
| **Robotics** | [`Cosmos`](packages/diffusion/cosmos) [`Genesis`](packages/sim/genesis) [`ROS`](packages/robots/ros) [`LeRobot`](packages/robots/lerobot) [`OpenVLA`](packages/vla/openvla) [`3D Diffusion Policy`](packages/diffusion/3d_diffusion_policy) [`Crossformer`](packages/diffusion/crossformer) [`MimicGen`](packages/sim/mimicgen) [`OpenDroneMap`](packages/robots/opendronemap) [`ZED`](packages/hardware/zed)                                                                                                                                                                                         |
| **Graphics** | [`stable-diffusion-webui`](packages/diffusion/stable-diffusion-webui) [`comfyui`](packages/diffusion/comfyui) [`nerfstudio`](packages/nerf/nerfstudio) [`meshlab`](packages/nerf/meshlab) [`pixsfm`](packages/nerf/pixsfm) [`gsplat`](packages/nerf/gsplat)                                                                                                                                                                                                                                                                                                                                    |
| **Mamba** | [`mamba`](packages/mamba/mamba) [`mambavision`](packages/mamba/mambavision) [`cobra`](packages/mamba/cobra) [`dimba`](packages/mamba/dimba) [`videomambasuite`](packages/mamba/videomambasuite)                                                                                                                                                                                                                                                                                                                                                                                                |
| **Speech** | [`whisper`](packages/speech/whisper) [`whisper_trt`](packages/speech/whisper_trt) [`piper`](packages/speech/piper-tts) [`riva`](packages/speech/riva-client) [`audiocraft`](packages/speech/audiocraft) [`voicecraft`](packages/speech/voicecraft) [`xtts`](packages/speech/xtts)                                                                                                                                                                                                                                                                                                              |
| **Home/IoT** | [`homeassistant-core`](packages/smart-home/homeassistant-core) [`wyoming-whisper`](packages/smart-home/wyoming/wyoming-whisper) [`wyoming-openwakeword`](packages/smart-home/wyoming/openwakeword) [`wyoming-piper`](packages/smart-home/wyoming/piper)                                                                                                                                                                                                                                                                                                                                        |

See the [**`packages`**](packages) directory for the full list, including pre-built container images for JetPack/L4T.

Using the included tools, you can easily combine packages together for building your own containers.  Want to run ROS2 with PyTorch and Transformers?  No problem - just do the [system setup](/docs/setup.md), and build it on your Jetson:

```bash
$ jetson-containers build --name=my_container pytorch transformers ros:humble-desktop
```

There are shortcuts for running containers too - this will pull or build a [`l4t-pytorch`](packages/l4t/l4t-pytorch) image that's compatible:

```bash
$ jetson-containers run $(autotag l4t-pytorch)
```
> <sup>[`jetson-containers run`](/docs/run.md) launches [`docker run`](https://docs.docker.com/engine/reference/commandline/run/) with some added defaults (like `--runtime nvidia`, mounted `/data` cache and devices)</sup><br>
> <sup>[`autotag`](/docs/run.md#autotag) finds a container image that's compatible with your version of JetPack/L4T - either locally, pulled from a registry, or by building it.</sup>

If you look at any package's readme (like [`l4t-pytorch`](packages/l4t/l4t-pytorch)), it will have detailed instructions for running it.

#### Changing CUDA Versions

You can rebuild the container stack for different versions of CUDA by setting the `CUDA_VERSION` variable:

```bash
CUDA_VERSION=12.4 jetson-containers build transformers
```

It will then go off and either pull or build all the dependencies needed, including PyTorch and other packages that would be time-consuming to compile.  There is a [Pip server](/docs/build.md#pip-server) that caches the wheels to accelerate builds.  You can also request specific versions of cuDNN, TensorRT, Python, and PyTorch with similar environment variables like [here](/docs/build.md#changing-versions).

## Documentation

<a href="https://www.jetson-ai-lab.com"><img align="right" width="200" height="200" src="https://nvidia-ai-iot.github.io/jetson-generative-ai-playground/images/JON_Gen-AI-panels.png"></a>

* [Package List](/packages)
* [Package Definitions](/docs/packages.md)
* [System Setup](/docs/setup.md)
* [Building Containers](/docs/build.md)
* [Running Containers](/docs/run.md)

Check out the tutorials at the [**Jetson Generative AI Lab**](https://www.jetson-ai-lab.com)!

## Getting Started

Refer to the [System Setup](/docs/setup.md) page for tips about setting up your Docker daemon and memory/storage tuning.

```bash
# install the container tools
git clone https://github.com/dusty-nv/jetson-containers
bash jetson-containers/install.sh

# automatically pull & run any container
jetson-containers run $(autotag l4t-pytorch)
```

Or you can manually run a [container image](https://hub.docker.com/r/dustynv) of your choice without using the helper scripts above:

```bash
sudo docker run --runtime nvidia -it --rm --network=host dustynv/l4t-pytorch:r36.2.0
```

Looking for the old jetson-containers?   See the [`legacy`](https://github.com/dusty-nv/jetson-containers/tree/legacy) branch.

## Gallery

<a href="https://www.youtube.com/watch?v=UOjqF3YCGkY"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_llava_clip.gif"></a>
> [Multimodal Voice Chat with LLaVA-1.5 13B on NVIDIA Jetson AGX Orin](https://www.youtube.com/watch?v=9ObzbbBTbcc) (container: [`NanoLLM`](https://dusty-nv.github.io/NanoLLM/))

<br/>

<a href="https://www.youtube.com/watch?v=hswNSZTvEFE"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/llamaspeak_70b_yt.jpg" width="800px"></a>
> [Interactive Voice Chat with Llama-2-70B on NVIDIA Jetson AGX Orin](https://www.youtube.com/watch?v=wzLHAgDxMjQ) (container: [`NanoLLM`](https://dusty-nv.github.io/NanoLLM/))

<br/>

<a href="https://www.youtube.com/watch?v=OJT-Ax0CkhU"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/nanodb_tennis.jpg"></a>
> [Realtime Multimodal VectorDB on NVIDIA Jetson](https://www.youtube.com/watch?v=wzLHAgDxMjQ) (container: [`nanodb`](/packages/vectordb/nanodb))

<br/>

<a href="https://www.jetson-ai-lab.com/tutorial_nanoowl.html"><img src="https://github.com/NVIDIA-AI-IOT/nanoowl/raw/main/assets/jetson_person_2x.gif"></a>
> [NanoOWL - Open Vocabulary Object Detection ViT](https://www.jetson-ai-lab.com/tutorial_nanoowl.html) (container: [`nanoowl`](/packages/vit/nanoowl))

<a href="https://www.youtube.com/watch?v=w48i8FmVvLA"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/live_llava.gif"></a>
> [Live Llava on Jetson AGX Orin](https://youtu.be/X-OXxPiUTuU) (container: [`NanoLLM`](https://dusty-nv.github.io/NanoLLM/))

<a href="https://www.youtube.com/watch?v=wZq7ynbgRoE"><img width="640px" src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/live_llava_bear.jpg"></a>
> [Live Llava 2.0 - VILA + Multimodal NanoDB on Jetson Orin](https://youtu.be/X-OXxPiUTuU) (container: [`NanoLLM`](https://dusty-nv.github.io/NanoLLM/))

<a href="https://www.jetson-ai-lab.com/tutorial_slm.html"><img src="https://www.jetson-ai-lab.com/images/slm_console.gif"></a>
> [Small Language Models (SLM) on Jetson Orin Nano](https://www.jetson-ai-lab.com/tutorial_slm.html) (container: [`NanoLLM`](https://dusty-nv.github.io/NanoLLM/))

<a href="https://www.jetson-ai-lab.com/tutorial_nano-vlm.html#video-sequences"><img src="https://raw.githubusercontent.com/dusty-nv/jetson-containers/docs/docs/images/video_vila_wildfire.gif"></a>
> [Realtime Video Vision/Language Model with VILA1.5-3b](https://www.jetson-ai-lab.com/tutorial_nano-vlm.html#video-sequences) (container: [`NanoLLM`](https://dusty-nv.github.io/NanoLLM/))

