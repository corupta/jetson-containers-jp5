#!/usr/bin/env bash

/usr/src/tensorrt/bin/trtexec --help
/usr/src/tensorrt/bin/trtexec --onnx=/usr/src/tensorrt/data/mnist/mnist.onnx

python3 -c "import tensorrt; print('TensorRT version:', tensorrt.__version__)" || echo "WARNING - failed to 'import tensorrt' under Python $(python3 --version)"