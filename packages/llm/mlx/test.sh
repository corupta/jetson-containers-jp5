#!/usr/bin/env bash
set -ex

cd /opt/mlx

./build/examples/cpp/tutorial

python -m unittest discover /opt/mlx/python/tests

# which mlx
# mlx --help
# TODO we can improve the test :)
