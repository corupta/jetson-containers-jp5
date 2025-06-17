#!/usr/bin/env bash
set -ex

ncclras --version # tests wouldn't work idk why
# cd /opt/nccl/nccl-tests
# NCCL_DEBUG=INFO ./build/all_reduce_perf -b 8 -e 256M -f 2 -g 1