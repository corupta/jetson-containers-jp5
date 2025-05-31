#!/usr/bin/env python3
print('testing lmdeploy...')

import lmdeploy

try:
    print('lmdeploy version:', lmdeploy.__version__)
except Exception as error:
    print(f"failed to print lmdeploy version ({error})")