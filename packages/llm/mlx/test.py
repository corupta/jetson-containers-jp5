#!/usr/bin/env python3
print('testing mlx...')

from mlx.core import __version__

try:
    print('mlx version:', mlx.__version__)
except Exception as error:
    print(f"failed to print mlx version ({error})")

print('testing mlx_lm...')


import mlx_lm

try:
    print('mlx_lm version:', mlx_lm.__version__)
except Exception as error:
    print(f"failed to print mlx_lm version ({error})")
