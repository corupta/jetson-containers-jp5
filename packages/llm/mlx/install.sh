#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of MLX ${MLX_VERSION} (commit=${MLX_COMMIT})"
	exit 1
fi