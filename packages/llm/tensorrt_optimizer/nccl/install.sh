#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of NCCL ${NCCL_VERSION}"
	exit 1
fi

# How to install is not implemented :)
exit 1