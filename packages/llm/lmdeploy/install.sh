#!/usr/bin/env bash
set -ex

if [ "$FORCE_BUILD" == "on" ]; then
	echo "Forcing build of LMDEPLOY ${LMDEPLOY_VERSION} (commit=${LMDEPLOY_COMMIT})"
	exit 1
fi