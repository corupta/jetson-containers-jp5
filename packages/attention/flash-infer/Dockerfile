#---
# name: flashinfer
# group: attention
# config: config.py
# depends: [pytorch, triton]
# requires: '>=36'
# test: [test1.py, test2a.py, test2b.py, test3.py]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG FLASHINFER_VERSION \
    FLASHINFER_VERSION_SPEC \
    FLASHINFER_ENABLE_AOT=1 \
    MAX_JOBS="$(nproc)" \
    FORCE_BUILD=on \
    TMP=/tmp/flashinfer

COPY build.sh install.sh $TMP/
COPY patches/${FLASHINFER_VERSION_SPEC}.diff $TMP/patch.diff

RUN $TMP/install.sh || $TMP/build.sh || touch $TMP/.build.failed

# this retains the stage above for debugging on build failure
RUN if [ -f $TMP/.build.failed ]; then \
      echo "FlashInfer ${FLASHINFER_VERSION} build failed!"; \
      exit 1; \
    fi
    
