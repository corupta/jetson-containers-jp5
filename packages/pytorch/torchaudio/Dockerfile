#---
# name: torchaudio
# group: pytorch
# config: config.py
# depends: [pytorch, cmake]
# test: test.py
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG TORCHAUDIO_VERSION \
    FORCE_BUILD=on

COPY install.sh build.sh /tmp/torchaudio/

RUN /tmp/torchaudio/install.sh || /tmp/torchaudio/build.sh
