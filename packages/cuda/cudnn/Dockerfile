#---
# name: cudnn
# group: cuda
# depends: cuda
# config: config.py
# test: test.sh
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG CUDNN_URL
ARG CUDNN_DEB
ARG CUDNN_PACKAGES

RUN ls /etc/apt/sources.list.d/ && \
    apt-get update && \
    apt-cache search cudnn

RUN echo "Downloading ${CUDNN_DEB}" && \
    rm -rf /tmp/cudnn && mkdir /tmp/cudnn && cd /tmp/cudnn && \
    wget ${WGET_FLAGS} ${CUDNN_URL} && \
    dpkg -i *.deb && \
    cp /var/cudnn-*-repo-*/cudnn-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && \
    apt-cache search cudnn && \
    apt list --installed | grep 'cuda\|cudnn\|cublas' && \
    apt-get install -y --no-install-recommends ${CUDNN_PACKAGES} file && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    dpkg --list | grep cudnn && \
    dpkg -P ${CUDNN_DEB} && \
    rm -rf /tmp/cudnn

# RUN cd /usr/src/cudnn_samples_v*/conv_sample/ && cat Makefile

RUN cd /usr/src/cudnn_samples_v*/conv_sample/ && cat ../samples_common.mk

RUN echo "Patching makefile support matrix list"

RUN sed -i '/^SMS_BLACKWELL\s\+?=/c\SMS_BLACKWELL ?=\nifeq ($(shell expr $(CUDA_VERSION) \\>= 13000),1)\nSMS_BLACKWELL ?= 100 101 120\nendif' /usr/src/cudnn_samples_v*/samples_common.mk

RUN cd /usr/src/cudnn_samples_v*/conv_sample/ && cat ../samples_common.mk

RUN cd /usr/src/cudnn_samples_v*/conv_sample/ && \
    make -j$(nproc)
