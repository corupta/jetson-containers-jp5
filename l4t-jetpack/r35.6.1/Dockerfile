FROM nvcr.io/nvidia/l4t-jetpack:r35.4.1

# The idea is that we can follow 
# https://docs.nvidia.com/jetson/archives/jetpack-archived/jetpack-515/install-setup/index.html
# to update jetpack components.

ARG DEBIAN_FRONTEND=noninteractive

# apt install -y nvidia-jetpack won't succeed in docker, but upgrade works ok.
RUN sed -i '/^deb https:\/\/repo\.download\.nvidia\.com\/jetson\/common/d' /etc/apt/sources.list && \
		echo 'deb https://repo.download.nvidia.com/jetson/common r35.6 main' >> /etc/apt/sources.list && \
		apt update && \
		apt dist-upgrade -y && \
		apt install -y --fix-broken -o Dpkg::Options::="--force-overwrite" && \
		apt upgrade -y && \
		rm -rf /var/lib/apt/lists/* && apt-get clean