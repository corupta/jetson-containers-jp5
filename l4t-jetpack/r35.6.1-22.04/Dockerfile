FROM corupta/l4t-jetpack:r35.6.1

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt full-upgrade -y && \
    apt install -y update-manager-core && \
    do-release-upgrade -f DistUpgradeViewNonInteractive && \
    sed -i '/^# deb https:\/\/repo\.download\.nvidia\.com\/jetson\/common/d' /etc/apt/sources.list && \
    echo 'deb https://repo.download.nvidia.com/jetson/common r35.6 main' >> /etc/apt/sources.list && \
		rm -rf /var/lib/apt/lists/* && apt-get clean