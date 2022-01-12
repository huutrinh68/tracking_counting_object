FROM nvcr.io/nvidia/tensorflow:21.11-tf1-py3
ENV NVIDIA_VISIBLE_DEVICES all

#### Setting noninteractive user install ####
ENV DEBIAN_FRONTEND noninteractive

#### Install common library ####
RUN apt-get update && \
    apt-get -y dist-upgrade && \
    apt-get install -y default-libmysqlclient-dev && \
    apt-get install -y --no-install-recommends build-essential \
    gosu \
    libpq-dev \
    libxml2-dev \
    python3-all-dev \
    python3 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

#### Setting timezone, language ####
ENV TZ=Asia/Tokyo \
    LANG=ja_JP.utf8 \
    LC_COLLATE=C

#### Install other libs ####
RUN apt-get -y update && \
    apt-get install -y libxml2-dev wget git cmake curl \
    libglib2.0-0 libsm6 libxrender1 libfontconfig1 \
    gnupg xserver-xorg x11-apps xorg-dev

RUN pip3 install -U pip
WORKDIR /app
COPY /requirements.txt /app/requirements.txt

RUN pip3 install -r /app/requirements.txt

CMD tail -f /dev/null
