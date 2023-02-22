# This Dockerfile was heavily inspired by this one:
#   https://github.com/xychelsea/anaconda3-docker/blob/main/Dockerfile

# Use tensorflow base with gpu support + jupyter pre-installed
FROM tensorflow/tensorflow:latest-gpu-jupyter

#FROM continuumio/anaconda3
# All necessary graphic and other utilities
# Update packages
RUN apt-get update --fix-missing \
    && apt-get -y upgrade \
    && apt-get -y dist-upgrade

RUN apt-get install -y  git cmake build-essential libgl1-mesa-dev libsdl2-dev \
                        libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
                        libdirectfb-dev libst-dev mesa-utils xvfb x11vnc libgl1-mesa-glx \
                        libcublas-11-8 libcublas-dev-11-8 \
                        python3-pip wget sudo locales curl ca-certificates bzip2

# Install google chrome for development and other
COPY install-chrome.sh .
RUN bash install-chrome.sh 

RUN pip install pip setuptools psutil wheel
RUN pip install gfootball

# Install other pythondeps
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd -ms /bin/bash xglosm01
RUN echo 'xglosm01:pass' | chpasswd
RUN adduser xglosm01 sudo
RUN echo 'root:pass' | chpasswd
WORKDIR /xglosm01
USER xglosm01