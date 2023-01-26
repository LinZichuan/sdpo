# We need the CUDA base dockerfile to enable GPU rendering
# on hosts with GPUs.
# The image below is a pinned version of nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04 (from Jan 2018)
# If updating the base image, be sure to test on GPU since it has broken in the past.
#FROM nvidia/cuda@sha256:4df157f2afde1cb6077a191104ab134ed4b2fd62927f27b69d788e8e79a45fa1
#FROM nvidia/cuda@sha256:8480ffb4a547ba36cb9b9553eac5cdbb3fd33c346351c41a947279838817c7d8
#FROM nvidia/cuda:10.0-base
FROM nvidia/cuda:10.0-cudnn7-devel
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y --allow-unauthenticated wget git vim apt-utils build-essential
RUN apt install -y openmpi-bin openmpi-common openssh-client zlib1g-dev unzip patchelf \
    build-essential apt-utils curl ca-certificates \
    libjpeg-dev libpng-dev \
    libgtk-3-0 libsm6 cmake ffmpeg pkg-config \
    qtbase5-dev libqt5opengl5-dev libassimp-dev \
    libboost-python-dev libtinyxml-dev \
    libosmesa6-dev software-properties-common \
    libopenmpi-dev libglew-dev openssh-server \
    libosmesa6-dev libgl1-mesa-glx libgl1-mesa-dev libglfw3 libglfw3-dev lrzsz zip unzip p7zip-full unrar

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.7-dev python3.7 python3-pip
RUN virtualenv --python=python3.7 env
#RUN rm /usr/bin/python
RUN ln -sf /env/bin/python3.7 /usr/bin/python
#RUN ln -s /env/bin/pip3.6 /usr/bin/pip
RUN ln -sf /env/bin/pytest /usr/bin/pytest

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8

RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz

ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin/libmujoco210.so:${LD_LIBRARY_PATH}

COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /mujoco_py
# Copy over just requirements.txt at first. That way, the Docker cache doesn't
# expire until we actually change the requirements.
COPY ./requirements.txt /mujoco_py/
COPY ./requirements.dev.txt /mujoco_py/
#RUN set -xe \
 #   && apt-get update \
  #  && apt-get install python3-pip
RUN python -m pip install --upgrade pip 
RUN python -m pip install --upgrade pillow
#RUN python3 -m pip install pillow
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.dev.txt

# Delay moving in the entire code until the very end.
ENTRYPOINT ["/mujoco_py/vendor/Xdummy-entrypoint"]
CMD ["pytest"]
COPY . /mujoco_py
RUN python setup.py install
RUN python -m pip install tensorflow-gpu==1.14
RUN python -m pip install dm_control
#RUN cd dmc2gym-master && python setup.py install
#RUN python3 -m pip install dm_control
#RUN pip3 install git+git://github.com/deepmind/dm_control.git
#RUN cd dm_control && python3 -m pip install .
#RUN cd dm_control2gym && python -m pip install .
RUN python -m pip install glfw Cython imageio lockfile
RUN cd baselines && python -m pip install -e . 
RUN python -m pip install atari-py
#RUN python -m pip install gym[atari]
RUN unrar x Roms.rar
RUN python -m atari_py.import_roms .  
#RUN python -m atari_py.import_roms ./ROMS
RUN python -m pip install git+https://github.com/mingfeisun/dm_control2gym.git
#RUN cd dm2gym && python -m pip install -e .
#RUN python -m pip install gym-dmc
#RUN apt install lrzsz
#RUN apt-get update
#RUN apt-get install zip unzip
