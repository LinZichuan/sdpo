FROM python:3.6

RUN apt update && apt install -y --allow-unauthenticated wget git vim apt-utils build-essential
RUN apt install -y openmpi-bin openmpi-common openssh-client zlib1g-dev unzip patchelf \
    build-essential apt-utils curl ca-certificates \
    libjpeg-dev libpng-dev \
    libgtk-3-0 libsm6 cmake ffmpeg pkg-config \
    qtbase5-dev libqt5opengl5-dev libassimp-dev \
    libboost-python-dev libtinyxml-dev \
    libosmesa6-dev software-properties-common \
    libopenmpi-dev libglew-dev openssh-server \
    libosmesa6-dev libgl1-mesa-glx libgl1-mesa-dev libglfw3 libglfw3-dev

ARG UID
RUN useradd -u $UID --create-home mingfei
USER mingfei
WORKDIR /home/mingfei

RUN mkdir -p /home/mingfei/.mujoco 

RUN wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /home/mingfei/.mujoco \
    && rm mujoco.zip
RUN wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /home/mingfei/.mujoco \
    && rm mujoco.zip

# put the liscense under the current folder
COPY mjkey.txt /home/mingfei/.mujoco/mjkey.txt

ENV LD_LIBRARY_PATH /home/mingfei/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /home/mingfei/.mujoco/mjpro200_linux/bin:${LD_LIBRARY_PATH}

# install all requirements
RUN python -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN echo 'set editing-mode vi' >> /home/mingfei/.inputrc
RUN echo 'set keymap vi' >> /home/mingfei/.inputrc

WORKDIR /home/mingfei/ppo
CMD /bin/bash