FROM osrf/ros:humble-desktop-full

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=1000
ARG DIALOUT_GID=20
ARG VIDEO_GID=44
ARG RENDER_GID=0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    bash-completion \
    python3-argcomplete \
    build-essential \
    git \
    nano \
    wget \
    curl \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    qtwayland5 \
    libqt5waylandclient5 \
    libqt5waylandcompositor5 \
    mesa-utils \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-gazebo-ros2-control \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-joint-state-broadcaster \
    ros-humble-ros2-controllers \
    ros-humble-mecanum-drive-controller \
    ros-humble-slam-toolbox \
    ros-humble-robot-localization \
    ros-humble-diagnostic-updater \
    ros-humble-rplidar-ros \
    && rm -rf /var/lib/apt/lists/*

# PPO training uses a configurable PyTorch wheel. The default remains the
# CUDA-enabled wheel for the existing NVIDIA setup; the CPU compose variant
# overrides this build argument. The NVIDIA driver itself is supplied at
# runtime by NVIDIA Container Toolkit; it is not installed in the image.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
RUN python3 -m pip install --no-cache-dir \
    torch==2.2.2 \
    --index-url ${TORCH_INDEX_URL}

RUN python3 -m pip install --no-cache-dir \
    gymnasium==0.29.1 \
    pyserial==3.5

# Stable-Baselines3 and the rest of martha_nav's Python pins, from its requirements.txt
# (torch is already installed above, so pip keeps that wheel).
COPY requirements.txt /tmp/martha_nav_requirements.txt
RUN python3 -m pip install --no-cache-dir -r /tmp/martha_nav_requirements.txt

RUN if ! id -u ${USER_UID} >/dev/null 2>&1; then \
      groupadd --gid ${USER_GID} ${USERNAME} && \
      useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME}; \
    fi

# RUN usermod -aG sudo,dialout,video,render ${USERNAME} || true

RUN getent group dialout || groupadd -g ${DIALOUT_GID} dialout
RUN getent group video || groupadd -g ${VIDEO_GID} video
RUN if [ -n "${RENDER_GID}" ] && ! getent group render >/dev/null && ! getent group "${RENDER_GID}" >/dev/null; then \
      groupadd -g ${RENDER_GID} render; \
    fi

RUN usermod -aG sudo,dialout,video ${USERNAME}
RUN if getent group render >/dev/null; then usermod -aG render ${USERNAME}; fi

RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}

USER ${USERNAME}

WORKDIR /home/${USERNAME}/ros2_ws

RUN echo "source /opt/ros/humble/setup.bash" >> /home/ros/.bashrc
RUN echo "source /home/ros/ros2_ws/install/setup.bash" >> /home/ros/.bashrc
RUN echo "source /usr/share/bash-completion/bash_completion" >> /home/ros/.bashrc
RUN echo "eval \"\$(register-python-argcomplete3 colcon)\"" >> /home/ros/.bashrc

CMD ["bash"]
