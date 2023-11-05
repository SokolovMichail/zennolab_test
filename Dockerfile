FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app
ARG DEBIAN_FRONTEND=noninteractive

# pipenv needs these lines
ENV LANG = en_US.UTF-8
ENV LC_ALL = en_US.UTF-8
ENV LC_CTYPE = en_US.UTF-8

# suppress TF information and warning messages
ENV TF_CPP_MIN_LOG_LEVEL = 2

# install apt packages
RUN apt-get update -y && apt-get install -y python3-pip && \
    apt-get install ffmpeg libsm6 libxext6 -y
# install pip packages
RUN pip3 install -U pipenv==2023.5.19
COPY . .
RUN pipenv install --system

CMD python3 run.py