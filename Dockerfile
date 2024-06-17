# syntax=docker/dockerfile:1

# python version 3.6
FROM waggle/plugin-base:1.1.1-ml-torch1.9
# CUDA supports backward compatibility, so the application version <= driver version
# 11.8 for Dell blade server, 10.2 for waggle node
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-24-03.html

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install pywaggle[all]
RUN pip install h5py
# native pyymal version is too old with the python
RUN pip install --ignore-installed PyYAML

# COPY . .

CMD ["/bin/sh", "-c", "bash"]
