# syntax=docker/dockerfile:1

FROM python:3.12.3-bookworm

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install pywaggle[all]
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install h5py
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# COPY . .

ENTRYPOINT ["/bin/bash", "-c"]
