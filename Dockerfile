# syntax=docker/dockerfile:1

FROM python:latest

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install -U pywaggle[all]
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

ENTRYPOINT ["python", "main.py"]
