FROM python:3.8-slim

RUN apt-get update
RUN apt-get install -y git-all
RUN apt-get install -y git-lfs

RUN mkdir -p /opt
WORKDIR /opt
RUN git lfs install
RUN git clone https://github.com/aksg87/adpkd-segmentation-pytorch.git
WORKDIR /opt/adpkd-segmentation-pytorch
COPY requirements.barebones_inference.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -e .

# pre-download model
RUN python adpkd_segmentation/inference/inference.py

# remove unused models in ./checkpoints
RUN rm checkpoints/*nocrop*

# load test data
#COPY inference_input inference_input
