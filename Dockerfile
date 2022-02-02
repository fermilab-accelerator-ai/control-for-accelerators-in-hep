FROM python:3.8-slim-buster
FROM docker.io/tensorflow/tensorflow:latest-gpu
#FROM docker.io/nvidia/cuda:11.6.0-runtime-ubuntu20.04

WORKDIR /gmps-project

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python3 -c "import tensorflow as tf; tf.config.list_physical_devices('GPU')"
RUN python3 run_training.py