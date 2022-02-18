FROM docker.io/python:3.9
FROM docker.io/tensorflow/tensorflow:latest-gpu

WORKDIR /gmps-project

COPY . /gmps-project

RUN pip install --upgrade pip
RUN pip install -r requirements.txt


