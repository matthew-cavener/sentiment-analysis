FROM tensorflow/tensorflow:1.13.1-py3
RUN apt-get update -q -y && apt-get install -q -y python3-dev

WORKDIR /usr/src/app/sentiment
COPY requirements.txt .
COPY setup.py .
RUN pip3 install -e .

COPY . ./

ENV TFHUB_CACHE_DIR=/usr/src/app/sentiment/models/hub_modules
ENV PYTHONUNBUFFERED=0
ENV TF_CPP_MIN_LOG_LEVEL=3

RUN useradd -ms /bin/bash sentiment
USER sentiment
EXPOSE 8081
