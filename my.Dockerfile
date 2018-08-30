FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python-pip
RUN pip install pipenv

RUN apt-get install -y \
    libglu1-mesa

RUN apt-get update && \
    apt-get install -y \
    libvtk5-dev

RUN apt-get update && \
    apt-get install -y \
    mayavi2

WORKDIR /build
COPY Pipfile* /build/
RUN pipenv install --system --deploy

ENTRYPOINT /project/sentence_classify.py
