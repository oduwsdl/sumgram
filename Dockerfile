FROM python:3.7-stretch

LABEL maintainer="Alexander Nwala <anwala@cs.odu.edu>"

WORKDIR /home/sumgram

RUN pip install --upgrade pip && pip install sumgram
ENTRYPOINT ["sumgram"]