#FROM tensorflow/tensorflow:2.11.1@sha256:1f5ec29ae06feea88b4b0fa3b2cfee9ee79a940fe293071307e50d3218f09915
FROM tensorflow/tensorflow:2.9.1-gpu@sha256:a34c2420739cd5a7b5662449bc21eb32d3d1c98063726ae2bd7db819cc93d72f  
WORKDIR /usr/app

#INIT
#---CHANGES BEFORE HERE WILL RERUN UPDATE---

#CV2 dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

#entrypoint dependencies
RUN apt-get install gosu -y

#UIDs

ARG UNAME=user
ARG UID=1000
ARG GID=1000

RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME

#PROJECT

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

#RUN
USER $UNAME
CMD [ "python", "main.py"]
