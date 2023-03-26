FROM tensorflow/tensorflow:2.9.1-gpu@sha256:a34c2420739cd5a7b5662449bc21eb32d3d1c98063726ae2bd7db819cc93d72f  
WORKDIR /app

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

#INIT
#---CHANGES BEFORE HERE WILL RERUN UPDATE---

#CV2 dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

#git
RUN apt-get -y install git
RUN pip install --upgrade pip

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

#PROJECT
COPY . /app

#UIDs
ARG UNAME=user
ARG UID=1000
ARG GID=1000

RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME

# VSCODE DEFAULT:
# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
#RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
#USER appuser

USER $UNAME
# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "main.py"]
