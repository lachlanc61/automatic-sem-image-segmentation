#FROM tensorflow/tensorflow:2.11.1@sha256:1f5ec29ae06feea88b4b0fa3b2cfee9ee79a940fe293071307e50d3218f09915
FROM tensorflow/tensorflow:2.9.1-gpu@sha256:a34c2420739cd5a7b5662449bc21eb32d3d1c98063726ae2bd7db819cc93d72f  
WORKDIR /usr/app

#CV2 dependencies
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

#entrypoint dependencies
RUN apt-get install gosu -y

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD [ "python", "main.py"]
