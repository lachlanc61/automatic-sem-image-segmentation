FROM tensorflow/tensorflow:2.11.1@sha256:1f5ec29ae06feea88b4b0fa3b2cfee9ee79a940fe293071307e50d3218f09915
WORKDIR /usr/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD [ "python", "main.py"]
