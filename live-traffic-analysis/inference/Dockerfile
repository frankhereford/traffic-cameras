FROM ubuntu:latest

RUN apt-get update && apt-get install -y python3 python3-pip vim ffmpeg libgl1-mesa-glx libglib2.0-0 aptitude

RUN mkdir /app
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt

ENTRYPOINT ["/bin/bash"]