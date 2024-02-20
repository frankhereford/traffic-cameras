FROM ubuntu:latest

# Update Ubuntu packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean

# normal stuff
RUN apt-get install -y vim aptitude python3-pip magic-wormhole file ffmpeg

# install youtube downloader
RUN python3 -m pip install -U yt-dlp vtt_to_srt3 redis moviepy psycopg2-binary

RUN mkdir /application
COPY . /application
WORKDIR /application

ENTRYPOINT ["/application/download_and_serve.py"]
#ENTRYPOINT ["/bin/bash"]
