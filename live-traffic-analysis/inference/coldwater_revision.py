import os
import redis
import torch
import ffmpeg
import requests
import psycopg2
import subprocess
import numpy as np
from PIL import Image
import psycopg2.extras
from dotenv import load_dotenv
# from transformers import DetrImageProcessor, DetrForObjectDetection

fps = 30

load_dotenv()

try:
    db = psycopg2.connect(
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT"),
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    cursor_factory=psycopg2.extras.RealDictCursor,
    )

    cursor = db.cursor()
    # Print PostgreSQL Connection properties
    print(db.get_dsn_parameters(), "\n")

    # Print PostgreSQL version
    cursor.execute("SELECT version();")
    record = cursor.fetchone()
    print("You are connected to - ", record, "\n")
except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL", error)

redis = redis.Redis(host="localhost", port=6379, db=0)

def hls_frame_generator(hls_url):
    # Set up the ffmpeg command to capture the stream
    command = (
        ffmpeg.input(hls_url, format="hls", loglevel="quiet", vcodec="h264_cuvid")
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", r=fps)
        # .global_args("-loglevel", "quiet")
        .global_args("-re")
        .compile()
    )
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    while True:
        # Read 1920*1080*3 bytes (= 1 frame)
        in_bytes = process.stdout.read(1920 * 1080 * 3)
        if not in_bytes:
            break

        frame = np.frombuffer(in_bytes, np.uint8).reshape([1080, 1920, 3])

        yield frame

    process.terminate()

def stream_frames_to_rtmp(rtmp_url, frame_generator):
    command = (
        ffmpeg.input(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="1920x1080", framerate=fps
        )  
        .output(
            rtmp_url, format="flv", vcodec="h264_nvenc", pix_fmt="yuv420p", r=fps,
            video_bitrate="2M", maxrate="5M", bufsize="1000k", g=48
        )  # Configure output
        .overwrite_output()
        .compile()
    )

    process = subprocess.Popen(command, stdin=subprocess.PIPE)


    for frame in frame_generator:

        # image = Image.frombytes('RGB', (1920, 1080), frame, 'raw')
        # # print("Image dimensions:", image.size)

        # inputs = processor(images=image, return_tensors="pt")
        # inputs = inputs.to(device)
        # outputs = model(**inputs)


        annotated_frame = frame.copy()



        process.stdin.write(annotated_frame.tobytes())

    process.stdin.close()
    process.wait()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
# model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
# model = model.to(device)


hls_url = "http://10.0.3.228:8080/memfs/9ea806cb-a214-4971-8b29-76cc9fc9de75.m3u8"
frame_generator = hls_frame_generator(hls_url)

rtmp_url = "rtmp://10.0.3.228/8495ebad-db94-44fb-9a05-45ac7630933a.stream"
stream_frames_to_rtmp(rtmp_url, frame_generator)

