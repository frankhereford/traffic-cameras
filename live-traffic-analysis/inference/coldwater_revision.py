import os
import time
import redis
import torch
import ffmpeg
import requests
import psycopg2
import subprocess
import numpy as np
from PIL import Image
import psycopg2.extras
import supervision as sv
from dotenv import load_dotenv
from torch_tps import ThinPlateSpline
from utilities.transformation import read_points_file
from inference.models.utils import get_roboflow_model
from ultralytics import YOLO

from utilities.sql import (
    prepare_detection,
    insert_detections,
    create_new_session,
    get_class_id,
    compute_speed,
)

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

    cursor = db.cursor()
    session = create_new_session(cursor)

    queued_inserts = 0
    classes = {}
    for frame in frame_generator:

        # result = model.infer(frame)[0]
        # detections = sv.Detections.from_inference(result)
        result = model(frame,  verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = byte_track.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

        # for prediction in result.predictions:
        #     classes[prediction.class_id] = prediction.class_name.title()

        detections_xy = torch.tensor(points).float()
        detections_latlon = tps.transform(detections_xy)

        annotated_frame = frame.copy()


        if not (
            detections.tracker_id is None
            or points is None
            or detections_latlon is None
            or detections.class_id is None
        ):
            
            for tracker_id, point, location, class_id in zip(
                detections.tracker_id, points, detections_latlon, detections.class_id
            ):
                our_class_id = get_class_id(
                    db, cursor, session, int(class_id), str(class_id)
                )
                prepare_detection(
                    tracker_id,
                    our_class_id,
                    point[0],
                    point[1],
                    time.time(),
                    session,
                    location[0],
                    location[1],
                )
                queued_inserts += 1
            if queued_inserts >= 100:
                insert_detections(db, cursor)
                queued_inserts = 0

            speeds = []
            for tracker_id in detections.tracker_id:
                # Try to get the speed from Redis
                speed = redis.get(f"speed:{session}:{tracker_id}")
                if speed is None:
                    # If the speed is not in Redis, compute it and store it in Redis with a 1 second expiration
                    speed = compute_speed(cursor, session, tracker_id, 30)
                    # print("fresh speed: ", speed)
                    # Convert None to 'None' before storing in Redis
                    redis.set(
                        f"speed:{session}:{tracker_id}",
                        speed if speed is not None else "None",
                        ex=1,
                    )
                else:
                    # Decode bytes to string and If the speed is in Redis, convert it to a float if it's not 'None'
                    speed = speed.decode("utf-8")
                    speed = float(speed) if speed != "None" else None
                speeds.append(speed)

            labels = [
                # f"Class: {classes[class_id]} #{tracker_id}"
                f"Class: {class_id} #{tracker_id}"
                + (f", Speed: {speed:.1f} MPH" if speed is not None else "")
                for tracker_id, class_id, speed in zip(
                    detections.tracker_id, detections.class_id, speeds
                )
            ]
            annotated_frame = frame.copy()

            annotated_frame = round_box_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
            )

            annotated_frame = dot_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
            )


            annotated_frame = label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels,
            )

            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

            # annotated_frame = ellipse_annotator.annotate(
            #     scene=annotated_frame,
            #     detections=detections,
            # )






        process.stdin.write(annotated_frame.tobytes())

    process.stdin.close()
    process.wait()




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

coordinates = read_points_file("./gcp/coldwater_mi.points")
tps = ThinPlateSpline(0.5)
tps.fit(coordinates["image_coordinates"], coordinates["map_coordinates"])

# model = get_roboflow_model("yolov8s-640")
model = YOLO("yolov8n.pt")
resolution_wy = (1920, 1080)
byte_track = sv.ByteTrack(frame_rate=15)
thickness = sv.calculate_dynamic_line_thickness(resolution_wh=resolution_wy)
text_scale = sv.calculate_dynamic_text_scale(resolution_wh=resolution_wy)
round_box_annotator = sv.RoundBoxAnnotator(thickness=thickness)
dot_annotator = sv.DotAnnotator(position=sv.Position.BOTTOM_CENTER, radius=10)
label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=2)
trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=20, position=sv.Position.BOTTOM_CENTER)
ellipse_annotator = sv.EllipseAnnotator(
    thickness=thickness,
    # start_angle=0,
    # end_angle=360,
)
smoother = sv.DetectionsSmoother()




hls_url = "http://10.0.3.228:8080/memfs/9ea806cb-a214-4971-8b29-76cc9fc9de75.m3u8"
frame_generator = hls_frame_generator(hls_url)

rtmp_url = "rtmp://10.0.3.228/8495ebad-db94-44fb-9a05-45ac7630933a.stream"
stream_frames_to_rtmp(rtmp_url, frame_generator)

