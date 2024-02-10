#!/usr/bin/env python

import os
import time
import redis
import torch
import ffmpeg
import random
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

fps = 10

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
    print(db.get_dsn_parameters(), "\n")

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


def generate_boolean_list(length):
    return [random.random() <= 1 for _ in range(length)]


def build_keep_list_tensor(data_obj, center_points):
    xyxy_tensor = data_obj.xyxy
    computed_center_points = (xyxy_tensor[:, :2] + xyxy_tensor[:, 2:4]) / 2

    # Initialize an array for the boolean values
    bool_array = []
    for cp in computed_center_points:
        bool_value = True
        for center_x, center_y, distance_threshold in center_points:
            center_tensor = torch.tensor(
                [center_x, center_y], device=xyxy_tensor.device
            )
            dist_to_center = torch.norm(center_tensor - cp)
            if dist_to_center < distance_threshold:
                bool_value = False
                break
        bool_array.append(bool_value)

    return bool_array


def filter_tensors(data_obj, keep_list):
    # Ensuring all tensors in the object are of the same length as keep_list
    for attr in vars(data_obj):
        tensor = getattr(data_obj, attr)
        if torch.is_tensor(tensor) and len(tensor) != len(keep_list):
            raise ValueError("Tensor and keep_list lengths do not match.")

    for attr in vars(data_obj):
        tensor = getattr(data_obj, attr)
        if torch.is_tensor(tensor):
            # Filtering the tensor
            filtered_tensor = tensor[torch.tensor(keep_list)]
            setattr(data_obj, attr, filtered_tensor)

    return data_obj


def stream_frames_to_rtmp(rtmp_url, frame_generator):
    command = (
        ffmpeg.input(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="1920x1080", framerate=fps
        )
        .output(
            rtmp_url,
            format="flv",
            vcodec="h264_nvenc",
            pix_fmt="yuv420p",
            r=fps,
            video_bitrate="2M",
            maxrate="5M",
            bufsize="1000k",
            g=48,
        )  # Configure output
        .overwrite_output()
        # .global_args("-loglevel", "quiet")
        .compile()
    )

    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    cursor = db.cursor()
    session = create_new_session(cursor)

    queued_inserts = 0
    for frame in frame_generator:

        result = model(frame, verbose=False)[0]

        center_points_to_avoid = [
            (1368, 371, 15),  # bank atm
            (1814, 672, 15),  # lower-right flag
            (1700, 458, 15),  # right street closest light
            (1610, 437, 15),  # right street middle light
            (1770, 674, 15),  # lower right flag
            (496, 490, 15),  # left street light, closest
            (1833, 413, 15),  # right street far street light near the pole
            (667, 739, 15),  # lower left light
            (640, 729, 15),  # lower left light
            (38, 432, 10),  # clock
            (637, 731, 25),  # light lower left
            (375, 442, 20),
            (1044, 580, 20),
            (945, 396, 20),
        ]
        keep_list = build_keep_list_tensor(result.boxes, center_points_to_avoid)
        result.boxes = filter_tensors(result.boxes, keep_list)

        # print(result.boxes)

        detections = sv.Detections.from_ultralytics(result)
        detections = byte_track.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        center_points = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)

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
                    db, cursor, session, int(class_id), result.names[class_id]
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
            queue_size = 100
            if queued_inserts >= queue_size:
                # print(f"Inserting {queued_inserts} detections")
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

            class_labels = [
                f"{result.names[class_id].title()} #{tracker_id} (X: {int(point[0])}, Y: {int(point[1])})"
                for tracker_id, class_id, point in zip(
                    detections.tracker_id, detections.class_id, center_points
                )
            ]

            speed_labels = [
                (f"{speed:.1f} MPH" if speed is not None else "calculating...")
                for speed in speeds
            ]

            annotated_frame = frame.copy()

            annotated_frame = round_box_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
            )

            annotated_frame = class_label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=class_labels,
            )
            annotated_frame = speed_label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=speed_labels,
            )

            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )

        process.stdin.write(annotated_frame.tobytes())

    process.stdin.close()
    process.wait()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

coordinates = read_points_file("./gcp/coldwater_mi.points")
tps = ThinPlateSpline(0.5)
tps.fit(coordinates["image_coordinates"], coordinates["map_coordinates"])

model = YOLO("yolov8m.pt")
resolution_wy = (1920, 1080)
byte_track = sv.ByteTrack(frame_rate=fps)
thickness = sv.calculate_dynamic_line_thickness(resolution_wh=resolution_wy)
text_scale = sv.calculate_dynamic_text_scale(resolution_wh=resolution_wy)
round_box_annotator = sv.RoundBoxAnnotator(thickness=2)
dot_annotator = sv.DotAnnotator(
    position=sv.Position.BOTTOM_CENTER, radius=5, color=sv.Color(r=0, g=0, b=0)
)
class_label_annotator = sv.LabelAnnotator(
    text_scale=0.5,
    text_thickness=1,
    text_position=sv.Position.TOP_CENTER,
    text_color=sv.Color(r=0, g=0, b=0),
)
speed_label_annotator = sv.LabelAnnotator(
    text_scale=0.5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
    text_color=sv.Color(r=0, g=0, b=0),
)
trace_annotator = sv.TraceAnnotator(
    thickness=thickness, trace_length=20, position=sv.Position.CENTER
)
ellipse_annotator = sv.EllipseAnnotator(
    thickness=thickness,
    # start_angle=0,
    # end_angle=360,
)
smoother = sv.DetectionsSmoother(length=3)


hls_url = "http://10.0.3.228:8080/memfs/9ea806cb-a214-4971-8b29-76cc9fc9de75.m3u8"
frame_generator = hls_frame_generator(hls_url)

rtmp_url = "rtmp://10.0.3.228/8495ebad-db94-44fb-9a05-45ac7630933a.stream"
stream_frames_to_rtmp(rtmp_url, frame_generator)
