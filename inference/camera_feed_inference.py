import supervision as sv
from inference.models.utils import get_roboflow_model

# from collections import defaultdict, deque

import ffmpeg
import subprocess
import numpy as np
import time
import subprocess
import torch
from torch_tps import ThinPlateSpline

import psycopg2.extras
import redis

from utilities.transformation import read_points_file
from utilities.sql import (
    insert_detection,
    create_new_session,
    get_class_id,
    compute_speed,
)

import psycopg2

try:
    db = psycopg2.connect(
        user="postgres",
        password="cameras",
        port="5431",
        host="localhost",
        database="live_cameras",
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
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", r=15)
        .global_args("-loglevel", "quiet")
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


def stream_frames_to_rtmp(rtmp_url, frame_generator, session_id, tps):
    command = (
        ffmpeg.input(
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="1920x1080", framerate=15
        )  # Set input specifications
        .output(
            rtmp_url, format="flv", vcodec="h264_nvenc", pix_fmt="yuv420p"
        )  # Configure output
        .overwrite_output()
        .global_args("-loglevel", "quiet")
        .compile()
    )

    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    classes = {}
    # coordinate_history = defaultdict(lambda: deque(maxlen=30))

    for frame in frame_generator:
        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = byte_track.update_with_detections(detections)
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

        for prediction in result.predictions:
            classes[prediction.class_id] = prediction.class_name.title()

        detections_xy = torch.tensor(points).float()
        detections_latlon = tps.transform(detections_xy)

        # print(detections_latlon)

        # for tracker_id, [longitude, latitude] in zip(
        #     detections.tracker_id, detections_latlon
        # ):
        #     coordinate_history[tracker_id].append(
        #         {
        #             "time": time.time(),
        #             "longitude": float(longitude),
        #             "latitude": float(latitude),
        #         }
        #     )

        # Insert each detection into the database
        for tracker_id, point, location, class_id in zip(
            detections.tracker_id, points, detections_latlon, detections.class_id
        ):
            our_class_id = get_class_id(
                db, cursor, session_id, class_id, classes[class_id]
            )
            insert_detection(
                db,
                cursor,
                tracker_id,
                our_class_id,
                point[0],
                point[1],
                time.time(),
                session_id,
                location[0],
                location[1],
            )

        # labels = [f"#{tracker_id} Class: {class_id}" for tracker_id, class_id in zip(detections.tracker_id, detections.class_id)]
        # labels = [f"x: {int(x)}, y: {int(y)}" for [x, y] in points]
        # labels = [
        #     f"Class: {classes[class_id]} #{tracker_id}, x: {int(point[0])}, y: {int(point[1])}"
        #     for tracker_id, class_id, point in zip(
        #         detections.tracker_id, detections.class_id, points
        #     )
        # ]

        speeds = []
        for tracker_id in detections.tracker_id:
            # Try to get the speed from Redis
            speed = redis.get(f"speed:{session_id}:{tracker_id}")
            if speed is None:
                # If the speed is not in Redis, compute it and store it in Redis with a 2 second expiration
                speed = compute_speed(cursor, session, tracker_id, 20)
                if speed is not None:
                    print("fresh speed: ", speed)
                    redis.set(f"speed:{session_id}:{tracker_id}", speed, ex=2)
                else:
                    print("failed speed read")
                    speeds.append(None)
                    continue
            else:
                # If the speed is in Redis, convert it to a float
                speed = float(speed)
            speeds.append(speed)

        labels = [
            f"Class: {classes[class_id]} #{tracker_id}"
            + (f", Speed: {speed:.1f} MPH" if speed is not None else "")
            for tracker_id, class_id, speed in zip(
                detections.tracker_id, detections.class_id, speeds
            )
        ]
        annotated_frame = frame.copy()

        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels,
        )

        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )

        annotated_frame = ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
        )

        process.stdin.write(annotated_frame.tobytes())

    process.stdin.close()
    process.wait()


cursor = db.cursor()
session = create_new_session(cursor)

print("session: ", session)

torch.set_printoptions(precision=10)

coordinates = read_points_file("./gcp/orange_ca.points")
tps = ThinPlateSpline(0.5)
tps.fit(coordinates["image_coordinates"], coordinates["map_coordinates"])

hls_url = "http://10.10.10.97:8080/memfs/8bd9ac69-e88e-4f6c-a054-5a4176d597e3.m3u8"
frame_generator = hls_frame_generator(hls_url)

model = get_roboflow_model("yolov8s-640")

resolution_wy = (1920, 1080)
byte_track = sv.ByteTrack(frame_rate=15)
thickness = sv.calculate_dynamic_line_thickness(resolution_wh=resolution_wy)
text_scale = sv.calculate_dynamic_text_scale(resolution_wh=resolution_wy)
bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)
trace_annotator = sv.TraceAnnotator(thickness=thickness, trace_length=60)
ellipse_annotator = sv.EllipseAnnotator(
    thickness=thickness,
    # start_angle=0,
    # end_angle=360,
)

rtmp_url = "rtmp://10.10.10.97/ebb55a3f-2eee-4070-b556-6da4aed2a92a.stream"
stream_frames_to_rtmp(rtmp_url, frame_generator, session, tps)
