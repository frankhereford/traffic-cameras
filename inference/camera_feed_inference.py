import supervision as sv
from inference.models.utils import get_roboflow_model

import cv2
import ffmpeg
import subprocess
import numpy as np
import os
import time
import subprocess
import json
import torch
from torch_tps import ThinPlateSpline

from utilities.transformation import read_points_file


def hls_frame_generator(hls_url):
    # Set up the ffmpeg command to capture the stream
    command = (
        ffmpeg.input(hls_url, format="hls", loglevel="quiet")
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
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
            "pipe:", format="rawvideo", pix_fmt="rgb24", s="1920x1080", framerate=30
        )  # Set input specifications
        .output(
            rtmp_url, format="flv", vcodec="libx264", pix_fmt="yuv420p"
        )  # Configure output
        .overwrite_output()
        .compile()
    )

    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    for frame in frame_generator:
        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)
        detections = byte_track.update_with_detections(detections)

        # midpoints = convert_to_midpoints(detections.tracker_id, detections.xyxy)
        # print(midpoints)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        print(points)

        # labels = [f"#{tracker_id} Class: {class_id}" for tracker_id, class_id in zip(detections.tracker_id, detections.class_id)]

        annotated_frame = frame.copy()

        annotated_frame = label_annotator.annotate(
            # scene=annotated_frame, detections=detections, labels=labels
            scene=annotated_frame,
            detections=detections,
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


torch.set_printoptions(precision=10)

coordinates = read_points_file("./gcp/orange_ca.points")
tps = ThinPlateSpline(0.5)
tps.fit(coordinates["image_coordinates"], coordinates["map_coordinates"])


hls_url = "http://10.10.10.97:8080/memfs/8bd9ac69-e88e-4f6c-a054-5a4176d597e3.m3u8"
frame_generator = hls_frame_generator(hls_url)

model = get_roboflow_model("yolov8s-640")


resolution_wy = (1920, 1080)
byte_track = sv.ByteTrack(frame_rate=30)
thickness = sv.calculate_dynamic_line_thickness(resolution_wh=resolution_wy)
text_scale = sv.calculate_dynamic_text_scale(resolution_wh=resolution_wy)
bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)
trace_annotator = sv.TraceAnnotator(thickness=thickness)
ellipse_annotator = sv.EllipseAnnotator(
    thickness=thickness,
    # start_angle=0,
    # end_angle=360,
)
# halo_annotator = sv.HaloAnnotator()

rtmp_url = "rtmp://10.10.10.97/ebb55a3f-2eee-4070-b556-6da4aed2a92a.stream"
stream_frames_to_rtmp(rtmp_url, frame_generator)
