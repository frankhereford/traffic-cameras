import supervision as sv
from inference.models.utils import get_roboflow_model

import cv2
import ffmpeg
import subprocess
import numpy as np
import os


def hls_frame_generator(hls_url):
    # Set up the ffmpeg command to capture the stream
    command = (
        ffmpeg.input(hls_url, format="hls", loglevel="quiet")
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .compile()
    )

    # Start the ffmpeg process
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    while True:
        # Read 1920*1080*3 bytes (= 1 frame)
        in_bytes = process.stdout.read(1920 * 1080 * 3)
        if not in_bytes:
            break

        # Transform the byte read into a numpy array
        frame = np.frombuffer(in_bytes, np.uint8).reshape([1080, 1920, 3])

        yield frame

    # Close the process
    process.terminate()


# The directory to be created
directory = "/app/frames"

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

# Example usage
hls_url = "http://media:8080/memfs/f1adde7d-4364-4a62-ba7d-700766d7f4e2.m3u8"
frame_generator = hls_frame_generator(hls_url)

# for i in range(256):
#     frame = next(frame_generator)
#     # Save the frame as a JPEG file
#     # cv2.imwrite(f"/app/frames/frame_{i}.jpg", frame)

model = get_roboflow_model("yolov8s-640")

bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=4)

i = 0
for frame in frame_generator:
    if i >= 25:
        break

    result = model.infer(frame)[0]
    detections = sv.Detections.from_inference(result)

    annotated_frame = frame.copy()
    annotated_frame = bounding_box_annotator.annotate(
        scene=annotated_frame, detections=detections
    )

    print("frame write out")
    cv2.imwrite(f"/app/frames/frame_{i}.jpg", annotated_frame)
    i += 1
