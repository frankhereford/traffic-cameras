import supervision as sv
from inference.models.utils import get_roboflow_model

import cv2
import ffmpeg
import subprocess
import numpy as np
import os
import time
import subprocess



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
directory = "./frames"

# Create the directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

# Example usage
hls_url = "http://10.10.10.97:8080/memfs/8bd9ac69-e88e-4f6c-a054-5a4176d597e3.m3u8"
frame_generator = hls_frame_generator(hls_url)

resolution_wy = (1920, 1080)
thickness = sv.calculate_dynamic_line_thickness(resolution_wh=resolution_wy)

model = get_roboflow_model("yolov8s-640")

bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=4)

# frame_number = 0
# start_time = time.time()

# for frame in frame_generator:
#     current_time = time.time()
#     elapsed_time = current_time - start_time
#     fps = frame_number / elapsed_time if elapsed_time > 0 else 0
#     print("Frame: ", frame_number, "FPS: ", fps)
#     frame_number += 1

#     result = model.infer(frame)[0]
#     detections = sv.Detections.from_inference(result)

#     annotated_frame = frame.copy()
#     annotated_frame = bounding_box_annotator.annotate(
#         scene=annotated_frame, detections=detections
#     )

    # print("frame write out")
    # cv2.imwrite(f"./frames/frame_{i}.jpg", annotated_frame)



def stream_frames_to_rtmp(rtmp_url, frame_generator):
    # FFmpeg command for streaming to RTMP
    command = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='1920x1080', framerate=30)  # Set input specifications
        .output(rtmp_url, format='flv', vcodec='libx264', pix_fmt='yuv420p')  # Configure output
        .overwrite_output()
        .compile()
    )

    # Start the FFmpeg process for streaming
    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    for frame in frame_generator:
        # Process the frame (your existing processing logic)
        # ...

        result = model.infer(frame)[0]
        detections = sv.Detections.from_inference(result)

        annotated_frame = frame.copy()
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )

        # Write the annotated frame to FFmpeg's stdin
        process.stdin.write(annotated_frame.tobytes())

    # Close the FFmpeg process
    process.stdin.close()
    process.wait()

rtmp_url = 'rtmp://10.10.10.97/ebb55a3f-2eee-4070-b556-6da4aed2a92a.stream'
stream_frames_to_rtmp(rtmp_url, frame_generator)