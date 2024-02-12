#!/usr/bin/env python3

import subprocess
import ffmpeg
import numpy as np
import redis
import time

fps = 30


def mp4_frame_generator(mp4_file, fps="30"):
    # Set up the ffmpeg command to read frames from the mp4 file
    command = (
        ffmpeg.input(mp4_file, format="mp4").output(
            "pipe:", format="rawvideo", pix_fmt="rgb24", r=fps
        )
        # .global_args("-loglevel", "quiet")
        .compile()
    )
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    while True:
        # Read bytes for one frame; adjust the number based on resolution
        # For example, for 1920x1080 resolution, use 1920 * 1080 * 3 bytes
        in_bytes = process.stdout.read(1920 * 1080 * 3)
        if not in_bytes:
            break

        frame = np.frombuffer(in_bytes, np.uint8).reshape([1080, 1920, 3])

        yield frame

    process.terminate()


if __name__ == "__main__":
    r = redis.Redis(host="localhost", port=6379, db=0)

    rtmp_url = "rtmp://10.0.3.228/dc05b5e7-3ef8-455f-8c62-547d2e9f2c18.stream"

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
            video_bitrate="1M",
            maxrate="2M",
            bufsize="1000k",
            g=48,
        )  # Configure output
        .overwrite_output()
        # .global_args("-loglevel", "quiet")
        .compile()
    )

    process = subprocess.Popen(command, stdin=subprocess.PIPE)

    while True:
        print("\nNew Video Time\n")
        last_processed_video = r.get("last-processed-video")
        if last_processed_video is not None:
            last_processed_video = last_processed_video.decode("utf-8")

        path = f"./output_media/processed-media/{last_processed_video}"
        generator = mp4_frame_generator(path)
        start_time = time.time()
        frame_count = 0
        for frame in generator:
            frame_count += 1
            if frame_count % 30 == 0:
                last_processed_video_check = r.get("last-processed-video")
                if last_processed_video_check is not None:
                    last_processed_video_check = last_processed_video_check.decode(
                        "utf-8"
                    )
                if last_processed_video_check != last_processed_video:
                    print("\nBreaking for new video\n")
                    break

            annotated_frame = frame.copy()
            process.stdin.write(annotated_frame.tobytes())

            time_taken = time.time() - start_time
            if time_taken < 1.0 / fps:
                time.sleep(1.0 / fps - time_taken)

            start_time = time.time()
