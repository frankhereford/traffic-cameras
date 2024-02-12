#!/usr/bin/env python3

import supervision as sv
import os
import time
import subprocess
import ffmpeg
import numpy as np
import time


class FolderWatcher:
    def __init__(self, folder_path, static_for_n_frames=5):
        self.folder_path = folder_path
        self.static_for_n_frames = static_for_n_frames
        self.current_file = None
        self.current_generator = None
        self.frame_number = 0
        self.files_info = (
            {}
        )  # Store info as {filename: {'first_seen': time, 'last_seen': time, ...}}

    def update(self):
        current_files = {f for f in os.listdir(self.folder_path) if f.endswith(".mp4")}
        current_time = time.time()

        for file in current_files:
            file_path = os.path.join(self.folder_path, file)
            file_size = os.path.getsize(file_path)

            if file not in self.files_info:
                # New file found
                self.files_info[file] = {
                    "first_seen": current_time,
                    "last_seen": current_time,
                    "last_seen_size": file_size,
                    "number_times_seen_at_current_size": 1,
                }
            else:
                # Update existing file info
                file_info = self.files_info[file]
                file_info["last_seen"] = current_time

                if file_info["last_seen_size"] == file_size:
                    file_info["number_times_seen_at_current_size"] += 1
                else:
                    file_info["last_seen_size"] = file_size
                    file_info["number_times_seen_at_current_size"] = 1

        # Remove info about files that no longer exist
        self.files_info = {
            file: info
            for file, info in self.files_info.items()
            if file in current_files
        }

    def get_files_info(self):
        return self.files_info

    def get_current_file(self):
        if (self.frame_number % 60) == 0:
            self.update()
        last_file = None
        for file, info in self.files_info.items():
            if (
                info["number_times_seen_at_current_size"] >= self.static_for_n_frames
                and info["last_seen_size"] > 30 * 1024 * 1024
            ):
                if last_file is None or file > last_file:
                    last_file = file
        return last_file


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


fps = 30

# Example usage:
watcher = FolderWatcher("./output_media/")
while watcher.get_current_file() is None:
    time.sleep(1)

rtmp_url = "rtmp://10.0.3.228/8495ebad-db94-44fb-9a05-45ac7630933a.stream"

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

while True:
    frame_count = 0
    current_file = watcher.get_current_file()
    print("Current file:", current_file)
    path = "./output_media/" + current_file
    generator = mp4_frame_generator(path)
    start_time = time.time()
    for frame in generator:
        frame_count += 1
        if frame_count % 30 == 0:  # Check every 100 frames
            new_file = watcher.get_current_file()
            if new_file != current_file:  # If the current file has changed
                break

        annotated_frame = frame.copy()
        process.stdin.write(annotated_frame.tobytes())

        time_taken = time.time() - start_time
        if time_taken < 1.0 / fps:
            time.sleep(1.0 / fps - time_taken)

        start_time = time.time()
