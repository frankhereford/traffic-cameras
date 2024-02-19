#!/usr/bin/env python3

import argparse
import subprocess
import datetime
import signal
import time
import redis

target_video_length = 20
factor_for_waiting = 6


def send_sigint_to_process(process):
    process.send_signal(signal.SIGINT)


def download_video(video_id):
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_filename = f"media/{video_id}-{timestamp}.mp4"
    command = f"timeout --signal=SIGINT {target_video_length}s yt-dlp {youtube_url} -o {output_filename}"
    print("command: ", command)

    # Start the subprocess
    process = subprocess.Popen(command, shell=True)

    # Wait for the subprocess to finish
    process.wait()

    # Connect to the Redis server
    r = redis.Redis(host="redis", port=6379, db=0)

    # Set the key
    r.set("last-downloaded-video", f"{video_id}-{timestamp}.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download video media from youtube.")
    parser.add_argument(
        "video_id", type=str, help="The ID of the YouTube video.", default="B0YjuKbVZ5w"
    )

    args = parser.parse_args()
    while True:
        download_video(args.video_id)
        print("sleeping: ", (target_video_length * factor_for_waiting) + 5)
        time.sleep((target_video_length * factor_for_waiting) + 5)
