#!/usr/bin/env python3

import requests
import os
import time
from datetime import datetime
import ffmpeg


def download_image(url, folder):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Get current time for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{folder}/{timestamp}.jpg"

    # Download and save the image
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
        return filename
    else:
        print(f"Failed to download image: {response.status_code}")
        return None


def create_video(folder):
    # Define the output video filename
    video_filename = f"{folder}/timelapse.mp4"

    # Use ffmpeg to create a video from the images
    (
        ffmpeg.input(f"{folder}/*.jpg", pattern_type="glob", framerate=30)
        .output(
            video_filename,
            vcodec="libx264",
            crf=25,
            pix_fmt="yuv420p",
        )
        .overwrite_output()
        .run()
    )


def main():
    url = "https://cctv.austinmobility.io/image/30.jpg"
    folder = "downloaded_images"
    interval = 5 * 60  # 5 minutes

    while True:
        filename = download_image(url, folder)
        if filename:
            print(f"Downloaded {filename}")
            create_video(folder)
            print(f"Video updated.")
        time.sleep(interval)


if __name__ == "__main__":
    main()
