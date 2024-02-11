#!/usr/bin/env python3

import argparse
import subprocess
import datetime


import datetime


def download_video(video_id):
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_filename = f"media/media-from-camera/{video_id}-{timestamp}.mp4"
    print("output_filename:", output_filename)
    # command = f'yt-dlp {youtube_url} -o {output_filename} --postprocessor-args "-ss 00:00:00 -t 00:01:00"'
    # subprocess.run(command, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download video media from youtube.")
    parser.add_argument(
        "video_id", type=str, help="The ID of the YouTube video.", default="B0YjuKbVZ5w"
    )

    args = parser.parse_args()
    download_video(args.video_id)
