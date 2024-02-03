#!/usr/bin/env python3

import argparse
import os


def download_video(video_id):
    command = f"yt-dlp -f 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4' https://www.youtube.com/watch?v={video_id} -o {video_id}.mp4"
    os.system(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a YouTube video.")
    parser.add_argument(
        "video_id", type=str, help="The ID of the YouTube video.", default="B0YjuKbVZ5w"
    )

    args = parser.parse_args()
    download_video(args.video_id)
