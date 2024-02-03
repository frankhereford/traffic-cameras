#!/usr/bin/env python3

import argparse
import subprocess


def stream_video(video_id):
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    command = f"yt-dlp -f best {youtube_url} -o - | ffmpeg -i - -f rtsp rtsp://localhost:8554/trafficCamera"
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stream a YouTube video to an RTSP server."
    )
    parser.add_argument(
        "video_id", type=str, help="The ID of the YouTube video.", default="B0YjuKbVZ5w"
    )

    args = parser.parse_args()
    stream_video(args.video_id)
