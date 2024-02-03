#!/usr/bin/env python3

import argparse
import subprocess


def stream_video(video_id):
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    command = f"yt-dlp -f best {youtube_url} -o - | ffmpeg -stream_loop -1 -re -i pipe:0 -c:v libx264 -c:a aac -ar 44100 -strict experimental -f flv rtmp://10.10.10.97/8bd9ac69-e88e-4f6c-a054-5a4176d597e3.stream"
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