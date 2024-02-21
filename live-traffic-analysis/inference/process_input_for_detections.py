#!/usr/bin/env python3

import os
import argparse
import psycopg2
from tqdm import tqdm
import psycopg2.extras
import supervision as sv
import redis as redis_library
from dotenv import load_dotenv


def setup_service_handles():
    load_dotenv()

    redis = redis_library.Redis(host="localhost", port=6379, db=0)

    db = None
    try:
        db = psycopg2.connect(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT"),
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            cursor_factory=psycopg2.extras.RealDictCursor,
        )

    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)

    return db, redis


def receive_arguments():
    parser = argparse.ArgumentParser(
        description="Process new video and record detections as quickly as possible"
    )

    args = parser.parse_args()
    return args


def get_a_job(redis):
    _, value = redis.brpop("downloaded-videos-queue")
    # Decode bytes object to string
    value = value.decode("utf-8")
    print("Received job: ", value)
    return value


def get_frame_generator(video):
    video_path = f"./input_media/{video}"
    frame_generator = sv.get_video_frames_generator(source_path=video_path)
    return frame_generator


def get_video_information(video):
    video_path = f"./input_media/{video}"
    information = sv.VideoInfo.from_video_path(video_path)
    return information


def main():
    db, redis = setup_service_handles()
    args = receive_arguments()
    while True:
        job = get_a_job(redis)
        print(f"Processing video: {job}")
        information = get_video_information(job)
        print(f"Video information: {information}")
        input = get_frame_generator(job)
        for frame in tqdm(input, total=information.total_frames):
            pass
            # annotated_frame = self.process_frame(frame)
            # sink.write_frame(annotated_frame)


if __name__ == "__main__":
    main()
