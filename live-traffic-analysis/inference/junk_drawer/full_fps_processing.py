#!/usr/bin/env python3

import argparse
import datetime
from utilities.videoprocessor import VideoProcessor
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import os
import redis

from utilities.locationhistory import VehicleHistoryAndInference


def main(input_file, output_file, db, location_tracker):
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    image_processor = VideoProcessor(
        input=input_file, output=output_file, db=db, location_tracker=location_tracker
    )
    image_processor.process_video()
    # location_tracker.pretty_print_all_history()


if __name__ == "__main__":
    location_tracker = VehicleHistoryAndInference()

    load_dotenv()

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

    r = redis.Redis(host="localhost", port=6379, db=0)

    # while True:
    #     # Block until a value is available in the queue
    #     _, value = r.brpop("downloaded-videos-queue")
    #     print("Popped value: ", value)

    parser = argparse.ArgumentParser(description="Process some files.")

    last_downloaded_video = r.get("last-downloaded-video")
    if last_downloaded_video is not None:
        last_downloaded_video = last_downloaded_video.decode("utf-8")

    input = f"./input_media/{last_downloaded_video}"

    parser.add_argument(
        "-i",
        "--input",
        default=input,
        help="Input file name",
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_filename = f"{last_downloaded_video[:-4]}-{timestamp}.mp4"
    default_output = f"./output_media/processed-media/{output_filename}"

    parser.add_argument(
        "-o", "--output", default=default_output, help="Output file name"
    )

    args = parser.parse_args()

    main(args.input, args.output, db, location_tracker)

    r.set("last-processed-video", output_filename)
