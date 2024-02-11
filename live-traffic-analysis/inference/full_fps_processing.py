#!/usr/bin/env python3

import argparse
import datetime
from utilities.videoprocessor import VideoProcessor
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import os


def main(input_file, output_file, db):
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    image_processor = VideoProcessor(input=input_file, output=output_file, db=db)
    image_processor.process_video()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some files.")

    parser.add_argument(
        "-i",
        "--input",
        default="../download/media/ByED80IKdIU.mp4",
        help="Input file name",
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = f"./output_media/{timestamp}.mp4"

    parser.add_argument(
        "-o", "--output", default=default_output, help="Output file name"
    )

    args = parser.parse_args()

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

        cursor = db.cursor()
        # print(db.get_dsn_parameters(), "\n")

        # cursor.execute("SELECT version();")
        # record = cursor.fetchone()
        # print("You are connected to - ", record, "\n")
    except (Exception, psycopg2.Error) as error:
        print("Error while connecting to PostgreSQL", error)

    main(args.input, args.output, db)
