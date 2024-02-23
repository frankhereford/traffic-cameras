#!/usr/bin/env python3

import os
import time
import psycopg2
import argparse
import psycopg2.extras
from tqdm import tqdm
from dotenv import load_dotenv
import redis as redis_library


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
    parser = argparse.ArgumentParser(description="")
    args = parser.parse_args()
    return args


def main():
    db, redis = setup_service_handles()
    # args = receive_arguments()
    with db.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
        cursor.execute("SELECT COUNT(*) FROM detections.tracks")
        total_rows = cursor.fetchone()[0]

        cursor.execute("SELECT tracker_id FROM detections.tracks")
        for row in tqdm(cursor, total=total_rows, desc="Processing rows"):
            tracker_id = row["tracker_id"]
            # print(f"Tracker ID: {tracker_id}")

            with db.cursor(cursor_factory=psycopg2.extras.DictCursor) as inner_cursor:
                # Select all detections.detections with the current tracker_id
                inner_cursor.execute(
                    """
                    SELECT detections.id
                    FROM detections.detections 
                    JOIN detections.frames ON detections.frame_id = frames.id
                    WHERE tracker_id = %s
                    ORDER BY frames.time ASC
                    """,
                    (tracker_id,),
                )
                detections = inner_cursor.fetchall()

                for detection in detections:
                    # print(f"Detection count: {detection['count']}")
                    pass


if __name__ == "__main__":
    main()
