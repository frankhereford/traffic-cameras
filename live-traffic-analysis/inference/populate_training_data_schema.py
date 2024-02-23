#!/usr/bin/env python3

import os
import time
import psycopg2
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
    parser = argparse.ArgumentParser(
        description="Process new video and record detections as quickly as possible"
    )

    parser.add_argument(
        "-d",
        "--detections",
        action="store_true",
        help="Flag to enable or disable storing detections",
    )

    args = parser.parse_args()
    return args


def main():
    db, redis = setup_service_handles()
    # args = receive_arguments()
    with db.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
        cursor.execute("SELECT * FROM detections.tracks")
        for row in tqdm(cursor, desc="Processing rows"):
            time.sleep(0.001)


if __name__ == "__main__":
    main()
