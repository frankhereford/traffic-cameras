#!/usr/bin/env python3

import os
import argparse
import psycopg2
import psycopg2.extras
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


def main():
    db, redis = setup_service_handles()
    args = receive_arguments()


if __name__ == "__main__":
    main()
