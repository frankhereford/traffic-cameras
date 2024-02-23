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
    with db.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
        cursor.execute("SELECT COUNT(*) FROM detections.tracks")
        total_rows = cursor.fetchone()[0]

        cursor.execute("SELECT tracker_id FROM detections.tracks")
        for row in tqdm(cursor, total=total_rows, desc="Processing rows"):
            tracker_id = row["tracker_id"]

            with db.cursor(cursor_factory=psycopg2.extras.DictCursor) as inner_cursor:
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

                # Divide detections into groups of 60
                groups = [detections[n : n + 60] for n in range(0, len(detections), 60)]

                # Discard groups with less than 60 detections
                groups = [group for group in groups if len(group) == 60]

                # Discard extra detections from the start and end of the list
                if len(groups) > 1:
                    groups = groups[1:-1]

                for i, group in enumerate(groups):
                    # Insert a new record into the training_data.samples table
                    inner_cursor.execute(
                        """
                        INSERT INTO training_data.samples (tracker_id)
                        VALUES (%s)
                        RETURNING id
                        """,
                        (tracker_id,),
                    )
                    sample_id = inner_cursor.fetchone()[0]

                    # Buffer for the queries
                    query_buffer = []

                    # For each detection in the group, prepare the parameters for the INSERT query
                    for detection in group:
                        query_buffer.append((sample_id, detection["id"]))

                        # If there are more than 1000 queries in the buffer, execute them with executemany
                        if len(query_buffer) >= 1000:
                            inner_cursor.executemany(
                                """
                                INSERT INTO training_data.detections (sample_id, detection_id)
                                VALUES (%s, %s)
                                """,
                                query_buffer,
                            )
                            query_buffer = []

                    # Execute any remaining queries in the buffer
                    if query_buffer:
                        inner_cursor.executemany(
                            """
                            INSERT INTO training_data.detections (sample_id, detection_id)
                            VALUES (%s, %s)
                            """,
                            query_buffer,
                        )

                    db.commit()


if __name__ == "__main__":
    main()
