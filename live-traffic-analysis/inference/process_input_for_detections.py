#!/usr/bin/env python3

import os
import json
import pytz
import uuid
import torch
import pickle
import hashlib
import argparse
import psycopg2
import numpy as np
from tqdm import tqdm
import psycopg2.extras
import supervision as sv
from ultralytics import YOLO
import redis as redis_library
from datetime import datetime, timedelta
from dotenv import load_dotenv
from torch_tps import ThinPlateSpline
from utilities.transformation import read_points_file


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


def get_a_job(redis):
    _, value = redis.brpop("downloaded-videos-queue")
    print("value: ", value)
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


def hash_frame(frame):
    frame_bytes = frame.tobytes()
    hash_object = hashlib.sha1(frame_bytes)
    hex_dig = hash_object.hexdigest()
    return hex_dig


def make_detections(model, tracker, frame):
    result = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = tracker.update_with_detections(detections)
    return result, detections


def get_supervision_objects():
    model = YOLO("yolov8m.pt")
    tracker = sv.ByteTrack()
    return model, tracker


def get_image_space_locations(detections):
    points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    tensor = torch.tensor(points).float()
    tensor = tensor.to("cuda")
    return tensor


def get_map_space_locations(tps, image_space):
    # ! i'm not sure this is accelerated
    map_space = tps.transform(image_space)
    return map_space


def get_tps():
    coordinates = read_points_file("./gcp/coldwater_mi.points")
    tps = ThinPlateSpline(0.5)
    tps.fit(coordinates["image_coordinates"], coordinates["map_coordinates"])
    reverse_tps = ThinPlateSpline(0.5)
    reverse_tps.fit(coordinates["map_coordinates"], coordinates["image_coordinates"])
    return tps, reverse_tps


def get_datetime_from_job(job):
    # Extract the date and time from the job string
    date_time_str = job.split("-")[1:]
    date_time_str = "-".join(date_time_str).split(".")[0]

    # Convert the date and time string to a datetime object
    date_time_obj = datetime.strptime(date_time_str, "%Y%m%d-%H%M%S")

    return date_time_obj


def get_frame_duration(information):
    # Calculate the duration of one frame in seconds
    frame_duration_seconds = 1 / information.fps

    # Convert the duration to a timedelta object
    frame_duration = timedelta(seconds=frame_duration_seconds)

    return frame_duration


def prepare_detection(
    records_to_insert,
    tracker_id,
    class_id,
    image_x,
    image_y,
    timestamp,
    session_id,
    longitude,
    latitude,
    frame_hash,
):
    # Convert the timestamp to Central Time
    central = pytz.timezone("America/Chicago")
    timestamp = timestamp.astimezone(central)

    record_to_insert = (
        int(tracker_id),
        int(image_x),
        int(image_y),
        timestamp,
        session_id,
        float(longitude),
        float(latitude),
        int(class_id),
        frame_hash,
    )
    records_to_insert.append(record_to_insert)

    return records_to_insert


def create_new_session(db):
    # Generate a fresh UUID
    new_uuid = uuid.uuid4()

    # Insert a new session record and return the id
    insert_query = """INSERT INTO sessions (uuid) VALUES (%s) RETURNING id;"""

    with db.cursor() as cursor:
        cursor.execute(insert_query, (str(new_uuid),))

        # Fetch the id of the newly inserted record
        session_id = cursor.fetchone()
        return session_id["id"]


def get_class_id(db, redis, session, results, detections):
    select_query = """
    SELECT id FROM classes 
    WHERE session_id = %s AND class_id = %s AND class_name = %s
    """

    insert_query = """
    INSERT INTO classes (session_id, class_id, class_name) 
    VALUES (%s, %s, %s) RETURNING id
    """

    class_names = [results.names.get(class_id) for class_id in detections.class_id]

    ids = []
    for class_id, class_name in zip(detections.class_id, class_names):
        # Create a unique key for each class_id and class_name pair
        key = f"{session}:{class_id}:{class_name}"

        # Try to get the id from Redis
        id = redis.get(key)

        if id is None:
            # If the id is not in Redis, get it from the database
            with db.cursor() as cursor:
                cursor.execute(select_query, (session, int(class_id), class_name))
                result = cursor.fetchone()

                if result:
                    id = result["id"]
                else:
                    # If the record does not exist, insert it
                    cursor.execute(insert_query, (session, int(class_id), class_name))
                    db.commit()
                    result = cursor.fetchone()
                    if result:
                        id = result["id"]
                    else:
                        raise Exception(
                            "Failed to insert new record into classes table"
                        )

                # Store the id in Redis
                redis.set(key, json.dumps(id))

        else:
            # If the id is in Redis, decode it
            id = json.loads(id)

        ids.append(id)

    return ids


def get_class_names(class_ids, result):
    return [result.names.get(class_id) for class_id in class_ids]


def do_bulk_insert(db, records_to_insert):
    insert_query = """
    INSERT INTO detections (tracker_id, image_x, image_y, timestamp, session_id, location, class_id, frame_hash) 
    VALUES (%s, %s, %s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 2253), %s, %s)
    """
    with db.cursor() as cursor:
        cursor.executemany(insert_query, records_to_insert)
    db.commit()
    # Clear the records list
    records_to_insert.clear()
    return records_to_insert


def process_detections(
    detections,
    detection_classes,
    image_space_locations,
    map_space_locations,
    time,
    session,
    hash,
):
    records_to_insert = []

    for (
        tracker_id,
        detection_class,
        image_space_location,
        map_space_location,
    ) in zip(
        detections.tracker_id,
        detection_classes,
        image_space_locations,
        map_space_locations,
    ):
        records_to_insert = prepare_detection(
            records_to_insert,
            tracker_id,
            detection_class,
            image_space_location[0],
            image_space_location[1],
            time,
            session,
            map_space_location[0],
            map_space_location[1],
            hash,
        )

    return records_to_insert


def main():
    db, redis = setup_service_handles()
    args = receive_arguments()
    while True:
        job = get_a_job(redis)
        print("Processing job: ", job)
        session = create_new_session(db)
        information = get_video_information(job)
        input = get_frame_generator(job)
        model, tracker = get_supervision_objects()
        tps, inverse_tps = get_tps()
        time = get_datetime_from_job(job)
        frame_duration = get_frame_duration(information)
        records_to_insert = []
        for frame in tqdm(input, total=information.total_frames):
            results, detections = make_detections(model, tracker, frame)
            if args.detections == True:
                hash = hash_frame(frame)
                detection_classes = get_class_id(
                    db, redis, session, results, detections
                )
                image_space_locations = get_image_space_locations(detections)
                map_space_locations = get_map_space_locations(
                    tps, image_space_locations
                )
                records_to_insert.extend(
                    process_detections(
                        detections,
                        detection_classes,
                        image_space_locations,
                        map_space_locations,
                        time,
                        session,
                        hash,
                    )
                )
                if len(records_to_insert) >= 10000:
                    records_to_insert = do_bulk_insert(db, records_to_insert)
                time += frame_duration
        # process the tail
        if args.detections == True:
            records_to_insert = do_bulk_insert(db, records_to_insert)


if __name__ == "__main__":
    main()
