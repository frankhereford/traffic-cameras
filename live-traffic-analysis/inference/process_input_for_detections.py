#!/usr/bin/env python3

import os
import json
import pytz
import uuid
import torch
import hashlib
import argparse
import psycopg2
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
    # print("value: ", value)
    # Decode bytes object to string
    value = value.decode("utf-8")
    # print("Received job: ", value)
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
    tracker_id,
    xyxy,
    confidence,
    map_space_location,
    frame_id,
):
    # Extract coordinates and location
    x1, y1, x2, y2 = map(float, xyxy)
    longitude, latitude = map_space_location

    location = f"POINT({longitude} {latitude})"

    record_to_insert = (
        frame_id,
        tracker_id,
        x1,
        y1,
        x2,
        y2,
        float(confidence),
        location,
    )

    return record_to_insert


def create_new_recording(db, job, time):
    insert_query = """INSERT INTO detections.recordings (filename, start_time) VALUES (%s, %s) RETURNING id;"""

    with db.cursor() as cursor:
        cursor.execute(insert_query, (job, time))
        recording_id = cursor.fetchone()
        db.commit()
        print(f"Recording id: {recording_id['id']}")
        return recording_id["id"]


def get_class_ids(db, redis, recording, results, detections):
    select_query = """
    SELECT id FROM detections.classes 
    WHERE recording_id = %s AND ultralytics_id = %s AND ultralytics_name = %s
    """

    insert_query = """
    INSERT INTO detections.classes (recording_id, ultralytics_id, ultralytics_name) 
    VALUES (%s, %s, %s) RETURNING id
    """

    class_names = [results.names.get(class_id) for class_id in detections.class_id]

    ids = []
    for class_id, class_name in zip(detections.class_id, class_names):
        # Create a unique key for each class_id and class_name pair
        key = f"{recording}:{class_id}:{class_name}"

        # Try to get the id from Redis
        id = redis.get(key)

        if id is None:
            # If the id is not in Redis, get it from the database
            with db.cursor() as cursor:
                cursor.execute(select_query, (recording, int(class_id), class_name))
                result = cursor.fetchone()

                if result:
                    id = result["id"]
                else:
                    # If the record does not exist, insert it
                    cursor.execute(insert_query, (recording, int(class_id), class_name))
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
    # print(records_to_insert[0])
    insert_query = """
    INSERT INTO detections.detections (frame_id, tracker_id, x1, y1, x2, y2, confidence, location) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    with db.cursor() as cursor:
        cursor.executemany(insert_query, records_to_insert)
    db.commit()
    # Clear the records list
    records_to_insert.clear()
    return records_to_insert


def process_detections(frame_id, detections, map_space_locations, tracker_ids):
    records_to_insert = []

    for tracker_id, xyxy, confidence, map_space_location in zip(
        tracker_ids,
        detections.xyxy,
        detections.confidence,
        map_space_locations,
    ):
        records_to_insert.append(
            prepare_detection(
                tracker_id,
                xyxy,
                confidence,
                map_space_location,
                frame_id,
            )
        )

    return records_to_insert


def get_frame_id(db, redis, recording, hash, time):
    select_query = """
    SELECT id FROM detections.frames 
    WHERE recording_id = %s AND hash = %s
    """

    insert_query = """
    INSERT INTO detections.frames (recording_id, hash, time) 
    VALUES (%s, %s, %s) RETURNING id
    """

    # Create a unique key for the Redis cache
    key = f"frame:{recording}:{hash}"

    # Try to get the id from Redis
    id = redis.get(key)

    if id is None:
        # If the id is not in Redis, get it from the database
        with db.cursor() as cursor:
            cursor.execute(select_query, (recording, hash))
            result = cursor.fetchone()

            if result:
                id = result["id"]
            else:
                # If the record does not exist, insert it
                cursor.execute(insert_query, (recording, hash, time))
                db.commit()
                result = cursor.fetchone()
                if result:
                    id = result["id"]
                else:
                    raise Exception("Failed to insert new record into frames table")

            # Store the id in Redis
            redis.set(key, json.dumps(id))

    else:
        # If the id is in Redis, decode it
        id = json.loads(id)

    return id


def get_tracker_ids(db, redis, detection_classes, detections):
    select_query = """
    SELECT id FROM detections.trackers 
    WHERE class_id = %s AND ultralytics_id = %s
    """

    insert_query = """
    INSERT INTO detections.trackers (class_id, ultralytics_id) 
    VALUES (%s, %s) RETURNING id
    """

    tracker_ids = []
    for class_id, tracker_id in zip(detection_classes, detections.tracker_id):
        # Create a unique key for each class_id and tracker_id pair
        key = f"tracker:{class_id}:{tracker_id}"

        # Try to get the id from Redis
        id = redis.get(key)

        if id is None:
            # If the id is not in Redis, get it from the database
            with db.cursor() as cursor:
                cursor.execute(select_query, (class_id, int(tracker_id)))
                result = cursor.fetchone()

                if result:
                    id = result["id"]
                else:
                    # If the record does not exist, insert it
                    cursor.execute(insert_query, (class_id, int(tracker_id)))
                    db.commit()
                    result = cursor.fetchone()
                    if result:
                        id = result["id"]
                    else:
                        raise Exception(
                            "Failed to insert new record into trackers table"
                        )

                # Store the id in Redis
                redis.set(key, json.dumps(id))

        else:
            # If the id is in Redis, decode it
            id = json.loads(id)

        tracker_ids.append(id)

    return tracker_ids


# fmt: off
def detections(redis, db):
    while True:
        job = get_a_job(redis)
        print("Processing job: ", job)
        time = get_datetime_from_job(job)
        recording = create_new_recording(db, job, time)
        information = get_video_information(job)
        input = get_frame_generator(job)
        model, tracker = get_supervision_objects()
        tps, inverse_tps = get_tps()
        frame_duration = get_frame_duration(information)
        records_to_insert = []
        for frame in tqdm(input, total=information.total_frames):
            hash = hash_frame(frame)
            frame_id = get_frame_id(db, redis, recording, hash, time)
            results, detections = make_detections(model, tracker, frame)
            detection_classes = get_class_ids(db, redis, recording, results, detections)
            tracker_ids = get_tracker_ids(db, redis, detection_classes, detections)
            image_space_locations = get_image_space_locations(detections)
            map_space_locations = get_map_space_locations(tps, image_space_locations)
            records_to_insert.extend(process_detections(frame_id, detections, map_space_locations, tracker_ids))
            if len(records_to_insert) >= 10000:
                records_to_insert = do_bulk_insert(db, records_to_insert)
            time += frame_duration
        # process the tail
        records_to_insert = do_bulk_insert(db, records_to_insert)
# fmt: on


# ? the way this should work is that you run it in -d mode and it does detections
# ? and then you put those completions in a queue and run it in -c mode to purge out tracks/detections from void lists
# ? and pipe that into a queue which does the next thing.
# ? the idea is that decoding is super cheap and detecting isn't bad.
def main():
    db, redis = setup_service_handles()
    args = receive_arguments()
    if args.detections:
        detections(redis, db)


if __name__ == "__main__":
    main()
