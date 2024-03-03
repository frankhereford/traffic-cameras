#!/usr/bin/env python3

import os
import json
import pytz
import uuid
import copy
import torch
import random
import hashlib
import argparse
import psycopg2
import numpy as np
from tqdm import tqdm
import psycopg2.extras
import supervision as sv
from ultralytics import YOLO
import redis as redis_library
from dotenv import load_dotenv
from torch_tps import ThinPlateSpline
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime, timedelta, timezone
from utilities.transformation import read_points_file
from utilities.database_results_detections import (
    Results,
    from_database,
)


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

    parser.add_argument(
        "-r",
        "--render",
        action="store_true",
        default=False,
        help="Flag to control if the output video will be rendered",
    )

    args = parser.parse_args()
    return args


def get_a_job(redis, queue_name):
    _, value = redis.brpop(queue_name)
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


def recall_detections(db, tracker, hash):
    result = Results()
    sv.Detections.from_database = from_database
    # sv.ByteTrack.update_with_tensors = update_with_tensors
    # sv.ByteTrack.update_with_detections = update_with_detections
    detections = sv.Detections.from_database(db, hash)
    # print("detections before: ", detections.speed)
    detections = tracker.update_with_detections(detections)
    # print("detections after: ", detections.speed)
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


def get_recording_id(db, redis, filename, start_time):
    select_query = """
    SELECT id FROM detections.recordings 
    WHERE filename = %s AND start_time = %s
    """

    insert_query = """
    INSERT INTO detections.recordings (filename, start_time) 
    VALUES (%s, %s) RETURNING id
    """

    # Create a unique key for the recording
    key = f"recording:{filename}:{start_time}"

    # Try to get the id from Redis
    recording_id = redis.get(key)

    if recording_id is None:
        # If the id is not in Redis, get it from the database
        with db.cursor() as cursor:
            cursor.execute(select_query, (filename, start_time))
            result = cursor.fetchone()

            if result:
                recording_id = result["id"]
            else:
                # If the record does not exist, insert it
                cursor.execute(insert_query, (filename, start_time))
                db.commit()
                result = cursor.fetchone()
                if result:
                    recording_id = result["id"]
                else:
                    raise Exception("Failed to insert new record into recordings table")

            # Store the id in Redis
            redis.set(key, json.dumps(recording_id))
    else:
        # If the id is in Redis, decode it
        recording_id = json.loads(recording_id)

    return recording_id


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
        key = f"class:{recording}:{class_id}:{class_name}"

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


def get_output_path(job):
    # Get the filename without the extension
    base_name = os.path.splitext(job)[0]

    # Get the current date and time
    now = datetime.now()

    # Format the date and time as YYMMDD-HHMMSS
    date_time_str = now.strftime("%y%m%d-%H%M%S")

    # Create the output file name
    output_file_name = (
        "/home/frank/development/traffic-cameras/live-traffic-analysis/inference/output_media/full_fps_output/"
        + f"{base_name}_{date_time_str}.mp4"
    )

    print(f"Output file name: {output_file_name}")
    return output_file_name


def get_colors():
    return sv.ColorPalette.DEFAULT


def get_annotators(color):
    box = sv.BoxCornerAnnotator(color=color)
    trace = sv.TraceAnnotator(
        thickness=2, trace_length=30, position=sv.Position.BOTTOM_CENTER
    )
    classs = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=2,
        text_position=sv.Position.TOP_CENTER,
        text_color=sv.Color(r=0, g=0, b=0),
    )
    speed = sv.LabelAnnotator(
        text_scale=0.5,
        text_thickness=1,
        text_padding=2,
        text_position=sv.Position.BOTTOM_CENTER,
        text_color=sv.Color(r=0, g=0, b=0),
    )
    smooth = sv.DetectionsSmoother(length=3)

    return box, trace, classs, speed, smooth


def get_image_space_centers(detections):
    points = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
    # tensor = torch.tensor(points).float()
    # tensor = tensor.to("cuda")
    return points


def get_top_labels(class_names, centers):
    labels = []
    for class_name, center in zip(class_names, centers):
        # print(center)
        # labels.append(f"{class_name} {center}")
        labels.append(f"{class_name}: (x: {int(center[0])}, y: {int(center[1])})")
    return labels


def burn_in_timestamp(frame, time):
    # Convert the frame to a PIL Image
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)

    # Specify that the incoming time is in UTC
    time = time.replace(tzinfo=timezone.utc)

    # Convert the datetime object from UTC to Central Time
    central = pytz.timezone("US/Central")
    time_central = time.astimezone(central)

    datetime_str = time_central.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # Load the font (you may need to adjust the path)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=18
    )

    # Draw the datetime string onto the image
    draw.text((10, 10), datetime_str, fill="white", font=font)

    # Convert the image back to a numpy array
    frame_with_timestamp = np.array(image)

    return frame_with_timestamp


def get_speed_labels(detections):
    if detections.speed is None:
        return None
    formatted_rates = [
        f"{int(rate)} mph" if not np.isnan(rate) else None for rate in detections.speed
    ]

    # Convert the list to a numpy array
    speed_labels = np.array(formatted_rates, dtype=str)

    return speed_labels


def get_random_job(db):
    # Create a new cursor object with DictCursor
    cursor = db.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Execute a query to get all ids from the recordings table
    cursor.execute("SELECT id FROM detections.recordings;")

    # Fetch all the results
    ids = cursor.fetchall()

    # Close the cursor
    cursor.close()

    # If there are no ids, return None
    if not ids:
        return None

    # Choose a random id
    random_id = random.choice(ids)[0]

    # Create a new cursor object with DictCursor
    cursor = db.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Execute a query to get the recording with the random id
    cursor.execute("SELECT * FROM detections.recordings WHERE id = %s;", (random_id,))

    # Fetch the result
    recording = cursor.fetchone()

    # Close the cursor
    cursor.close()

    # Return the random recording
    return recording.get("filename")


def draw_futures(db, frame, detections):
    if detections is None or detections.detection_id is None:
        return frame

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    radius = 5

    for detection in detections.detection_id:
        detection_int = int(detection)
        # print(f"detection_int: {detection_int}")
        sql = """
        select pixels
        from detections.predictions
        where detection_id = %s
        """
        cursor = db.cursor()
        cursor.execute(sql, (detection_int,))
        pixels = cursor.fetchone()
        # print(f"pixels: {pixels}")
        if pixels is not None:
            previous_loc = None
            for loc in pixels["pixels"]:
                x, y = map(int, loc)
                # print(f"x: {x}, y: {y}")
                upper_left = (x - radius, y - radius)
                lower_right = (x + radius, y + radius)

                draw.ellipse([upper_left, lower_right], fill="red")

                if previous_loc is not None:
                    draw.line([previous_loc, (x, y)], fill="red", width=3)
                previous_loc = (x, y)

    frame = np.array(image)
    return frame


# fmt: off
def detections(redis, db):
    while True:
        job = None
        job = get_a_job(redis, 'downloaded-videos-queue')
        print("Processing job: ", job)
        time = get_datetime_from_job(job)
        recording = get_recording_id(db, redis, job, time)
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

def render(redis, db):
    # while True:
    job = None
    # job = get_random_job(db)
    # job = get_a_job(redis, 'render-videos-queue')
    job = "ByED80IKdIU-20240220-124256.mp4"
    print("Processing job: ", job)
    time = get_datetime_from_job(job)
    information = get_video_information(job)
    input = get_frame_generator(job)
    _, tracker = get_supervision_objects()
    frame_duration = get_frame_duration(information)
    output_path = get_output_path(job)
    colors = get_colors()
    box, trace, classs, speed, smooth = get_annotators(colors)
    with sv.VideoSink(output_path, information) as sink:
        frame_count = 0
        for frame in tqdm(input, total=information.total_frames):
            if frame_count == 1024:
                pass
                # break
            hash = hash_frame(frame)
            results, detections = recall_detections(db, tracker, hash)
            # print(f"detections: {detections}")
            class_names = get_class_names(detections.class_id, results)
            centers = get_image_space_centers(detections)
            labels = get_top_labels(class_names, centers)
            speed_labels = get_speed_labels(detections)
            frame = box.annotate(frame, detections)
            frame = trace.annotate(frame, detections)
            frame = classs.annotate(frame, detections, labels)
            frame = speed.annotate(frame, detections, speed_labels)
            frame = burn_in_timestamp(frame, time)
            frame = draw_futures(db, frame, detections)
            sink.write_frame(frame=frame)
            time += frame_duration
            frame_count += 1



def main():
    db, redis = setup_service_handles()
    args = receive_arguments()
    if args.detections:
        pass
        # detections(redis, db)
    if args.render:
        render(redis, db)


if __name__ == "__main__":
    main()
