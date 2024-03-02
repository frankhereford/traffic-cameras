#!/usr/bin/env python3

import os
import json
import torch
import joblib
import psycopg2
import argparse
import numpy as np
from tqdm import tqdm
import psycopg2.extras
import redis as redis_library
from dotenv import load_dotenv
from prediction_model.libraries.lstmvehicletracker import LSTMVehicleTracker


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
        description="Compute and store the predictions for the traffic data."
    )

    parser.add_argument('-p', '--populate_queue', action='store_true', default=False,
                        help='Populate the queue with data')
    parser.add_argument('-q', '--process_queue', action='store_true', default=False,
                        help='Process the data in the queue')

    args = parser.parse_args()
    return args

def load_min_max_scalar():
    min_max_scaler = joblib.load("./prediction_model/model_data/min_max_scaler.save")
    return min_max_scaler


def load_vehicle_tracker(hidden_size, num_layers):
    vehicle_tracker = LSTMVehicleTracker(
        input_size=2,
        hidden_size=hidden_size,
        num_layers=num_layers,
        seq_length=60,
        output_pairs=13,
    )
    return vehicle_tracker


def load_intersection_model(vehicle_tracker):
    vehicle_tracker.load_state_dict(
        torch.load("./prediction_model/model_data/lstm_model.pth")
    )
    return vehicle_tracker


def get_detections(db):
    cursor = db.cursor()
    query = """
        SELECT trackers.id AS tracker_id,
            detections.id AS detection_id,
            frames.id AS frame_id,
            frames.time as time,
            detections.location as location
        FROM detections.trackers
        JOIN detections.detections ON trackers.id = detections.tracker_id
        JOIN detections.frames ON detections.frame_id = frames.id
        ORDER BY trackers.id, frames.time asc
    """
        # LIMIT 1000000
    cursor.execute(query)
    detections = cursor.fetchall()
    return detections


def get_previous_detections(db, detection, min_length=60):
    cursor = db.cursor()
    tracker_id = detection["tracker_id"]
    time = detection["time"]
    query = f"""
            WITH ordered_detections AS (
                SELECT ARRAY[ST_X(detections.location), ST_Y(detections.location)] AS coordinates
                FROM detections.trackers
                JOIN detections.detections ON trackers.id = detections.tracker_id
                JOIN detections.frames ON detections.frame_id = frames.id
                WHERE trackers.id = %s
                AND frames.time <= %s
                ORDER BY frames.time
                LIMIT 60
            )
            SELECT ARRAY_AGG(coordinates) AS coordinates
            FROM ordered_detections
    """
    cursor.execute(query, (tracker_id, time))
    result = cursor.fetchone()
    if result is None or len(result['coordinates']) < min_length:
        return None
    return np.array(result['coordinates'])

def insert_into_redis_detection_queue(redis, detection):
    # Convert the RealDictRow to a dict
    detection_dict = dict(detection)

    # Convert the dict to a JSON string
    detection_json = json.dumps(detection_dict, default=str)

    # Insert the JSON string into the Redis queue
    redis.rpush('detection_queue', detection_json)
    
def clear_redis_queue(redis):
    redis.delete('detection_queue')
    
def get_detection_from_queue(redis):
    # Get the JSON string from the Redis queue
    detection_json = redis.lpop('detection_queue')

    # If the queue is empty, lpop will return None
    if detection_json is None:
        return None

    # Convert the JSON string back to a dict
    detection_dict = json.loads(detection_json)

    return detection_dict

def truncate_and_populate_queue(db, redis):
    clear_redis_queue(redis)
    cursor = db.cursor()
    query = """
        SELECT trackers.id AS tracker_id,
            detections.id AS detection_id,
            frames.id AS frame_id,
            frames.time as time,
            detections.location as location
        FROM detections.trackers
        JOIN detections.detections ON trackers.id = detections.tracker_id
        JOIN detections.frames ON detections.frame_id = frames.id
        ORDER BY trackers.id, frames.time asc
    """
        # LIMIT 1000000
    cursor.execute(query)
    print("starting iteration")
    while True:
        detection = cursor.fetchone()
        if detection is None:
            break
        insert_into_redis_detection_queue(redis, detection)

def infer_future_locaftion(min_max_scalar, input):
    print("input", input.shape)
    # input_2d = input.reshape(-1, input.shape[-1])
    # history_scaled_2d = min_max_scalar.transform(input_2d)
    # history_scaled = history_scaled_2d.reshape(1, 60, 2)
    # history_scalaed_tensor = torch.from_numpy(history_scaled).float()


def process_queue(redis, db):
    min_max_scalar = load_min_max_scalar()
    vehicle_tracker = load_vehicle_tracker(hidden_size=256, num_layers=5)
    intersection_model = load_intersection_model(vehicle_tracker)
    while True:
        detection = get_detection_from_queue(redis)
        if detection is None:
            break
        # print("detection", detection)
        previous_detections = get_previous_detections(db=db, detection=detection, min_length=60)
        if previous_detections is None:
            continue
        # print("previous_detections", previous_detections)
        future = infer_future_locaftion(min_max_scalar, previous_detections)
        print(".")
        quit()


def main():
    db, redis = setup_service_handles()
    args = receive_arguments()
    if (args.populate_queue):
        truncate_and_populate_queue(db, redis)
    if (args.process_queue):
        process_queue(redis, db)

        # pass


if __name__ == "__main__":
    main()
