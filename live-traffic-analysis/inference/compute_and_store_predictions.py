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
from torch_tps import ThinPlateSpline
from utilities.transformation import read_points_file
from prediction_model.libraries.lstmvehicletracker import LSTMVehicleTracker

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(suppress=True, precision=8)  # Adjust precision as needed


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
    vehicle_tracker_on_device = vehicle_tracker.to(DEVICE)
    return vehicle_tracker_on_device


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
                SELECT ARRAY[ST_X(detections.location), ST_Y(detections.location)] AS coordinates,
                       ROW_NUMBER() OVER (ORDER BY frames.time DESC) AS row_num
                FROM detections.trackers
                JOIN detections.detections ON trackers.id = detections.tracker_id
                JOIN detections.frames ON detections.frame_id = frames.id
                WHERE trackers.id = %s
                AND frames.time <= %s
                ORDER BY frames.time DESC
                LIMIT 60
            )
            SELECT ARRAY_AGG(coordinates ORDER BY row_num ASC) AS coordinates
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

def infer_future_location(min_max_scalar, intersection_model, input):
    # print("input", input.shape)
    input_2d = input.reshape(-1, input.shape[-1])
    history_scaled_2d = min_max_scalar.transform(input_2d)
    history_scaled = history_scaled_2d.reshape(1, 60, 2)
    history_scaled_tensor = torch.from_numpy(history_scaled).float()
    on_device_history_scaled_tensor = history_scaled_tensor.to(DEVICE)
    prediction = None
    with torch.no_grad():
        intersection_model.eval()
        prediction = intersection_model(on_device_history_scaled_tensor)
        # print("prediction", prediction)

    input_array = (
        prediction.cpu().numpy().reshape(-1, 2)
    )

    input_inverted = min_max_scalar.inverse_transform(input_array)

    # print("input_inverted", input_inverted)
    return input_inverted

def get_tps():
    coordinates = read_points_file("./gcp/coldwater_mi.points")
    tps = ThinPlateSpline(0.5)
    tps.fit(coordinates["image_coordinates"], coordinates["map_coordinates"])
    reverse_tps = ThinPlateSpline(0.5)
    reverse_tps.fit(coordinates["map_coordinates"], coordinates["image_coordinates"])
    return tps, reverse_tps

def inverse_transform_prediction(inverse_tps, prediction):
    predictions_tensor = torch.tensor(prediction)
    predictions_tensor_cude = predictions_tensor.to(DEVICE)
    predictions_in_image_space = inverse_tps.transform(predictions_tensor_cude).cpu().numpy()
    predictions_in_image_space_int = predictions_in_image_space.astype(int)
    return predictions_in_image_space_int

# def save_future_to_db(db, detection, future, image_space_future):
#     print(detection["tracker_id"])
#     print(detection["detection_id"])
#     print(future)
#     print(image_space_future)
#     quit()
#     pass

def save_future_to_db(db, detection, future, image_space_future):
    cursor = db.cursor()
    tracker_id = detection["tracker_id"]
    detection_id = detection["detection_id"]
    locations = [f"ST_MakePoint({x}, {y})" for x, y in future.tolist()]
    pixels = image_space_future.tolist()

    gis_data = "ARRAY[" + ", ".join(locations) + "]"
    image_data = "ARRAY" + (str(pixels))

    query = f"""
        INSERT INTO detections.predictions (tracker_id, detection_id, locations, pixels)
        VALUES (%s, %s, {gis_data}, {image_data} )
        RETURNING id
    """
    cursor.execute(query, (tracker_id, detection_id))
    inserted_id = cursor.fetchone()['id']
    db.commit()
    return inserted_id

def process_queue(redis, db):
    tps, inverse_tps = get_tps()
    min_max_scalar = load_min_max_scalar()
    vehicle_tracker = load_vehicle_tracker(hidden_size=256, num_layers=5)
    intersection_model = load_intersection_model(vehicle_tracker)
    while True:
        detection = get_detection_from_queue(redis)
        if detection is None:
            break
        previous_detections = get_previous_detections(db=db, detection=detection, min_length=60)
        if previous_detections is None:
            continue
        future = infer_future_location(min_max_scalar, intersection_model, previous_detections)
        image_space_future = inverse_transform_prediction(inverse_tps, future)
        id = save_future_to_db(db, detection, future, image_space_future)
        print(f"inserted id: {id}", flush=True)


def main():
    db, redis = setup_service_handles()
    args = receive_arguments()
    if (args.populate_queue):
        truncate_and_populate_queue(db, redis)
    if (args.process_queue):
        process_queue(redis, db)


if __name__ == "__main__":
    main()
