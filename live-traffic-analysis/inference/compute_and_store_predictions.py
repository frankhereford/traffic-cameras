#!/usr/bin/env python3

import os
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
        seq_length=30,
    )
    return vehicle_tracker


def load_intersection_model(vehicle_tracker):
    vehicle_tracker.load_state_dict(
        torch.load("./prediction_model/model_data/lstm_model.pth")
    )
    return vehicle_tracker


def main():
    db, redis = setup_service_handles()
    min_max_scalar = load_min_max_scalar
    args = receive_arguments()
    min_max_scalar = load_min_max_scalar()
    vehicle_tracker = load_vehicle_tracker(hidden_size=256, num_layers=2)
    intersection_model = load_intersection_model(vehicle_tracker)


if __name__ == "__main__":
    main()
