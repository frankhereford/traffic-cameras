#!/usr/bin/env python3

import os
import torch
import random
import psycopg2
import numpy as np
import psycopg2.extras
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

SEGMENT_LENGTH = 30
PREDICTION_DISTANCE = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VehicleTrajectoryDataset(Dataset):
    def __init__(self, tracks):
        self.tracks = tracks

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track = self.tracks[idx]
        track_length = len(track)

        # If the track is longer than SEGMENT_LENGTH, randomly select a segment
        if track_length > SEGMENT_LENGTH + PREDICTION_DISTANCE:
            start = random.randint(
                0, track_length - SEGMENT_LENGTH - PREDICTION_DISTANCE
            )
            segment = track[start : start + SEGMENT_LENGTH]
        else:
            segment = track
        # Split the segment into input and target
        return torch.tensor(segment[:-1]), torch.tensor(segment[1:])


if __name__ == "__main__":

    load_dotenv()

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

    query = f"""
SELECT 
    detections.session_id, 
    detections.tracker_id, 
    ARRAY_AGG(ST_X(detections.location) ORDER BY detections.timestamp) as x_coords,
    ARRAY_AGG(ST_Y(detections.location) ORDER BY detections.timestamp) as y_coords,
    ARRAY_AGG(EXTRACT(EPOCH FROM detections.timestamp) ORDER BY detections.timestamp) as timestamps,
    paths.distance as track_length
FROM 
    detections_extended detections
    LEFT JOIN tracked_paths paths ON (detections.session_id = paths.session_id AND detections.tracker_id = paths.tracker_id)
WHERE 
    paths.distance IS NOT NULL
    AND paths.distance >= 15
GROUP BY 
    detections.session_id, detections.tracker_id, paths.distance
HAVING 
    COUNT(*) > ({SEGMENT_LENGTH} + {PREDICTION_DISTANCE})
ORDER BY 
    paths.distance DESC, detections.session_id DESC, detections.tracker_id;
"""

    cursor = db.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()

    normalized_tracks = []

    # Now you can work with the results
    for row in results:
        track = list(
            zip(row["x_coords"], row["y_coords"], [float(t) for t in row["timestamps"]])
        )
        # Extract coordinates and timestamps
        x_coords = np.array([point[0] for point in track]).reshape(-1, 1)
        y_coords = np.array([point[1] for point in track]).reshape(-1, 1)
        timestamps = np.array([point[2] for point in track]).reshape(-1, 1)

        # Initialize Min-Max Scaler
        min_max_scaler = MinMaxScaler()

        # Fit and transform
        x_coords_scaled = min_max_scaler.fit_transform(x_coords)
        y_coords_scaled = min_max_scaler.fit_transform(y_coords)
        timestamps_scaled = min_max_scaler.fit_transform(timestamps)

        # Combine back into normalized tracks
        normalized_track = list(
            zip(
                x_coords_scaled.flatten(),
                y_coords_scaled.flatten(),
                timestamps_scaled.flatten(),
            )
        )

        # print(normalized_track)
        normalized_tracks.append(normalized_track)

    # Initialize the dataset with the normalized tracks
    dataset = VehicleTrajectoryDataset(normalized_tracks)

    # DataLoader can now be used with this dataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Example: Iterate over the DataLoader
    for inputs, targets in dataloader:
        print("Input:", inputs)
        print("Target:", targets)

    print("data loader length", len(dataloader))
