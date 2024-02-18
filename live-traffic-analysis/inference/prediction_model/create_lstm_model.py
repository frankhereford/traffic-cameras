#!/usr/bin/env python3

import os
import psycopg2
import numpy as np
import psycopg2.extras
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from libraries.parameters import SEGMENT_LENGTH, PREDICTION_DISTANCE
from libraries.parameters import INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE


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

    print("Executing query to get all tracks from the database")
    query = f"""
        SELECT 
            detections.session_id, 
            detections.tracker_id, 
            ARRAY_AGG(ST_X(detections.location) ORDER BY detections.timestamp) as x_coords,
            ARRAY_AGG(ST_Y(detections.location) ORDER BY detections.timestamp) as y_coords,
            ARRAY_AGG(EXTRACT(EPOCH FROM detections.timestamp) ORDER BY detections.timestamp) as timestamps,
            MIN(detections.timestamp) as start_timestamp,
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
            COUNT(*) > ({SEGMENT_LENGTH} + {PREDICTION_DISTANCE} + 5)
        ORDER BY 
            MIN(detections.timestamp) asc
        """

    cursor = db.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()

    print("Total number of tracks retrieved: ", len(results))

    # treating X and Y coordinates as individual series, not pairs
    if False:
        x_coords_all = []
        y_coords_all = []
        timestamps_all = []

        print("Concatenating all data into numpy arrays")
        for row in results:
            # Convert data to np.float64 immediately after fetching
            x_coords_all.extend([np.float64(x) for x in row["x_coords"]])
            y_coords_all.extend([np.float64(y) for y in row["y_coords"]])
            timestamps_all.extend([np.float64(t) for t in row["timestamps"]])

        print("Length of x_coords_all: ", len(x_coords_all))
        print("Length of y_coords_all: ", len(y_coords_all))
        print("Length of timestamps_all: ", len(timestamps_all))

        print("Reshaping the all data numpy arrays")
        x_coords_all = np.array(x_coords_all).reshape(-1, 1)
        y_coords_all = np.array(y_coords_all).reshape(-1, 1)
        timestamps_all = np.array(timestamps_all).reshape(-1, 1)

        print("Shape of the x_coords_all: ", x_coords_all.shape)
        print("Shape of the y_coords_all: ", y_coords_all.shape)
        print("Shape of the timestamps_all: ", timestamps_all.shape)


coordinates_all = []

print("Concatenating all data into numpy arrays")
for row in results:
    # Convert data to np.float64 immediately after fetching
    coordinates_all.extend(
        [
            (np.float64(x), np.float64(y))
            for x, y in zip(row["x_coords"], row["y_coords"])
        ]
    )

print("Length of coordinates_all: ", len(coordinates_all))

print("Reshaping the all data numpy arrays")
coordinates_all = np.array(coordinates_all).reshape(-1, 2)

print("Shape of the coordinates_all: ", coordinates_all.shape)

print("Sample of coordinates all:\n", coordinates_all)

min_max_scaler = MinMaxScaler()

print("Fitting the Min-Max Scaler")
normalized_coordinates = min_max_scaler.fit_transform(coordinates_all)

print("Sample of normalized coordinates:\n", normalized_coordinates)

print("MinMaxScalar for dataset is defined")

# inspect the normalized data to make sure it fits the way we expect
if False:
    # Initialize min and max variables
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    # Assuming normalized_coordinates is a list of tuples (x, y)
    for i in range(0, len(normalized_coordinates), 30):
        batch = normalized_coordinates[i : i + 30]
        # print(batch)

        # Update min and max variables
        for x, y in batch:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        print(f"Min X: {min_x}, Max X: {max_x}, Min Y: {min_y}, Max Y: {max_y}")

        # input("Press Enter to continue...")

coordinate_pairs = []
for record in results:
    x_coords = record["x_coords"]
    y_coords = record["y_coords"]
    pairs = list(zip(x_coords, y_coords))

    for i in range(0, len(pairs), 60):
        batch = pairs[i : i + 60]
        if len(batch) == 60:
            coordinate_pairs.append(batch)

tracks = np.array([np.array(track, dtype=np.float64) for track in coordinate_pairs])

print("Shape of tracks: ", tracks.shape)
# print("Sample of tracks:\n", tracks)

# Save the original shape
original_shape = tracks.shape

# Reshape the tracks array into a 2D array
tracks_2d = tracks.reshape(-1, tracks.shape[-1])

# Use the MinMaxScaler to transform the 2D tracks array
tracks_scaled_2d = min_max_scaler.transform(tracks_2d)

# Reshape the scaled array back to its original shape
tracks_scaled = tracks_scaled_2d.reshape(original_shape)

print("Shape of tracks_scaled: ", tracks_scaled.shape)
# print("Sample of tracks_scaled:\n", tracks_scaled)


# demonstrate the inverse transform
if False:
    # Reshape the scaled tracks array into a 2D array
    tracks_scaled_2d = tracks_scaled.reshape(-1, tracks_scaled.shape[-1])

    # Use the MinMaxScaler to inverse transform the 2D tracks array
    tracks_inverse_2d = min_max_scaler.inverse_transform(tracks_scaled_2d)

    # Reshape the inverse transformed array back to its original shape
    tracks_inverse = tracks_inverse_2d.reshape(original_shape)

    print("Shape of tracks_inverse: ", tracks_inverse.shape)
    print("Sample of tracks_inverse:\n", tracks_inverse)

# print("Creating the one-second tracks dataset")
# tracks_small = tracks_scaled[:, :30, :]

# print("Shape of tracks_small: ", tracks_small.shape)
# # print("Sample of tracks_small:\n", tracks_small)


training_tracks, testing_tracks = train_test_split(
    tracks_scaled, test_size=0.1, random_state=42
)

print("Shape of training_tracks: ", training_tracks.shape)
print("Shape of testing_tracks: ", testing_tracks.shape)
