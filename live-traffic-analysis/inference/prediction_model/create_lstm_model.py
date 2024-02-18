#!/usr/bin/env python3

import os
import psycopg2
import numpy as np
import psycopg2.extras
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
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
    coordinate_pairs.append(pairs)

# Convert to a NumPy array
tracks = np.array(coordinate_pairs, dtype=object)

print("Shape of tracks: ", tracks.shape)
# print("Sample of tracks:\n", tracks)

# next, we need to make the length of all tracks the same,
# so we can get away from this array of lists and get this type of shape:

# If you have a numpy array where:

# There are 10,000 tracks
# Each track has 30 data points
# Each data point has an X and Y coordinate
# Then the shape of that numpy array would be (10000, 30, 2).

# Here's why:

# 10000 represents the number of tracks.
# 30 represents the number of data points in each track.
# 2 represents the X and Y coordinates of each data point.
