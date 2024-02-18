#!/usr/bin/env python3


import os
import time
import torch
import random
import psycopg2
import numpy as np
import torch.nn as nn
import psycopg2.extras
import torch.optim as optim
from dotenv import load_dotenv
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from libraries.parameters import SEGMENT_LENGTH, PREDICTION_DISTANCE

# from libraries.parameters import INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE
num_epochs = 200
hidden_size = 512
num_layers = 2
learning_rate = 0.0001
batch_size = 64
verification_loops = 128
epoch_print_interval = 25


class LSTMVehicleTracker(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, seq_length=30):
        super(LSTMVehicleTracker, self).__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 2)  # Output is 2D (X,Y)

    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(
            device
        )
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(
            device
        )

        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn_last_layer = hn[-1, :, :]  # Get the hidden state of the last layer
        out = self.fc(hn_last_layer)
        return out


if __name__ == "__main__":
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # this query is really just an experiment; it filters down tracks to ones with significant motion
    query = f"""
    WITH ordered_detections AS (
        SELECT 
            session_id, 
            tracker_id, 
            ST_X(location) as x_coord,
            ST_Y(location) as y_coord,
            ROW_NUMBER() OVER(PARTITION BY session_id, tracker_id ORDER BY timestamp) as rn
        FROM 
            detections_extended
    ),
    distances AS (
        SELECT 
            d1.session_id, 
            d1.tracker_id, 
            SQRT(POWER(d1.x_coord - d2.x_coord, 2) + POWER(d1.y_coord - d2.y_coord, 2)) as distance
        FROM 
            ordered_detections d1
            JOIN ordered_detections d2 ON d1.session_id = d2.session_id AND d1.tracker_id = d2.tracker_id
        WHERE 
            d1.rn = 1 AND d2.rn = 30
    )
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
        JOIN distances ON (detections.session_id = distances.session_id AND detections.tracker_id = distances.tracker_id)
    WHERE 
        paths.distance IS NOT NULL
        AND paths.distance >= 30
        AND distances.distance > 30
    GROUP BY 
        detections.session_id, detections.tracker_id, paths.distance
    HAVING 
        COUNT(*) > 60
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

    # print("Concatenating all data into numpy arrays")
    for row in results:
        # Convert data to np.float64 immediately after fetching
        coordinates_all.extend(
            [
                (np.float64(x), np.float64(y))
                for x, y in zip(row["x_coords"], row["y_coords"])
            ]
        )

    # print("Length of coordinates_all: ", len(coordinates_all))

    # print("Reshaping the all data numpy arrays")
    coordinates_all = np.array(coordinates_all).reshape(-1, 2)

    # print("Shape of the coordinates_all: ", coordinates_all.shape)

    # print("Sample of coordinates all:\n", coordinates_all)

    min_max_scaler = MinMaxScaler()

    # print("Fitting the Min-Max Scaler")
    normalized_coordinates = min_max_scaler.fit_transform(coordinates_all)

    # print("Sample of normalized coordinates:\n", normalized_coordinates)

    # print("MinMaxScalar for dataset is defined")

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

    # print("Shape of tracks: ", tracks.shape)
    # print("Sample of tracks:\n", tracks)

    # Save the original shape
    original_shape = tracks.shape

    # Reshape the tracks array into a 2D array
    tracks_2d = tracks.reshape(-1, tracks.shape[-1])

    # Use the MinMaxScaler to transform the 2D tracks array
    tracks_scaled_2d = min_max_scaler.transform(tracks_2d)

    # Reshape the scaled array back to its original shape
    tracks_scaled = tracks_scaled_2d.reshape(original_shape)

    # print("Shape of tracks_scaled: ", tracks_scaled.shape)
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

    training_tracks, testing_tracks = train_test_split(
        tracks_scaled, test_size=0.1, random_state=42
    )

    # print("Shape of training_tracks: ", training_tracks.shape)
    # print("Shape of testing_tracks: ", testing_tracks.shape)

    # print("Creating the one-second tracks dataset")
    input_training_tracks = training_tracks[:, :30, :]
    output_training_tracks = training_tracks[:, 30, :]

    input_testing_tracks = testing_tracks[:, :30, :]
    output_testing_tracks = testing_tracks[:, 30, :]

    # print("Shape of input_training_tracks: ", input_training_tracks.shape)
    # print("Shape of output_training_tracks: ", output_training_tracks.shape)
    # print("Shape of input_testing_tracks: ", input_testing_tracks.shape)
    # print("Shape of output_testing_tracks: ", output_testing_tracks.shape)

    # print("Making tensors out of these numpy arrays..")

    input_training_tensor = Variable(torch.Tensor(input_training_tracks)).to(device)
    output_training_tensor = Variable(torch.Tensor(output_training_tracks)).to(device)

    input_testing_tensor = Variable(torch.Tensor(input_testing_tracks)).to(device)
    output_testing_tensor = Variable(torch.Tensor(output_testing_tracks)).to(device)

    # print("Shape of input_training_tensor: ", input_training_tensor.shape)
    # print("Shape of output_training_tensor: ", output_training_tensor.shape)
    # print("Shape of input_testing_tensor: ", input_testing_tensor.shape)
    # print("Shape of output_testing_tensor: ", output_testing_tensor.shape)

    # red herring reshaping?
    if False:
        print("Reshaping the tensors..")

        # Reshape the tensors
        input_training_tensor = input_training_tensor.transpose(0, 1)
        output_training_tensor = output_training_tensor.unsqueeze(0)

        input_testing_tensor = input_testing_tensor.transpose(0, 1)
        output_testing_tensor = output_testing_tensor.unsqueeze(0)

        print("Shape of input_training_tensor: ", input_training_tensor.shape)
        print("Shape of output_training_tensor: ", output_training_tensor.shape)
        print("Shape of input_testing_tensor: ", input_testing_tensor.shape)
        print("Shape of output_testing_tensor: ", output_testing_tensor.shape)

    # Create a TensorDataset
    train_dataset = TensorDataset(input_training_tensor, output_training_tensor)
    test_dataset = TensorDataset(input_testing_tensor, output_testing_tensor)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    for batch in train_loader:
        input_batch, output_batch = batch
        # print("Input batch shape:", input_batch.shape)
        # print("Output batch shape:", output_batch.shape)
        # print("First input batch:\n", input_batch)
        # print("First output batch:\n", output_batch)
        break

    vehicle_tracker = LSTMVehicleTracker(
        input_size=2, hidden_size=hidden_size, num_layers=num_layers, seq_length=30
    )
    vehicle_tracker = vehicle_tracker.to(device)
    print(vehicle_tracker)

    # Loss and optimizer
    criterion = torch.nn.MSELoss()  # Mean Squared Error for regression task
    optimizer = optim.Adam(vehicle_tracker.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        vehicle_tracker.train()

        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = vehicle_tracker(inputs)
            loss = criterion(outputs, targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % epoch_print_interval == 0:
            # if True:
            print(
                f"Epoch [{epoch}/{num_epochs}], Loss: {running_loss / len(train_loader):.8f}"
            )

        # Validation step, if you have validation data
        vehicle_tracker.eval()
        with torch.no_grad():
            val_outputs = vehicle_tracker(input_testing_tensor)
            val_loss = criterion(val_outputs, output_testing_tensor)
            if epoch % epoch_print_interval == 0:
                # if True:
                print(f"Validation Loss: {val_loss.item():.8f}")

            # print("Try it out!")

    cursor = db.cursor()
    cursor.execute("truncate prediction cascade;")
    db.commit()
    cursor.close()

    counter = 0
    total_distance = 0

    # Loop 1024 times
    for _ in range(verification_loops):
        # while True:
        random_track = random.choice(results)
        coordinate_pairs = []
        x_coords = random_track["x_coords"]
        y_coords = random_track["y_coords"]
        pairs = list(zip(x_coords, y_coords))

        for i in range(0, len(pairs), 60):
            batch = pairs[i : i + 60]
            if len(batch) == 60:
                coordinate_pairs.append(batch)
                break  # even if we could make a couple, let's just do one
        # print("Coordinate pairs: ", coordinate_pairs)
        tracks = np.array(
            [np.array(track, dtype=np.float64) for track in coordinate_pairs]
        )

        # Save the original shape
        original_shape = tracks.shape

        # Reshape the tracks array into a 2D array
        tracks_2d = tracks.reshape(-1, tracks.shape[-1])

        # Use the MinMaxScaler to transform the 2D tracks array
        tracks_scaled_2d = min_max_scaler.transform(tracks_2d)

        # Reshape the scaled array back to its original shape
        tracks_scaled = tracks_scaled_2d.reshape(original_shape)

        # print("tracks_scaled shape: ", tracks_scaled.shape)
        # print("Tracks: ", tracks)

        input_tracks = tracks_scaled[:, :30, :]
        output_tracks = tracks_scaled[:, 30, :]

        input_tensor = Variable(torch.Tensor(input_tracks)).to(device)
        output_tensor = Variable(torch.Tensor(output_tracks)).to(device)

        # print("Shape of input_tensor: ", input_tensor.shape)
        # print("Shape of output_tensor: ", output_tensor.shape)

        with torch.no_grad():
            vehicle_tracker.eval()
            prediction = vehicle_tracker(input_tensor)

        # print("Prediction shape: ", prediction.shape)
        # print("Prediction: ", prediction)
        # print("Truth: ", output_tensor)

        input_array = input_tensor.cpu().numpy().reshape(-1, 2)  # Reshape to (30, 2)
        output_array = output_tensor.cpu().numpy().reshape(-1, 2)  # Reshape to (1, 2)
        prediction_array = prediction.cpu().numpy().reshape(-1, 2)  # Reshape to (1, 2)

        input_inverted = min_max_scaler.inverse_transform(input_array)
        output_inverted = min_max_scaler.inverse_transform(output_array)
        prediction_inverted = min_max_scaler.inverse_transform(prediction_array)

        # print("Shape of input_inverted: ", input_inverted.shape)
        # print("Sample of input_inverted:\n", input_inverted)

        # print("Shape of output_inverted: ", output_inverted.shape)
        # print("Sample of output_inverted:\n", output_inverted)

        # print("Shape of prediction_inverted: ", prediction_inverted.shape)
        # print("Sample of prediction_inverted:\n", prediction_inverted)

        distance = np.sqrt(np.sum((output_inverted - prediction_inverted) ** 2))
        # print("Distance: ", distance)

        # record track and prediction in DB
        if True:
            # Create a new prediction record
            cursor = db.cursor()
            # cursor.execute("truncate prediction cascade;")
            # db.commit()

            # Create a new prediction record
            cursor = db.cursor()
            cursor.execute("INSERT INTO prediction DEFAULT VALUES RETURNING id;")
            result = cursor.fetchone()
            prediction_id = result["id"]
            db.commit()

            # Insert known points
            for i, (x, y) in enumerate(input_inverted):
                cursor.execute(
                    """
                    INSERT INTO known_points (prediction, sequence_number, location)
                    VALUES (%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 2253));
                    """,
                    (prediction_id, i, float(x), float(y)),
                )
            db.commit()

            # Insert predicted points
            for i, (x, y) in enumerate(prediction_inverted):
                cursor.execute(
                    """
                    INSERT INTO predicted_points (prediction, sequence_number, location)
                    VALUES (%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 2253));
                    """,
                    (prediction_id, i, float(x), float(y)),
                )
            db.commit()

            cursor.close()

        # input("try again?")

        total_distance += distance

        # Increment the counter
        counter += 1

    average_distance = total_distance / counter
    print(f"Average distance: {average_distance:.2f}")

    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    # Calculate the hours
    hours, remainder = divmod(elapsed_time, 3600)

    # Calculate the minutes and seconds
    minutes, seconds = divmod(remainder, 60)

    print(
        "Part 1 took",
        int(hours),
        "hours,",
        int(minutes),
        "minutes, and",
        int(seconds),
        "seconds",
    )
