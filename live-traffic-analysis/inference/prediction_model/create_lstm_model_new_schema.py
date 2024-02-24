#!/usr/bin/env python3

import os
import time
import math
import torch
import random
import joblib
import psycopg2
import numpy as np
import torch.nn as nn
import psycopg2.extras
from psycopg2 import sql
import torch.optim as optim
from dotenv import load_dotenv
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from libraries.parameters import SEGMENT_LENGTH, PREDICTION_DISTANCE
from libraries.lstmvehicletracker import LSTMVehicleTracker

num_epochs = 100
epoch_print_interval = 10
hidden_size = 256
num_layers = 2
learning_rate = 0.0001
batch_size = 64
verification_loops = 64

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

    cursor = db.cursor()

    cursor.execute(
        "UPDATE training_data.sample_data SET is_training = false, is_testing = false"
    )
    db.commit()

    cursor.execute("SELECT * FROM training_data.sample_data")
    rows = cursor.fetchall()

    random.shuffle(rows)

    num_training = math.ceil(len(rows) * 0.9)
    num_testing = len(rows) - num_training

    cursor.execute(
        sql.SQL(
            "UPDATE training_data.sample_data SET is_training = true WHERE sample_id IN %s"
        ),
        (tuple(row["sample_id"] for row in rows[:num_training]),),
    )

    cursor.execute(
        sql.SQL(
            "UPDATE training_data.sample_data SET is_testing = true WHERE sample_id IN %s"
        ),
        (tuple(row["sample_id"] for row in rows[num_training:]),),
    )
    db.commit()

    query = "select * FROM training_data.sample_data"

    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()

    print("Total number of tracks retrieved: ", len(results))

    coordinates_all = []
    for row in results:
        coordinates_all.extend(
            [(np.float64(x), np.float64(y)) for x, y in row["coordinates"]]
        )

    coordinates_all = np.array(coordinates_all).reshape(-1, 2)
    min_max_scaler = MinMaxScaler()
    normalized_coordinates = min_max_scaler.fit_transform(coordinates_all)
    joblib.dump(min_max_scaler, "./model_data/min_max_scaler.save")

    cursor = db.cursor()
    query = "select * FROM training_data.sample_data where is_training is true"
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()

    coordinate_pairs = []
    for record in results:
        coordinate_pairs.extend(
            [(np.float64(x), np.float64(y)) for x, y in record["coordinates"]]
        )

    coordinate_pairs = np.array(coordinate_pairs).reshape(-1, 60, 2)
    original_shape = coordinate_pairs.shape
    coordinate_pairs_2d = coordinate_pairs.reshape(-1, coordinate_pairs.shape[-1])
    coordinate_pairs_scaled_2d = min_max_scaler.transform(coordinate_pairs_2d)
    training_tracks = coordinate_pairs_scaled_2d.reshape(original_shape)
    cursor.close()

    cursor = db.cursor()
    testing_query = "select * FROM training_data.sample_data where is_testing is true"
    cursor.execute(testing_query)
    testing_results = cursor.fetchall()
    cursor.close()

    testing_coordinate_pairs = []
    for record in testing_results:
        testing_coordinate_pairs.extend(
            [(np.float64(x), np.float64(y)) for x, y in record["coordinates"]]
        )

    testing_coordinate_pairs = np.array(testing_coordinate_pairs).reshape(-1, 60, 2)
    original_shape = testing_coordinate_pairs.shape
    testing_coordinate_pairs_2d = testing_coordinate_pairs.reshape(
        -1, testing_coordinate_pairs.shape[-1]
    )
    testing_coordinate_pairs_scaled_2d = min_max_scaler.transform(
        testing_coordinate_pairs_2d
    )
    testing_tracks = testing_coordinate_pairs_scaled_2d.reshape(original_shape)
    cursor.close()

    print("Shape of training_tracks: ", training_tracks.shape)
    print("Shape of testing_tracks: ", testing_tracks.shape)

    input_training_tracks = training_tracks[:, :30, :]
    input_testing_tracks = testing_tracks[:, :30, :]

    indices = list(range(30, 60, 5))  # This will create a list [30, 35, 40, 45, 50, 55]
    if 59 not in indices:
        indices.append(59)  # Add the last point if it wasn't already included

    output_training_tracks = training_tracks[:, indices, :]
    output_testing_tracks = testing_tracks[:, indices, :]

    print("Shape of output_training_tracks: ", output_training_tracks.shape)
    print("Shape of output_testing_tracks: ", output_testing_tracks.shape)

    input_training_tensor = Variable(torch.Tensor(input_training_tracks)).to(device)
    output_training_tensor = Variable(torch.Tensor(output_training_tracks)).to(device)

    input_testing_tensor = Variable(torch.Tensor(input_testing_tracks)).to(device)
    output_testing_tensor = Variable(torch.Tensor(output_testing_tracks)).to(device)

    train_dataset = TensorDataset(input_training_tensor, output_training_tensor)
    test_dataset = TensorDataset(input_testing_tensor, output_testing_tensor)

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    vehicle_tracker = LSTMVehicleTracker(
        input_size=2,
        hidden_size=hidden_size,
        num_layers=num_layers,
        seq_length=30,
        output_pairs=7,
    )
    vehicle_tracker = vehicle_tracker.to(device)
    print(vehicle_tracker)

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(vehicle_tracker.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        vehicle_tracker.train()

        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            outputs = vehicle_tracker(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if epoch % epoch_print_interval == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}], Loss: {running_loss / len(train_loader):.8f}"
            )

        vehicle_tracker.eval()
        with torch.no_grad():
            val_outputs = vehicle_tracker(input_testing_tensor)
            val_loss = criterion(val_outputs, output_testing_tensor)
            if epoch % epoch_print_interval == 0:
                print(f"Validation Loss: {val_loss.item():.8f}")

    torch.save(vehicle_tracker.state_dict(), "./model_data/lstm_model.pth")

    cursor = db.cursor()
    cursor.execute("truncate public.prediction cascade;")
    db.commit()
    cursor.close()

    counter = 0
    total_distance = 0

    for iteration in range(verification_loops):
        random_track = random.choice(testing_results)
        coordinate_pairs = []
        for x, y in random_track["coordinates"]:
            coordinate_pairs.append((np.float64(x), np.float64(y)))

        coordinate_pairs = np.array(coordinate_pairs).reshape(-1, 2)
        normalized_coordinates = min_max_scaler.transform(coordinate_pairs)
        normalized_coordinates = normalized_coordinates.reshape(1, -1, 2)

        input_tensor = Variable(torch.Tensor(normalized_coordinates[:, :30, :])).to(
            device
        )

        output_tensor = Variable(
            torch.Tensor(normalized_coordinates[:, indices, :])
        ).to(device)

        vehicle_tracker.eval()
        with torch.no_grad():
            vehicle_tracker.eval()
            prediction = vehicle_tracker(input_tensor)

        # print(f"Prediction shape: {prediction.shape}")

        input_array = input_tensor.cpu().numpy().reshape(-1, 2)  # Reshape to (30, 2)
        output_array = output_tensor.cpu().numpy().reshape(-1, 2)  # Reshape to (1, 2)
        prediction_array = prediction.cpu().numpy().reshape(-1, 2)  # Reshape to (1, 2)

        input_inverted = min_max_scaler.inverse_transform(input_array)
        output_inverted = min_max_scaler.inverse_transform(output_array)
        prediction_inverted = min_max_scaler.inverse_transform(prediction_array)

        distance = np.sqrt(np.sum((output_inverted - prediction_inverted) ** 2))
        # print("Distance: ", distance)

        # if iteration % 100 == 0:
        if iteration in [10, 20, 30]:
            # Create a new prediction record
            cursor = db.cursor()
            cursor.execute("INSERT INTO public.prediction DEFAULT VALUES RETURNING id;")
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

    total_distance += distance

    # Increment the counter
    counter += 6

    average_distance = total_distance / counter
    print(f"Average distance: {average_distance:.2f}")

    end_time = time.time()

    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(
        "Training and testing took",
        int(hours),
        "hours,",
        int(minutes),
        "minutes, and",
        int(seconds),
        "seconds",
    )
