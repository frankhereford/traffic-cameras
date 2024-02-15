#!/usr/bin/env python3

import os
import torch
import random
import logging
import psycopg2
import numpy as np
import torch.nn as nn
import psycopg2.extras
import torch.optim as optim
from decimal import Decimal
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import joblib
from libraries.vehiclelstm import VehicleLSTM
from libraries.vehicletrajectorydataset import VehicleTrajectoryDataset
from libraries.parameters import SEGMENT_LENGTH, PREDICTION_DISTANCE
from libraries.normalize import normalize, revert_normalization

# Set print options
np.set_printoptions(threshold=5, formatter={"float": "{: 0.3f}".format})


# Set up logging
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to save the scaler
def save_scaler(scaler, path):
    joblib.dump(scaler, path)


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
            COUNT(*) > ({SEGMENT_LENGTH} + {PREDICTION_DISTANCE} + 5)
        ORDER BY 
            paths.distance DESC, detections.session_id DESC, detections.tracker_id;
        """

    cursor = db.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()

    x_coords_all = []
    y_coords_all = []
    timestamps_all = []

    for row in results:
        # Convert data to np.float64 immediately after fetching
        x_coords_all.extend([np.float64(x) for x in row["x_coords"]])
        y_coords_all.extend([np.float64(y) for y in row["y_coords"]])
        timestamps_all.extend([np.float64(t) for t in row["timestamps"]])

    x_coords_all = np.array(x_coords_all).reshape(-1, 1)
    y_coords_all = np.array(y_coords_all).reshape(-1, 1)
    timestamps_all = np.array(timestamps_all).reshape(-1, 1)

    # Initialize Min-Max Scalers
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    timestamp_scaler = MinMaxScaler()

    # Fit the scalers
    x_scaler.fit(x_coords_all)
    y_scaler.fit(y_coords_all)
    timestamp_scaler.fit(timestamps_all)

    normalized_tracks = []

    normalized_tracks = [
        normalize(row, x_scaler, y_scaler, timestamp_scaler) for row in results
    ]

    # Randomly select a track
    random_track = random.choice(results)
    random_track["timestamps"] = [float(t) for t in random_track["timestamps"]]

    sample_x_coords = []
    sample_y_coords = []
    sample_timestamps = []

    sample_x_coords.extend(row["x_coords"])
    sample_y_coords.extend(row["y_coords"])
    sample_timestamps.extend([float(t) for t in row["timestamps"]])

    # Print the selected track
    print("Original track:")
    print(np.array(list(zip(sample_x_coords, sample_y_coords, sample_timestamps))))

    # Create a dictionary for the sample data
    sample_data = {
        "x_coords": sample_x_coords,
        "y_coords": sample_y_coords,
        "timestamps": sample_timestamps,
    }

    # Normalize the sample data
    normalized_sample_data = normalize(
        sample_data, x_scaler, y_scaler, timestamp_scaler
    )

    # Print the normalized sample data
    print("Normalized sample data:")
    print(np.array(normalized_sample_data))

    # Denormalize the sample data
    denormalized_sample_data = revert_normalization(
        normalized_sample_data, x_scaler, y_scaler, timestamp_scaler
    )

    # Print the denormalized sample data
    print("Denormalized sample data:")
    print(np.array(denormalized_sample_data))

    # Split the normalized_tracks into training and temporary datasets (combining validation and testing)
    train_tracks, temp_tracks = train_test_split(
        normalized_tracks, test_size=0.3, random_state=42
    )

    # Split the temporary dataset into validation and testing datasets
    val_tracks, test_tracks = train_test_split(
        temp_tracks, test_size=0.5, random_state=42
    )

    train_dataset = VehicleTrajectoryDataset(train_tracks)
    val_dataset = VehicleTrajectoryDataset(val_tracks)
    test_dataset = VehicleTrajectoryDataset(test_tracks)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


save_scaler(x_scaler, "x_scalar.save")
save_scaler(y_scaler, "y_scalar.save")
save_scaler(timestamp_scaler, "timestamp_scaler.save")

logging.info(f"train_dataloader length: {len(train_dataloader)}")
logging.info(f"val_dataloader length: {len(val_dataloader)}")
logging.info(f"test_dataloader length: {len(test_dataloader)}")


# # Fetch a single batch from the DataLoader
dataiter = iter(train_dataloader)
if len(train_dataloader) > 0:
    sample_inputs, sample_targets = next(dataiter)
    # Print the shape of the inputs and targets
    logging.info(f"Shape of input batch: {sample_inputs.shape}")
    logging.info(f"Shape of target batch: {sample_targets.shape}")
else:
    logging.error("DataLoader is empty.")

# Define the parameters for the LSTM model
input_size = 3  # X, Y, timestamp
hidden_size = 128  # This can be adjusted
num_layers = 2  # Number of LSTM layers
output_size = 3  # Predicting X, Y, timestamp

# Instantiate the model
vehicle_lstm_model = (
    VehicleLSTM(input_size, hidden_size, num_layers, output_size).double().to(device)
)


logging.info(vehicle_lstm_model)


# Loss function
criterion = nn.MSELoss()

# Optimizer (Adam is a good default)
optimizer = optim.Adam(vehicle_lstm_model.parameters(), lr=0.001)


# Number of epochs
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0

    for i, data in enumerate(train_dataloader, 0):
        # Get the inputs; data is a list of [inputs, targets]
        inputs, targets = data
        inputs, targets = inputs.double().to(device), targets.double().to(
            device
        )  # Convert to float64

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = vehicle_lstm_model(inputs)

        # Calculate the loss
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            logging.info(
                "[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100)
            )
            running_loss = 0.0

logging.info("Finished Training")
torch.save(vehicle_lstm_model.state_dict(), "vehicle_lstm_model.pth")

vehicle_lstm_model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    val_loss = 0.0
    for i, data in enumerate(val_dataloader, 0):
        inputs, targets = data
        inputs, targets = inputs.double().to(device), targets.double().to(
            device
        )  # Convert to double

        outputs = vehicle_lstm_model(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()

    average_val_loss = val_loss / len(val_dataloader)
    logging.info(f"Validation Loss: {average_val_loss:.3f}")


test_loss = 0.0
with torch.no_grad():
    for i, data in enumerate(test_dataloader, 0):
        inputs, targets = data
        inputs, targets = inputs.double().to(device), targets.double().to(
            device
        )  # Convert to double

        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = vehicle_lstm_model(inputs)

        # Calculate the loss
        loss = criterion(outputs, targets)
        test_loss += loss.item()

average_test_loss = test_loss / len(test_dataloader)
logging.info(f"Test Loss: {average_test_loss:.3f}")
