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
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO)


SEGMENT_LENGTH = 30
PREDICTION_DISTANCE = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VehicleTrajectoryDataset(Dataset):
    def __init__(self, tracks):
        logging.info("Initializing VehicleTrajectoryDataset...")
        self.tracks = [track for track in tracks if len(track) >= SEGMENT_LENGTH]
        logging.info(f"Filtered {len(self.tracks)} tracks of sufficient length.")

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        logging.info(f"Getting item {idx}...")
        track = self.tracks[idx]
        track_length = len(track)

        if track_length > SEGMENT_LENGTH:
            start = random.randint(0, track_length - SEGMENT_LENGTH)
            segment = track[start : start + SEGMENT_LENGTH]
        else:
            # For tracks exactly equal to SEGMENT_LENGTH
            segment = track

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
    COUNT(*) > ({SEGMENT_LENGTH} + {PREDICTION_DISTANCE} + 5)
ORDER BY 
    paths.distance DESC, detections.session_id DESC, detections.tracker_id;
"""

    cursor = db.cursor()
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()

    normalized_tracks = []

    # Now you can work with the results
    logging.info(f"Processing {len(results)} tracks...")
    for i, row in enumerate(results):
        # logging.info(f"Processing row {i+1} of {len(results)}...")
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
        # logging.info("Scaling x coordinates...")
        x_coords_scaled = min_max_scaler.fit_transform(x_coords)
        # logging.info("Scaling y coordinates...")
        y_coords_scaled = min_max_scaler.fit_transform(y_coords)
        # logging.info("Scaling timestamps...")
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

    logging.info("Initializing the dataset with the normalized tracks...")
    # Initialize the dataset with the normalized tracks
    dataset = VehicleTrajectoryDataset(normalized_tracks)

    logging.info("Creating DataLoader...")
    # DataLoader can now be used with this dataset
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # # Example: Iterate over the DataLoader
    # for inputs, targets in dataloader:
    # print("Input:", inputs)
    # print("Target:", targets)
    # print("Input shape:", inputs.shape)
    # print("Target shape:", targets.shape)

logging.info(f"data loader length: {len(dataloader)}")


class VehicleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(VehicleLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer to predict the next position
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Predict the next position for each time step
        out = self.fc(out)  # Applying the fully connected layer to all time steps

        return out[:, -1, :]  # Returning the prediction for the next time step only


# Fetch a single batch from the DataLoader
dataiter = iter(dataloader)
if len(dataloader) > 0:
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
vehicle_lstm_model = VehicleLSTM(input_size, hidden_size, num_layers, output_size).to(
    device
)

logging.info(vehicle_lstm_model)


# Loss function
criterion = nn.MSELoss()

# Optimizer (Adam is a good default)
optimizer = optim.Adam(vehicle_lstm_model.parameters(), lr=0.001)
