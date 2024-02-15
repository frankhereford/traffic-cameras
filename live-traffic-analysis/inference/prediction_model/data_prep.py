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

# Set print options
np.set_printoptions(threshold=5, formatter={"float": "{: 0.3f}".format})


# Set up logging
logging.basicConfig(level=logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(row, x_scaler, y_scaler, timestamp_scaler):
    track = list(
        zip(row["x_coords"], row["y_coords"], [float(t) for t in row["timestamps"]])
    )

    x_coords = np.array([point[0] for point in track]).reshape(-1, 1)
    y_coords = np.array([point[1] for point in track]).reshape(-1, 1)
    timestamps = np.array([point[2] for point in track]).reshape(-1, 1)

    x_coords_scaled = x_scaler.transform(x_coords)
    y_coords_scaled = y_scaler.transform(y_coords)
    timestamps_scaled = timestamp_scaler.transform(timestamps)

    normalized_track = list(
        zip(
            x_coords_scaled.flatten(),
            y_coords_scaled.flatten(),
            timestamps_scaled.flatten(),
        )
    )
    return normalized_track


def revert_normalization(normalized_track, x_scaler, y_scaler, timestamp_scaler):
    x_coords = np.array([point[0] for point in normalized_track]).reshape(-1, 1)
    y_coords = np.array([point[1] for point in normalized_track]).reshape(-1, 1)
    timestamps = np.array([point[2] for point in normalized_track]).reshape(-1, 1)

    x_coords_original = x_scaler.inverse_transform(x_coords)
    y_coords_original = y_scaler.inverse_transform(y_coords)
    timestamps_original = timestamp_scaler.inverse_transform(timestamps)

    return list(
        zip(
            x_coords_original.flatten(),
            y_coords_original.flatten(),
            timestamps_original.flatten(),
        )
    )


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
        x_coords_all.extend(row["x_coords"])
        y_coords_all.extend(row["y_coords"])
        timestamps_all.extend([float(t) for t in row["timestamps"]])

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

    # Existing code...
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

    quit()

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


# Apply this function after training your model and scaler
save_scaler(min_max_scaler, "scaler.save")


logging.info(f"train_dataloader length: {len(train_dataloader)}")
logging.info(f"val_dataloader length: {len(val_dataloader)}")
logging.info(f"test_dataloader length: {len(test_dataloader)}")


# # Fetch a single batch from the DataLoader
# dataiter = iter(dataloader)
# if len(dataloader) > 0:
#     sample_inputs, sample_targets = next(dataiter)
#     # Print the shape of the inputs and targets
#     logging.info(f"Shape of input batch: {sample_inputs.shape}")
#     logging.info(f"Shape of target batch: {sample_targets.shape}")
# else:
#     logging.error("DataLoader is empty.")

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


# Number of epochs
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0

    for i, data in enumerate(train_dataloader, 0):
        # Get the inputs; data is a list of [inputs, targets]
        inputs, targets = data
        inputs, targets = inputs.float().to(device), targets.float().to(
            device
        )  # Convert to float32

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

# Assuming you have a validation dataloader (val_dataloader)
vehicle_lstm_model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    val_loss = 0.0
    for i, data in enumerate(val_dataloader, 0):
        inputs, targets = data
        inputs, targets = inputs.float().to(device), targets.float().to(device)

        outputs = vehicle_lstm_model(inputs)
        loss = criterion(outputs, targets)
        val_loss += loss.item()

    average_val_loss = val_loss / len(val_dataloader)
    logging.info(f"Validation Loss: {average_val_loss:.3f}")


vehicle_lstm_model.eval()  # Set the model to evaluation mode

test_loss = 0.0
with torch.no_grad():
    for i, data in enumerate(test_dataloader, 0):
        inputs, targets = data
        inputs, targets = inputs.float().to(device), targets.float().to(device)

        # Forward pass: compute predicted outputs by passing inputs to the model
        outputs = vehicle_lstm_model(inputs)

        # Calculate the loss
        loss = criterion(outputs, targets)
        test_loss += loss.item()

average_test_loss = test_loss / len(test_dataloader)
logging.info(f"Test Loss: {average_test_loss:.3f}")

# Later, during prediction, load the scaler
# min_max_scaler = load_scaler("scaler.save")


# # Modify the predict function to include denormalization
# def predict(model, input_data, scaler):
#     processed_input = preprocess_input(input_data, scaler)
#     with torch.no_grad():
#         output = model(processed_input)
#     denormalized_output = denormalize(output.numpy(), scaler)
#     return denormalized_output
