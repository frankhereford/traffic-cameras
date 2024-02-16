#!/usr/bin/env python3

import torch
import psycopg2
import numpy as np
from dotenv import load_dotenv
import psycopg2.extras
from sklearn.preprocessing import MinMaxScaler
import joblib
from libraries.vehiclelstm import VehicleLSTM
import os
from libraries.parameters import SEGMENT_LENGTH, PREDICTION_DISTANCE
from libraries.normalize import normalize, revert_normalization


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 3  # X, Y, timestamp
hidden_size = 128  # This can be adjusted
num_layers = 2  # Number of LSTM layers
output_size = 3  # Predicting X, Y, timestamp


model = VehicleLSTM(input_size, hidden_size, num_layers, output_size)

# Move the model to the device
model = model.to(device)

model.load_state_dict(torch.load("vehicle_lstm_model.pth"))
model.to(device)
model.eval()


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
    detections.session_id DESC, detections.tracker_id desc
LIMIT 1
"""
cursor = db.cursor()
cursor.execute(query)
results = cursor.fetchall()
cursor.close()


# Function to load the scaler
def load_scaler(path):
    return joblib.load(path)


x_scaler = load_scaler("x_scalar.save")
y_scaler = load_scaler("y_scalar.save")
timestamp_scaler = load_scaler("timestamp_scaler.save")

normalized_tracks = [
    normalize(row, x_scaler, y_scaler, timestamp_scaler) for row in results
]

print("Normalized track points: ", len(normalized_tracks[0]))

sample_track_start = normalized_tracks[0][:30]

print("Sample track points: ", len(sample_track_start))


# Convert the sample data to a tensor, ensure it's float32, and reshape it to match the model's input shape
sample_tensor = (
    torch.tensor(sample_track_start, dtype=torch.float32).unsqueeze(0).to(device)
)

# Make a prediction
with torch.no_grad():
    model.eval()  # Ensure the model is in evaluation mode
    prediction = model(sample_tensor)

# Convert prediction back to numpy array and denormalize
prediction_np = prediction.cpu().numpy().squeeze()
denormalized_prediction = revert_normalization(
    prediction_np, x_scaler, y_scaler, timestamp_scaler
)

print("Input track points:", results)
print("Input track points:", results)


print("Predicted track points (denormalized):", denormalized_prediction)
print("Predicted track points (denormalized):", len(denormalized_prediction))


# Create a new prediction record
cursor = db.cursor()
cursor.execute("INSERT INTO prediction DEFAULT VALUES RETURNING id;")
result = cursor.fetchone()
prediction_id = result["id"]
db.commit()

# Insert known points
for i, row in results:
    cursor.execute(
        """
        INSERT INTO known_points (prediction, sequence_number, location)
        VALUES (%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 2253));
        """,
        (prediction_id, i, float(point[0]), float(point[1])),
    )
db.commit()
quit()
# Insert predicted points
for i, point in enumerate(denormalized_prediction):
    cursor.execute(
        """
        INSERT INTO predicted_points (prediction, sequence_number, location)
        VALUES (%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 2253));
        """,
        (prediction_id, i, float(point[0]), float(point[1])),
    )
db.commit()

cursor.close()

quit()


# normalize
def process_test_data(test_data, min_max_scaler):
    normalized_tracks = []

    for i, row in enumerate(test_data):
        track = list(
            zip(row["x_coords"], row["y_coords"], [float(t) for t in row["timestamps"]])
        )
        x_coords = np.array([point[0] for point in track]).reshape(-1, 1)
        y_coords = np.array([point[1] for point in track]).reshape(-1, 1)
        timestamps = np.array([point[2] for point in track]).reshape(-1, 1)

        x_coords_scaled = min_max_scaler.transform(x_coords)
        y_coords_scaled = min_max_scaler.transform(y_coords)
        timestamps_scaled = min_max_scaler.transform(timestamps)

        normalized_track = list(
            zip(
                x_coords_scaled.flatten(),
                y_coords_scaled.flatten(),
                timestamps_scaled.flatten(),
            )
        )
        normalized_tracks.append(normalized_track)

    return normalized_tracks


print("results: ", results)
normalized_test_data = process_test_data(results, min_max_scaler)
print("normalized test data", normalized_test_data)

# print(results)
# normalized_track = process_results(results, min_max_scaler)

# numpy_array = np.array(normalized_track)
# print("Normalized track: ", numpy_array)
quit()

# normalized_tracks = []

# for i, row in enumerate(results):
#     # logging.info(f"Processing row {i+1} of {len(results)}...")
#     track = list(
#         zip(row["x_coords"], row["y_coords"], [float(t) for t in row["timestamps"]])
#     )
#     # Extract coordinates and timestamps
#     x_coords = np.array([point[0] for point in track]).reshape(-1, 1)
#     y_coords = np.array([point[1] for point in track]).reshape(-1, 1)
#     timestamps = np.array([point[2] for point in track]).reshape(-1, 1)

#     # Initialize Min-Max Scaler
#     min_max_scaler = MinMaxScaler()

#     # Fit and transform
#     # logging.info("Scaling x coordinates...")
#     x_coords_scaled = min_max_scaler.fit_transform(x_coords)
#     # logging.info("Scaling y coordinates...")
#     y_coords_scaled = min_max_scaler.fit_transform(y_coords)
#     # logging.info("Scaling timestamps...")
#     timestamps_scaled = min_max_scaler.fit_transform(timestamps)

#     # Combine back into normalized tracks
#     normalized_track = list(
#         zip(
#             x_coords_scaled.flatten(),
#             y_coords_scaled.flatten(),
#             timestamps_scaled.flatten(),
#         )
#     )


n = 29  # Replace with the number of points you want
first_n_points = numpy_array[:n, :]

print("First n points: ", first_n_points.shape)

# Reshape the numpy array and convert to tensor
test_tensor = torch.from_numpy(first_n_points).unsqueeze(0).float()

# Move the tensor to the device
test_tensor = test_tensor.to(device)

# Correct the shape of the tensor for LSTM input
# Remove the extra dimension
test_tensor = test_tensor.squeeze(1)

print("test_tensor shape after squeeze: ", test_tensor.shape)

# Now test_tensor should have the shape [1, 103, 3] which is suitable for LSTM

with torch.no_grad():
    predictions = model(test_tensor)


print("predictions.shape: ", predictions.shape)
predictions_numpy = predictions.cpu().detach().numpy()

print(predictions_numpy)


# # Denormalize predictions
# denormalized_predictions = denormalize(predictions.cpu().numpy(), min_max_scaler)

# # Output for GIS software
# # ... [Code to output/save denormalized_predictions for GIS evaluation] ...

# db.close()
