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
from libraries.parameters import INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = VehicleLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)

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

# print("Normalized track points: ", len(normalized_tracks[0]))

sample_track_start = normalized_tracks[0][:30]

# print("Sample track points: ", len(sample_track_start))


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


# Convert prediction back to numpy array and denormalize
# sample_track_start_np = sample_track_start.numpy().squeeze()
denormalized_known_points = revert_normalization(
    sample_track_start, x_scaler, y_scaler, timestamp_scaler
)


print(
    "Input track points (length: ",
    len(denormalized_known_points),
    "):",
    denormalized_known_points,
)


print(
    "Predicted track points (denormalized) (length: ",
    len(denormalized_prediction),
    "):",
    denormalized_prediction,
)


# Create a new prediction record
cursor = db.cursor()
cursor.execute("truncate prediction cascade;")
db.commit()


# Create a new prediction record
cursor = db.cursor()
cursor.execute("INSERT INTO prediction DEFAULT VALUES RETURNING id;")
result = cursor.fetchone()
prediction_id = result["id"]
db.commit()

# Insert known points
for i, (x, y, timestamp) in enumerate(denormalized_known_points):
    cursor.execute(
        """
        INSERT INTO known_points (prediction, sequence_number, timestamp, location)
        VALUES (%s, %s, to_timestamp(%s), ST_SetSRID(ST_MakePoint(%s, %s), 2253));
        """,
        (prediction_id, i, float(timestamp), float(x), float(y)),
    )
db.commit()

# Insert predicted points
for i, (x, y, timestamp) in enumerate(denormalized_prediction):
    cursor.execute(
        """
        INSERT INTO predicted_points (prediction, sequence_number, timestamp, location)
        VALUES (%s, %s, to_timestamp(%s), ST_SetSRID(ST_MakePoint(%s, %s), 2253));
        """,
        (prediction_id, i, float(timestamp), float(x), float(y)),
    )
db.commit()

cursor.close()
