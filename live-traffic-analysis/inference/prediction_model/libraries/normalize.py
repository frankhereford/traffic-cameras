import numpy as np


def normalize(row, x_scaler, y_scaler, timestamp_scaler):
    track = list(
        zip(row["x_coords"], row["y_coords"], [float(t) for t in row["timestamps"]])
    )

    # Ensure the data is in np.float64
    x_coords = np.array([point[0] for point in track], dtype=np.float64).reshape(-1, 1)
    y_coords = np.array([point[1] for point in track], dtype=np.float64).reshape(-1, 1)
    timestamps = np.array([point[2] for point in track], dtype=np.float64).reshape(
        -1, 1
    )

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
