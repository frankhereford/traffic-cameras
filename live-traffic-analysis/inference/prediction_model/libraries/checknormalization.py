import random
import numpy as np
from libraries.normalize import normalize, revert_normalization


def check_normalization_denormalization(results, x_scaler, y_scaler, timestamp_scaler):
    # Randomly select a track
    random_track = random.choice(results)
    random_track["timestamps"] = [float(t) for t in random_track["timestamps"]]

    sample_x_coords = []
    sample_y_coords = []
    sample_timestamps = []

    sample_x_coords.extend(random_track["x_coords"])
    sample_y_coords.extend(random_track["y_coords"])
    sample_timestamps.extend([float(t) for t in random_track["timestamps"]])

    # Print the selected track
    original_track = np.array(
        list(zip(sample_x_coords, sample_y_coords, sample_timestamps))
    )
    print("Original Track (", original_track.shape, "):\n", original_track)

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
    normalized_sample_data = np.array(normalized_sample_data)
    print(
        "Normalized Sample Data (",
        normalized_sample_data.shape,
        "):\n",
        normalized_sample_data,
    )

    # Denormalize the sample data
    denormalized_sample_data = revert_normalization(
        normalized_sample_data, x_scaler, y_scaler, timestamp_scaler
    )

    # Print the denormalized sample data
    denormalized_sample_data = np.array(denormalized_sample_data)
    print(
        "Denormalized Sample Data (",
        denormalized_sample_data.shape,
        "):\n",
        denormalized_sample_data,
    )
