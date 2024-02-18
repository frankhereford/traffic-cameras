import torch
import random
import logging
from libraries.parameters import SEGMENT_LENGTH, PREDICTION_DISTANCE
from torch.utils.data import DataLoader, Dataset


class VehicleTrajectoryDataset(Dataset):
    def __init__(self, tracks):
        logging.info("Initializing VehicleTrajectoryDataset...")
        self.tracks = [track for track in tracks if len(track) >= SEGMENT_LENGTH]
        logging.info(f"Filtered {len(self.tracks)} tracks of sufficient length.")

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        # logging.info(f"Getting item {idx}...")
        track = self.tracks[idx]
        track_length = len(track)

        if track_length > SEGMENT_LENGTH:
            start = random.randint(0, track_length - SEGMENT_LENGTH)
            segment = track[start : start + SEGMENT_LENGTH]
        else:
            # For tracks exactly equal to SEGMENT_LENGTH
            segment = track

        return torch.tensor(segment[:-1]), torch.tensor(segment[1:])
