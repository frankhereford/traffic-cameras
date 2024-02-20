import json
import torch
import joblib
import numpy as np
from collections import deque
from datetime import datetime
from prediction_model.libraries.lstmvehicletracker import LSTMVehicleTracker
from torchinfo import summary

np.set_printoptions(suppress=True, precision=4)  # Adjust precision as needed


class VehicleHistoryAndInference:
    def __init__(self):
        self.data = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.min_max_scaler = joblib.load(
            "./prediction_model/model_data/min_max_scaler.save"
        )

        self.num_epochs = 400
        self.epoch_print_interval = 25
        self.hidden_size = 256
        self.num_layers = 2
        self.learning_rate = 0.0001
        self.batch_size = 64
        self.verification_loops = 1024

        self.vehicle_tracker = LSTMVehicleTracker(
            input_size=2,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            seq_length=30,
        )
        self.vehicle_tracker = self.vehicle_tracker.to(self.device)

        # Load the model
        self.vehicle_tracker.load_state_dict(
            torch.load("./prediction_model/model_data/lstm_model.pth")
        )

        summary(self.vehicle_tracker, input_size=(1, 30, 2))
        for param in self.vehicle_tracker.parameters():
            print(param.name, param.dtype)

        # Ensure to switch to eval mode if you're doing inference
        self.vehicle_tracker.eval()

    def record_data_point(
        self, vehicle_id, image_x, image_y, location_x, location_y, timestamp
    ):
        if vehicle_id not in self.data:
            self.data[vehicle_id] = deque(maxlen=30)
        self.data[vehicle_id].append(
            (image_x, image_y, location_x, location_y, timestamp)
        )

    def get_recent_history(self, vehicle_id):
        if vehicle_id in self.data and len(self.data[vehicle_id]) == 30:
            # Select the 3rd and 4th elements from each data point
            selected_data = [
                [data_point[2], data_point[3]] for data_point in self.data[vehicle_id]
            ]
            # Convert the selected data to a numpy array (tensor)
            tensor = np.array(selected_data)
            return tensor
        else:
            return False

    def infer_future_locations(self, trackers):
        predictions = []
        for tracker in trackers:
            history = self.get_recent_history(tracker)
            if history is not False:

                # Reshape the tracks array into a 2D array
                history_2d = history.reshape(-1, history.shape[-1])

                # Use the MinMaxScaler to transform the 2D tracks array
                history_scaled_2d = self.min_max_scaler.transform(history_2d)

                # Reshape the scaled array back to its original shape
                # history_scaled = history_scaled_2d.reshape(original_shape)
                history_scaled = history_scaled_2d.reshape(1, 30, 2)

                # print("Shape of history_scaled: ", history_scaled.shape)
                # print("history_scaled: ", history_scaled)

                history_scaled_tensor = torch.from_numpy(history_scaled).float()
                history_scaled_tensor = history_scaled_tensor.to(self.device)

                # print("Shape of history_scaled_tensor: ", history_scaled_tensor.shape)
                # print("history_scaled_tensor: ", history_scaled_tensor)

                with torch.no_grad():
                    self.vehicle_tracker.eval()
                    prediction = self.vehicle_tracker(history_scaled_tensor)

                # print("Shape of prediction: ", prediction.shape)
                # print("Prediction: ", prediction)

                input_array = (
                    prediction.cpu().numpy().reshape(-1, 2)
                )  # Reshape to (30, 2)

                input_inverted = self.min_max_scaler.inverse_transform(input_array)

                # print("Shape of input_inverted: ", input_inverted.shape)
                # print("input_inverted: ", input_inverted)

                predictions.append(input_inverted)
            else:
                predictions.append(None)
        return predictions

    def pretty_print_all_history(self):
        # Convert deque to list for JSON serialization
        all_data = {
            str(vehicle_id): [
                [
                    (
                        data_point.isoformat()
                        if isinstance(data_point, datetime)
                        else (
                            float(data_point)
                            if isinstance(data_point, np.float32)
                            else data_point
                        )
                    )
                    for data_point in data_points_list
                ]
                for data_points_list in list(data_points)
            ]
            for vehicle_id, data_points in self.data.items()
        }
        print(json.dumps(all_data, indent=4, sort_keys=True))
