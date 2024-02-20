import supervision as sv
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
from utilities.transformation import read_points_file
from torch_tps import ThinPlateSpline
import torch
from datetime import datetime, timedelta
import re
import redis
from PIL import Image, ImageDraw, ImageFont
import pytz


from utilities.sql import (
    prepare_detection,
    insert_detections,
    create_new_session,
    get_class_id,
    compute_speed,
    get_future_locations_for_trackers,
)

COLORS = sv.ColorPalette.default()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DETECTION_ZONES = [
    np.array(
        [
            [715, 964],
            [1639, 832],
            [1814, 923],
            [796, 1076],
        ]
    ),
    np.array(
        [
            [242, 670],
            [315, 884],
            [129, 911],
            [105, 681],
        ]
    ),
    np.array(
        [
            [464, 595],
            [433, 520],
            [982, 496],
            [1078, 545],
        ]
    ),
    np.array(
        [
            [1349, 581],
            [1472, 570],
            [1779, 720],
            [1691, 733],
        ]
    ),
]

CENTER_POINTS_TO_AVOID = [
    (1368, 371, 15),  # bank atm
    (1814, 672, 15),  # lower-right flag
    (1700, 458, 15),  # right street closest light
    (1610, 437, 15),  # right street middle light
    (1770, 674, 15),  # lower right flag
    (496, 490, 15),  # left street light, closest
    (1833, 413, 15),  # right street far street light near the pole
    (667, 739, 15),  # lower left light
    (640, 729, 15),  # lower left light
    (38, 432, 10),  # clock
    (637, 731, 25),  # light lower left
    (375, 442, 20),
    (1044, 580, 20),
    (945, 396, 20),
    (668, 718, 20),
    (1526, 428, 10),
    (1740, 649, 15),
    (1762, 733, 20),
    (497, 488, 10),
    (871, 594, 10),
    (1041, 584, 10),
    (1194, 571, 10),
    (1081, 401, 10),
    (488, 245, 10),
    (1449, 142, 10),
    (534, 264, 10),
    (1550, 357, 10),
    (1049, 367, 10),
    (70, 501, 10),
]


class VideoProcessor:
    def __init__(self, input, output, db, location_tracker):
        self.location_tracker = location_tracker
        self.source_video_path = input

        # Extract the date and time from the file path using regex
        match = re.search(r"(\d{8})-(\d{6})", self.source_video_path)
        if match:
            date_str = match.group(1)
            time_str = match.group(2)

            # Combine date and time strings
            datetime_str = date_str + time_str

            # Parse the combined string into a datetime object
            self.video_datetime = datetime.strptime(datetime_str, "%Y%m%d%H%M%S")

            print("Parsed datetime:", self.video_datetime)
        else:
            print("Date and time pattern not found in the file path")

        self.output_video_path = output
        if not self.source_video_path or not self.output_video_path:
            raise ValueError("Input and output video paths are required.")
        self.model = YOLO("yolov8m.pt")
        self.tracker = sv.ByteTrack()
        self.video_info = sv.VideoInfo.from_video_path(self.source_video_path)
        self.box_annotator = sv.BoxCornerAnnotator(color=COLORS)
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2, trace_length=30, position=sv.Position.BOTTOM_CENTER
        )

        self.class_label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=2,
            text_position=sv.Position.TOP_CENTER,
            text_color=sv.Color(r=0, g=0, b=0),
        )

        self.speed_label_annotator = sv.LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_padding=2,
            text_position=sv.Position.BOTTOM_CENTER,
            text_color=sv.Color(r=0, g=0, b=0),
        )

        self.smoother = sv.DetectionsSmoother(length=3)

        self.db = db
        self.cursor = self.db.cursor()
        self.queue_size = 100
        self.queued_inserts = 0
        # self.cursor.execute("TRUNCATE sessions CASCADE")
        # db.commit()

        self.session = create_new_session(self.cursor)
        db.commit()

        self.coordinates = read_points_file("./gcp/coldwater_mi.points")
        self.tps = ThinPlateSpline(0.5)
        self.tps.fit(
            self.coordinates["image_coordinates"], self.coordinates["map_coordinates"]
        )
        self.reverse_tps = ThinPlateSpline(0.5)
        self.reverse_tps.fit(
            self.coordinates["map_coordinates"], self.coordinates["image_coordinates"]
        )

        self.frame_number = 0

        self.redis = redis.Redis(host="localhost", port=6379, db=0)

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        if self.output_video_path:
            with sv.VideoSink(self.output_video_path, self.video_info) as sink:
                for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)

    def build_keep_list_tensor(self, data_obj, center_points):
        xyxy_tensor = data_obj.xyxy
        computed_center_points = (xyxy_tensor[:, :2] + xyxy_tensor[:, 2:4]) / 2

        # Initialize an array for the boolean values
        bool_array = []
        for cp in computed_center_points:
            bool_value = True
            for center_x, center_y, distance_threshold in center_points:
                center_tensor = torch.tensor(
                    [center_x, center_y], device=xyxy_tensor.device
                )
                dist_to_center = torch.norm(center_tensor - cp)
                if dist_to_center < distance_threshold:
                    bool_value = False
                    break
            bool_array.append(bool_value)

        return bool_array

    def filter_tensors(self, data_obj, keep_list):
        # Ensuring all tensors in the object are of the same length as keep_list
        for attr in vars(data_obj):
            tensor = getattr(data_obj, attr)
            if torch.is_tensor(tensor) and len(tensor) != len(keep_list):
                raise ValueError("Tensor and keep_list lengths do not match.")

        for attr in vars(data_obj):
            tensor = getattr(data_obj, attr)
            if torch.is_tensor(tensor):
                # Filtering the tensor
                filtered_tensor = tensor[torch.tensor(keep_list)]
                setattr(data_obj, attr, filtered_tensor)

        return data_obj

    def transform_model_results_into_image_space(self, predictions):
        none_indices = [i for i, location in enumerate(predictions) if location is None]

        if len(none_indices) == len(predictions):
            return

        locations_tuples = np.concatenate([p for p in predictions if p is not None])

        locations_tensor = torch.tensor(locations_tuples)
        image_space_locations = self.reverse_tps.transform(locations_tensor).cpu()

        reshaped_image_space_locations = image_space_locations.reshape(-1, 6, 2)

        reshaped_image_space_locations = reshaped_image_space_locations.tolist()

        for index in none_indices:
            reshaped_image_space_locations.insert(index, None)

        return reshaped_image_space_locations

    def transform_into_image_space(self, future_locations):

        image_space_future_locations = future_locations

        none_indices = [
            i for i, location in enumerate(future_locations) if location is None
        ]

        # Build locations_tuples, excluding None elements
        locations_tuples = [
            (location["x"], location["y"])
            for location in future_locations
            if location is not None
        ]

        if locations_tuples is not None and len(locations_tuples) > 0:
            prediction_tensor = torch.tensor(locations_tuples)

            image_space_prediction_tensor = self.reverse_tps.transform(
                prediction_tensor
            )

            image_space_future_locations = [
                tuple(location) for location in image_space_prediction_tensor.tolist()
            ]

            for index in none_indices:
                image_space_future_locations.insert(index, None)

        return image_space_future_locations

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        result = self.model(frame, verbose=False)[0]

        keep_list = self.build_keep_list_tensor(result.boxes, CENTER_POINTS_TO_AVOID)
        result.boxes = self.filter_tensors(result.boxes, keep_list)

        detections = sv.Detections.from_ultralytics(result)
        detections = self.tracker.update_with_detections(detections)
        detections = self.smoother.update_with_detections(detections)

        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

        detections_xy = torch.tensor(points).float()
        detections_latlon = self.tps.transform(detections_xy)
        if not (
            detections.tracker_id is None
            or points is None
            or detections_latlon is None
            or detections.class_id is None
        ):

            for tracker_id, point, location, class_id in zip(
                detections.tracker_id, points, detections_latlon, detections.class_id
            ):
                our_class_id = get_class_id(
                    self.db,
                    self.cursor,
                    self.session,
                    int(class_id),
                    result.names[class_id],
                )
                prepare_detection(
                    tracker_id,
                    our_class_id,
                    point[0],
                    point[1],
                    self.video_datetime.timestamp(),
                    self.session,
                    location[0],
                    location[1],
                )
                self.queued_inserts += 1

                self.location_tracker.record_data_point(
                    tracker_id,
                    point[0],
                    point[1],
                    float(location[0]),
                    float(location[1]),
                    self.video_datetime,
                )

            if self.queued_inserts >= self.queue_size:
                # print(f"Inserting {queued_inserts} detections")
                insert_detections(self.db, self.cursor)
                self.queued_inserts = 0

        predictions = self.location_tracker.infer_future_locations(
            detections.tracker_id
        )

        image_space_future_locations = self.transform_model_results_into_image_space(
            predictions
        )

        # future_locations = get_future_locations_for_trackers(
        #     self.cursor, self.session, detections.tracker_id
        # )

        # print("future locations: ", future_locations)
        # image_space_future_locations = self.transform_into_image_space(future_locations)
        # print("image space future locations: ", image_space_future_locations)

        # image_space_future_locations = []

        self.frame_number += 1
        one_thirtieth_second_in_microseconds = int(1_000_000 / 30)
        self.video_datetime += timedelta(
            microseconds=one_thirtieth_second_in_microseconds
        )
        # print(f"Frame time: {self.video_datetime}")

        return self.annotate_frame(
            frame, detections, result, image_space_future_locations
        )

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections, result, future_locations
    ) -> np.ndarray:

        # labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

        center_points = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)

        annotated_frame = frame.copy()

        if not (detections.tracker_id is None or detections.class_id is None):

            class_labels = [
                f"{result.names[class_id].title()} #{tracker_id} (X: {int(point[0])}, Y: {int(point[1])})"
                # f"{result.names[class_id].title()} #{tracker_id}"
                for tracker_id, class_id, point in zip(
                    detections.tracker_id, detections.class_id, center_points
                )
            ]

            speeds = []
            for tracker_id in detections.tracker_id:
                # Try to get the speed from Redis
                speed = self.redis.get(f"speed:{self.session}:{tracker_id}")
                if speed is None:
                    # If the speed is not in Redis, compute it and store it in Redis with a 1 second expiration
                    speed = compute_speed(self.cursor, self.session, tracker_id, 30)
                    # print("fresh speed: ", speed)
                    # Convert None to 'None' before storing in Redis
                    self.redis.set(
                        f"speed:{self.session}:{tracker_id}",
                        speed if speed is not None else "None",
                        ex=1,
                    )
                else:
                    # Decode bytes to string and If the speed is in Redis, convert it to a float if it's not 'None'
                    speed = speed.decode("utf-8")
                    speed = float(speed) if speed != "None" else None
                speeds.append(speed)

            speed_labels = [
                (f"{speed:.1f} MPH" if speed is not None else "calculating...")
                for speed in speeds
            ]

            annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
            annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)

            annotated_frame = self.class_label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=class_labels,
            )

            annotated_frame = self.speed_label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=speed_labels,
            )

            image = Image.fromarray(annotated_frame)

            # Create a draw object
            draw = ImageDraw.Draw(image)

            # Define the radius of the circles
            radius = 5

            # # Iterate over the future_locations list
            if future_locations is not None:
                for location in future_locations:
                    if location is not None:
                        previous_loc = None
                        for loc in location:
                            # Convert coordinates to integers
                            x, y = map(int, loc)
                            # Calculate the bounding box of the circle
                            upper_left = (x - radius, y - radius)
                            lower_right = (x + radius, y + radius)

                            # Draw a blue circle
                            # we're in some strange non-RGB world here
                            draw.ellipse([upper_left, lower_right], fill="red")

                            # If there is a previous location, draw a line from the previous location to the current location
                            if previous_loc is not None:
                                draw.line([previous_loc, (x, y)], fill="red", width=3)
                            # Update the previous location
                            previous_loc = (x, y)
                # Convert the PIL image back to a numpy array
            annotated_frame = np.array(image)

        image = Image.fromarray(annotated_frame)
        draw = ImageDraw.Draw(image)

        # Convert the datetime object from UTC to Central Time
        central = pytz.timezone("US/Central")
        video_datetime_central = self.video_datetime.astimezone(central)

        datetime_str = video_datetime_central.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=18
        )

        # Draw the datetime string onto the image
        draw.text((10, 10), datetime_str, fill="white", font=font)

        for center_point in CENTER_POINTS_TO_AVOID:
            # Calculate the bounding box of the circle
            upper_left = (
                center_point[0] - center_point[2],
                center_point[1] - center_point[2],
            )
            lower_right = (
                center_point[0] + center_point[2],
                center_point[1] + center_point[2],
            )

            # Draw a red circle
            draw.ellipse([upper_left, lower_right], outline="blue")
        annotated_frame = np.array(image)

        return annotated_frame
