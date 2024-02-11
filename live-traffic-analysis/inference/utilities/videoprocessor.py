import supervision as sv
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
from utilities.transformation import read_points_file
from torch_tps import ThinPlateSpline
import torch
import time

from utilities.sql import (
    prepare_detection,
    insert_detections,
    create_new_session,
    get_class_id,
    compute_speed,
)

COLORS = sv.ColorPalette.default()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VideoProcessor:
    def __init__(self, input, output, db):
        self.source_video_path = input
        self.output_video_path = output
        if not self.source_video_path or not self.output_video_path:
            raise ValueError("Input and output video paths are required.")
        self.model = YOLO("yolov8m.pt")
        self.tracker = sv.ByteTrack()
        self.video_info = sv.VideoInfo.from_video_path(self.source_video_path)
        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.smoother = sv.DetectionsSmoother(length=5)

        self.db = db
        self.cursor = self.db.cursor()
        self.queue_size = 100
        self.queued_inserts = 0
        self.cursor.execute("TRUNCATE sessions CASCADE")
        db.commit()

        self.session = create_new_session(self.cursor)
        db.commit()

        self.coordinates = read_points_file("./gcp/coldwater_mi.points")
        self.tps = ThinPlateSpline(0.5)
        self.tps.fit(
            self.coordinates["image_coordinates"], self.coordinates["map_coordinates"]
        )

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        if self.output_video_path:
            with sv.VideoSink(self.output_video_path, self.video_info) as sink:
                for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections, result
    ) -> np.ndarray:

        # labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]

        center_points = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)

        annotated_frame = frame.copy()

        if not (detections.tracker_id is None or detections.class_id is None):

            class_labels = [
                f"{result.names[class_id].title()} #{tracker_id} (X: {int(point[0])}, Y: {int(point[1])})"
                for tracker_id, class_id, point in zip(
                    detections.tracker_id, detections.class_id, center_points
                )
            ]

            annotated_frame = self.box_annotator.annotate(
                annotated_frame, detections, labels=class_labels
            )
        return annotated_frame

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

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        result = self.model(frame, verbose=False)[0]

        center_points_to_avoid = [
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
            ## parked cars now,
        ]
        keep_list = self.build_keep_list_tensor(result.boxes, center_points_to_avoid)
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
                    time.time(),
                    self.session,
                    location[0],
                    location[1],
                )
                self.queued_inserts += 1
            if self.queued_inserts >= self.queue_size:
                # print(f"Inserting {queued_inserts} detections")
                insert_detections(self.db, self.cursor)
                self.queued_inserts = 0

        return self.annotate_frame(frame, detections, result)
