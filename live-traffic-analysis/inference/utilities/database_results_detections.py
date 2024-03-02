import numpy as np
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import psycopg2
import psycopg2.extras
from supervision.detection.core import Detections
from supervision.tracker.byte_tracker import matching
from supervision.tracker.byte_tracker.core import (
    STrack,
    joint_tracks,
    sub_tracks,
    remove_duplicate_tracks,
    TrackState,
    detections2boxes,
)


class Results:
    def __init__(self):
        self.names = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
            11: "stop sign",
            12: "parking meter",
            13: "bench",
            14: "bird",
            15: "cat",
            30: "skis",
            31: "snowboard",
            32: "sports ball",
            33: "kite",
            34: "baseball bat",
            35: "baseball glove",
            36: "skateboard",
            37: "surfboard",
            38: "tennis racket",
            39: "bottle",
            40: "wine glass",
            41: "cup",
            42: "fork",
            43: "knife",
            44: "spoon",
            45: "bowl",
            46: "banana",
            47: "apple",
            48: "sandwich",
            49: "orange",
            50: "broccoli",
            51: "carrot",
            52: "hot dog",
            53: "pizza",
            54: "donut",
            55: "cake",
            56: "chair",
            57: "couch",
            58: "potted plant",
            59: "bed",
            60: "dining table",
            61: "toilet",
            62: "tv",
            63: "laptop",
            64: "mouse",
            65: "remote",
            66: "keyboard",
            67: "cell phone",
            68: "microwave",
            69: "oven",
            70: "toaster",
            71: "sink",
            72: "refrigerator",
            73: "book",
            74: "clock",
            75: "vase",
            76: "scissors",
            77: "teddy bear",
            78: "hair drier",
            79: "toothbrush",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
            24: "backpack",
            25: "umbrella",
            26: "handbag",
            27: "tie",
            28: "suitcase",
            29: "frisbee",
        }


# this gets monkey patched into the supervision class
@classmethod
def from_database(cls, db, hash):
    cur = db.cursor()
    sql = """
    SELECT *
    FROM detections.detection_objects
    WHERE hash = %s
    """
    cur.execute(sql, (hash,))
    results = cur.fetchone()
    cur.close()

    if results is None:
        return None


    xyxy = np.array(results["xyxy"], dtype=np.float32)
    confidence = np.array(results["confidence"], dtype=np.float32)
    class_id = np.array(results["class_id"])
    speed = np.array(results["rate_mph"])

    return cls(xyxy=xyxy, confidence=confidence, class_id=class_id, speed=speed)
