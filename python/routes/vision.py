from flask import Flask, request, jsonify, send_file
import logging
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
from io import BytesIO
import time
import torch
from torch_tps import ThinPlateSpline

import redis
import pickle

logging.basicConfig(level=logging.INFO)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-101", revision="no_timm"
)
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-101", revision="no_timm"
)


def throttle(func):
    throttle._last_called = None

    def wrapper(*args, **kwargs):
        if throttle._last_called is not None:
            time_since_last_call = time.time() - throttle._last_called
            sleep_time = int(round(max(0, 1 - time_since_last_call), 0))
            print("sleep time: ", sleep_time)
            time.sleep(sleep_time)
        throttle._last_called = time.time()
        return func(*args, **kwargs)

    return wrapper


def extract_points(locations):
    cctv_points = torch.tensor([[location.x, location.y] for location in locations])
    map_points = torch.tensor(
        [[location.latitude, location.longitude] for location in locations]
    )
    return cctv_points, map_points


def vision(db, redis):
    while True:
        process_one_image(db, redis)


@throttle
def process_one_image(db, redis):
    job = db.image.find_first(
        where={
            "detectionsProcessed": False,
        },
        include={"camera": True},
    )
    if job is None:
        return

    print(job.hash)

    key = f"images:{job.hash}"

    if redis.exists(key):
        serialized_data = redis.get(key)
        image_byte_stream = pickle.loads(serialized_data)
    else:
        db.image.update(
            where={
                "id": job.id,
            },
            data={
                "detectionsProcessed": True,
            },
        )
        return

    image = Image.open(image_byte_stream)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9
    )[0]

    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        box = [round(i, 2) for i in box.tolist()]
        logging.info(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

        print("box: ", box)

        # box:  [602.06, 217.31, 623.63, 260.28]
        db.detection.create(
            data={
                "label": model.config.id2label[label.item()],
                "confidence": round(score.item(), 3),
                "xMin": box[0],
                "yMin": box[1],
                "xMax": box[2],
                "yMax": box[3],
                "imageId": job.id,
            }
        )

    # starting thin plate spline code

    logging.info("starting thin plate spline code")
    logging.info(job.camera.id)

    camera = db.camera.find_first(
        where={"id": job.camera.id}, include={"Location": True}
    )

    cctv_points, map_points = extract_points(camera.Location)

    if len(cctv_points) >= 5:
        tps = ThinPlateSpline(0.5)

        cctv_points = cctv_points.float()
        map_points = map_points.float()

        # Fit the surfaces
        tps.fit(cctv_points, map_points)

        image = db.image.find_first(
            where={"cameraId": camera.id},
            include={"detections": True},
            order={"createdAt": "desc"},
        )
        # logging.info(image.detections)

        # objects_to_transform = ["person", "car"]
        points_to_transform = torch.tensor(
            [
                [(d.xMin + d.xMax) / 2, d.yMax]
                for d in image.detections
                # if d.label in objects_to_transform
            ]
        ).float()
        # logging.info("points to transform")
        # logging.info(points_to_transform)
        transformed_xy = tps.transform(points_to_transform)
        transformed_xy_list = transformed_xy.tolist()
        # logging.info("transformed_xy_list")
        # logging.info(transformed_xy_list)
        transformed_objects = [
            {"id": d.id, "latitude": xy[0], "longitude": xy[1]}
            for d, xy in zip(image.detections, transformed_xy_list)
        ]
        # logging.info("transformed_objects")
        # logging.info(transformed_objects)
        for obj in transformed_objects:
            db.detection.update(
                where={"id": obj["id"]},
                data={"latitude": obj["latitude"], "longitude": obj["longitude"]},
            )

    db.image.update(
        where={
            "id": job.id,
        },
        data={
            "detectionsProcessed": True,
        },
    )
