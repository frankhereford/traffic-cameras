from flask import Flask, request, jsonify, send_file
import logging
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
from io import BytesIO
import time
import torch
from torch_tps import ThinPlateSpline
import io
import base64


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

        # Calculate the width and height of the bounding box
        width = box[2] - box[0]
        height = box[3] - box[1]

        # Calculate the padding
        padding_width = width * 0.2
        padding_height = height * 0.2

        # Calculate the new bounding box coordinates, ensuring they do not exceed the image boundaries
        new_box = [
            max(0, box[0] - padding_width),
            max(0, box[1] - padding_height),
            min(image.width, box[2] + padding_width),
            min(image.height, box[3] + padding_height),
        ]

        # Crop the image using the new bounding box
        detected_object = image.crop(new_box)

        # Calculate the relative bounding box coordinates for the cropped image
        relative_box = [
            box[0] - new_box[0],
            box[1] - new_box[1],
            box[2] - new_box[0],
            box[3] - new_box[1],
        ]

        # Draw the bounding box on the cropped image
        draw = ImageDraw.Draw(detected_object)
        draw.rectangle(relative_box, outline="red", width=1)

        byte_stream = io.BytesIO()
        detected_object.save(byte_stream, format="JPEG")
        byte_stream.seek(0)
        base64_encoded = base64.b64encode(byte_stream.getvalue()).decode("utf-8")
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
                "picture": base64_encoded,
            }
        )

    # starting thin plate spline code

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

        points_to_transform = torch.tensor(
            [[(d.xMin + d.xMax) / 2, d.yMax] for d in image.detections]
        ).float()

        if points_to_transform.shape[0] > 0:
            transformed_xy = tps.transform(points_to_transform)
            transformed_xy_list = transformed_xy.tolist()
            transformed_objects = [
                {"id": d.id, "latitude": xy[0], "longitude": xy[1]}
                for d, xy in zip(image.detections, transformed_xy_list)
            ]
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
