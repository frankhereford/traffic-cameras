import io
import time
import torch
import base64
import pickle
import logging
import numpy as np
from io import BytesIO
from matplotlib.path import Path
from PIL import Image, ImageDraw
from torch_tps import ThinPlateSpline
from scipy.spatial import ConvexHull
from transformers import DetrImageProcessor, DetrForObjectDetection
import requests
from flask import send_file
import matplotlib.pyplot as plt
import json

# https://xkcd.com/353/

logging.basicConfig(level=logging.INFO)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-101", revision="no_timm"
)
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-101", revision="no_timm"
)


def is_point_in_hull(hull, point):
    hull_path = Path(hull.points[hull.vertices])
    return hull_path.contains_point(point)


def check_point_in_camera_location(camera, point):
    locations = camera.Location
    points = np.array([[location.x, location.y] for location in locations])

    if len(points) >= 3 and len(points.shape) > 1 and points.shape[1] >= 2:
        hull = ConvexHull(points)
        return is_point_in_hull(hull, point)
    else:
        print(
            "Cannot create a convex hull because there are not enough points or the points do not have at least two dimensions."
        )
        return False


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

    camera = db.camera.find_first(
        where={"id": job.camera.id}, include={"Location": True}
    )

    # print(camera.Location)

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

        # Calculate the center point of the bounding box
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        point = (center_x, center_y)

        is_in_hull = check_point_in_camera_location(camera, point)
        logging.info(
            f"The point {point} is {'inside' if is_in_hull else 'outside'} the convex hull."
        )

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
                "isInsideConvexHull": is_in_hull,
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


def transformedImage(id, db, redis):
    image_url = f"https://cctv.austinmobility.io/image/{id}.jpg"
    image_key = f"requests:{image_url[8:]}"

    response = redis.get(image_url)

    # Try fetching the image content and status code from the cache
    cached_response = redis.get(image_url)
    if cached_response:
        logging.info("Image found in cache")
        status_code, image_content = pickle.loads(cached_response)
    else:
        logging.info(f"Image not found in cache, downloading from {image_url}")
        response = requests.get(image_url)
        status_code, image_content = response.status_code, response.content

        # Cache the status code and image content
        redis.setex(image_key, 300, pickle.dumps((status_code, image_content)))

    camera = db.camera.find_first(where={"coaId": id}, include={"Location": True})

    cctv_points, map_points = extract_points(camera.Location)

    if len(cctv_points) >= 3:
        tps = ThinPlateSpline(0.5)

        cctv_points = cctv_points.float()
        map_points = map_points.float()

        logging.info(f"cctv_points: {cctv_points}")
        logging.info(f"map_points: {map_points}")

        ### start normalization
        if True:
            tps = ThinPlateSpline(0.5)

            # Fit the surfaces
            tps.fit(cctv_points, map_points)

            points_to_transform = torch.tensor(
                [[0, 0], [1920, 0], [0, 1080], [1920, 1080]]
            ).float()

            if points_to_transform.shape[0] > 0:
                transformed_xy = tps.transform(points_to_transform)
                transformed_xy_list = transformed_xy.tolist()
                logging.info(f"transformed_xy_list: {transformed_xy_list}")

                geojson = {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "geometry": {
                                "type": "Point",
                                "coordinates": [
                                    point[1],
                                    point[0],
                                ],  # Flip latitude and longitude
                            },
                            "properties": {},
                        }
                        for point in transformed_xy_list
                    ],
                }

                # Convert the GeoJSON object to a string
                geojson_str = json.dumps(geojson)
                logging.info(f"geojson_str: \n\n{geojson_str}\n\n")

        ### end normalization

        # Fit the surfaces
        tps.fit(cctv_points, map_points)

        width = 1920
        height = 1080

        # Create the 2d meshgrid of indices for output image
        i = torch.arange(height, dtype=torch.float32)
        j = torch.arange(width, dtype=torch.float32)

        ii, jj = torch.meshgrid(i, j, indexing="ij")
        output_indices = torch.cat(
            (ii[..., None], jj[..., None]), dim=-1
        )  # Shape (H, W, 2)
        logging.info(f"output_indices.shape: {output_indices.shape}")

        # Transform it into the input indices
        input_indices = tps.transform(output_indices.reshape(-1, 2)).reshape(
            height, width, 2
        )

        logging.info(f"input_indices.shape: {input_indices.shape}")

        size = torch.tensor((height, width))

        logging.info(f"size: {size}")

        image = Image.open(BytesIO(image_content))

        # Interpolate the resulting image
        grid = 2 * input_indices / size - 1  # Into [-1, 1]
        grid = torch.flip(
            grid, (-1,)
        )  # Grid sample works with x,y coordinates, not i, j
        torch_image = torch.tensor(np.array(image), dtype=torch.float32).permute(
            2, 0, 1
        )[None, ...]
        warped = torch.nn.functional.grid_sample(
            torch_image, grid[None, ...], align_corners=False
        )[0]

        # Convert the Tensor to a PIL image
        warped_image = Image.fromarray(
            warped.permute(1, 2, 0).to(torch.uint8).byte().cpu().numpy()
        )

        # Create a BytesIO object and save the image to it
        byte_io = io.BytesIO()
        warped_image.save(byte_io, "JPEG")

        # Go back to the beginning of the BytesIO object
        byte_io.seek(0)

        return send_file(byte_io, mimetype="image/jpeg")

    return send_file(BytesIO(image_content), mimetype="image/jpeg")
