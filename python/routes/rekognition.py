import time
import logging
import boto3
import io
import pickle
from PIL import Image, ImageDraw
import base64
import numpy as np
from matplotlib.path import Path
from scipy.spatial import ConvexHull


logging.basicConfig(level=logging.INFO)


def throttle(func):
    throttle._last_called = None

    def wrapper(*args, **kwargs):
        if throttle._last_called is not None:
            time_since_last_call = time.time() - throttle._last_called
            sleep_time = int(round(max(0, 1 - time_since_last_call), 0))
            logging.info(f"sleep time: {sleep_time} ")
            time.sleep(sleep_time)
        throttle._last_called = time.time()
        return func(*args, **kwargs)

    return wrapper

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
        logging.info(
            "Cannot create a convex hull because there are not enough points or the points do not have at least two dimensions."
        )
        return False

def rekognition(db, redis):
    while True:
        process_one_image(db, redis)

@throttle
def process_one_image(db, redis):
    logging.info("rekognition processing one image")
    job = db.image.find_first(
        where={
            "detectionsProcessed": False,
        },
        include={"camera": True},
    )
    if job is None:
        return

    logging.info(job.hash)

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

    # 1. Convert PIL Image to raw bytes
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()

    # 2. Call Rekognition
    client = boto3.client('rekognition', region_name='us-east-1')  # adjust region
    response = client.detect_labels(
        Image={'Bytes': img_bytes},
        MaxLabels=100,
        MinConfidence=75
    )

    # 3. Inspect results and store detections
    img_width, img_height = image.size
    for lbl in response['Labels']:
        logging.info(f"{lbl['Name']} ({lbl['Confidence']:.1f}%)")
        for inst in lbl.get('Instances', []):
            bbox = inst['BoundingBox']
            # Rekognition bounding box is in relative coordinates (0-1)
            xMin = bbox['Left'] * img_width
            yMin = bbox['Top'] * img_height
            xMax = xMin + bbox['Width'] * img_width
            yMax = yMin + bbox['Height'] * img_height

            # Calculate the width and height of the bounding box
            width = xMax - xMin
            height = yMax - yMin

            # Calculate the padding (20%)
            padding_width = width * 0.2
            padding_height = height * 0.2

            # Calculate the new bounding box coordinates, ensuring they do not exceed the image boundaries
            new_box = [
                max(0, xMin - padding_width),
                max(0, yMin - padding_height),
                min(img_width, xMax + padding_width),
                min(img_height, yMax + padding_height),
            ]

            # Crop the image using the new bounding box
            detected_object = image.crop(new_box)

            # Calculate the relative bounding box coordinates for the cropped image
            relative_box = [
                xMin - new_box[0],
                yMin - new_box[1],
                xMax - new_box[0],
                yMax - new_box[1],
            ]

            # Draw the bounding box on the cropped image
            draw = ImageDraw.Draw(detected_object)
            draw.rectangle(relative_box, outline="red", width=1)

            byte_stream = io.BytesIO()
            detected_object.save(byte_stream, format="JPEG")
            byte_stream.seek(0)
            base64_encoded = base64.b64encode(byte_stream.getvalue()).decode("utf-8")

            # Calculate the center point of the bounding box
            center_x = (xMin + xMax) / 2
            center_y = (yMin + yMax) / 2
            point = (center_x, center_y)

            is_in_hull = check_point_in_camera_location(camera, point)
            logging.info(
                f"The point {point} is {'inside' if is_in_hull else 'outside'} the convex hull."
            )

            db.detection.create(
                data={
                    "label": lbl['Name'],
                    "confidence": round(inst['Confidence'], 3),
                    "xMin": xMin,
                    "yMin": yMin,
                    "xMax": xMax,
                    "yMax": yMax,
                    "imageId": job.id,
                    "picture": base64_encoded,
                    "isInsideConvexHull": is_in_hull,
                }
            )

    db.image.update(
        where={
            "id": job.id,
        },
        data={
            "detectionsProcessed": True,
        },
    )
