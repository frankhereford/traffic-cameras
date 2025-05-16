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
import json
import tempfile
import os
import subprocess
import re
import multiprocessing

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
        logging.info(
            "Cannot create a convex hull because there are not enough points or the points do not have at least two dimensions."
        )
        return False


def extract_points(locations):
    cctv_points = torch.tensor([[location.x, location.y] for location in locations])
    map_points = torch.tensor(
        [[location.latitude, location.longitude] for location in locations]
    )
    return cctv_points, map_points


def vision(db, redis, num_workers=2):
    """
    Process images using a configurable number of worker processes.
    """
    with multiprocessing.Pool(processes=num_workers) as pool:
        while True:
            # Fetch up to num_workers jobs at once
            jobs = db.image.find_many(
                where={"detectionsProcessed": False},
                include={"camera": True},
                take=num_workers,
            )
            if not jobs:
                time.sleep(1)
                continue

            # Prepare arguments for each job
            args = [(job, db, redis) for job in jobs]
            pool.starmap(process_one_image, args)


def process_one_image(job, db, redis):
    # job is now passed in directly
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


def create_geojson_of_image_borders_in_map_space(transformed_points):
    # Convert the tensor to a list of lists and reverse each point
    points = [list(reversed(point)) for point in transformed_points.tolist()]

    # Ensure the first and last points are the same
    points.append(points[0])

    # Create a GeoJSON FeatureCollection
    geojson_object = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [points]},
            }
        ],
    }

    # Add each point as a separate feature (excluding the last point which is a duplicate of the first)
    for point in points[:-1]:
        geojson_object["features"].append(
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Point", "coordinates": point},
            }
        )

    # Convert the GeoJSON object to a string
    geojson_str = json.dumps(geojson_object, indent=4)

    return (geojson_str, geojson_object)


def create_transform_and_process_tensor_of_input_points(
    image_registration_points, map_registration_points, list_to_process
):
    # Ensure image_registration_points and map_registration_points are float type
    image_registration_points = image_registration_points.float()
    map_registration_points = map_registration_points.float()
    list_to_process = list_to_process.float()

    # Create the thin plate spline object
    tps = ThinPlateSpline(0.5)
    tps.fit(image_registration_points, map_registration_points)
    transformed_points = tps.transform(list_to_process)

    return transformed_points


def generate_gdal_commands(control_points, image_path, temp_dir):
    gdal_translate_command = "/usr/bin/gdal_translate -of GTiff "
    for point in control_points:
        gdal_translate_command += "-gcp {} {} {} {} ".format(
            point[0][0], point[0][1], point[1][0], point[1][1]
        )
    gdal_translate_command += f"{image_path} "
    gdal_translate_command += f"{temp_dir}/intermediate.tif"

    gdalwarp_command = f"/usr/bin/gdalwarp -r near -tps  -dstalpha -t_srs EPSG:4326 {temp_dir}/intermediate.tif {temp_dir}/modified.tif"
    return gdal_translate_command, gdalwarp_command


def get_image_extent(gdalinfo_command):
    process = subprocess.Popen(gdalinfo_command, stdout=subprocess.PIPE, shell=True)
    output, _ = process.communicate()
    output = output.decode("utf-8")
    # logging.info(f"gdalinfo output: {output}")

    corners = re.findall(
        r"Upper Left\s+\(\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\).*\n.*\n.*\nLower Right\s+\(\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)\)",
        output,
    )

    if corners:
        min_lon, max_lat, max_lon, min_lat = map(float, corners[0])
        return min_lon, min_lat, max_lon, max_lat

    return None


def transformedImage(id, db, redis):
    image_url = f"https://cctv.austinmobility.io/image/{id}.jpg"
    image_key = f"requests:{image_url[8:]}"

    response = redis.get(image_key)

    # Try fetching the image content and status code from the cache
    cached_response = redis.get(image_url)
    if cached_response:
        logging.info("Image found in cache")
        status_code, image_content = pickle.loads(cached_response)
    else:
        logging.info(f"Image not found in cache, downloading from {image_url}")
        response = requests.get(image_url)
        status_code, image_content = response.status_code, response.content

        # # Cache the status code and image content
        redis.setex(image_key, 300, pickle.dumps((status_code, image_content)))

        with tempfile.TemporaryDirectory() as temp_dir:
            logging.info(f"Temporary directory created at {temp_dir}")

            image = Image.open(BytesIO(image_content))
            image_path = f"{temp_dir}/{id}.jpg"
            image.save(image_path)

            camera = db.camera.find_first(
                where={"coaId": id}, include={"Location": True}
            )

            control_points = []
            for location in camera.Location:
                control_points.append(
                    (
                        (location.x, location.y),
                        (location.longitude, location.latitude),
                    )
                )

            logging.info(control_points)
            gdal_translate_command, gdalwarp_command = generate_gdal_commands(
                control_points, image_path, temp_dir
            )

            logging.info(gdal_translate_command)
            os.system(gdal_translate_command)

            logging.info(gdalwarp_command)
            os.system(gdalwarp_command)

            gdalinfo_command = f"/usr/bin/gdalinfo {temp_dir}/modified.tif"
            extent = get_image_extent(gdalinfo_command)
            logging.info(f"extent: {extent}")
            output_geotiff = f"{temp_dir}/modified.tif"

            image = Image.open(output_geotiff)

            cctv_points, map_points = extract_points(camera.Location)

            cctv_points.float()
            map_points.float()

            corners_in_image_space = [(0, 0), (0, 1080), (1920, 1080), (1920, 0)]
            list_to_process = torch.tensor(corners_in_image_space)

            image_extents_as_geographic_coordinates = (
                create_transform_and_process_tensor_of_input_points(
                    cctv_points, map_points, list_to_process
                )
            )

            # logging.info(
            #     f"image_extents_as_geographic_coordinates: {image_extents_as_geographic_coordinates}"
            # )

            (
                geojson_str,
                geojson_object,
            ) = create_geojson_of_image_borders_in_map_space(
                image_extents_as_geographic_coordinates
            )

            # Convert the Pillow image to a base64 string
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            payload = {
                "extent": extent,
                "geojson": geojson_object,
                "image": img_str,
            }

            return json.dumps(payload, indent=4)

            value = f"<pre>{geojson_str}</pre>"
            value += f"<pre>{gdal_translate_command}</pre>"
            value += f"<pre>{gdalwarp_command}</pre>"
            return value
