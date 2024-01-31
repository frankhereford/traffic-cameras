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
import torch.nn.functional as F

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


def create_geojson_of_image_borders_in_map_space(transformed_points):
    # Convert the tensor to a list of lists and reverse each point
    points = [list(reversed(point)) for point in transformed_points.tolist()]

    # Ensure the first and last points are the same
    points.append(points[0])

    # Create a GeoJSON FeatureCollection
    geojson = {
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
        geojson["features"].append(
            {
                "type": "Feature",
                "properties": {},
                "geometry": {"type": "Point", "coordinates": point},
            }
        )

    # Convert the GeoJSON object to a string
    geojson_str = json.dumps(geojson, indent=4)

    return geojson_str


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

    # logging.info(
    #     f"Original CCTV points range: X[min,max]={cctv_points[:, 0].min(), cctv_points[:, 0].max()}, Y[min,max]={cctv_points[:, 1].min(), cctv_points[:, 1].max()}"
    # )
    logging.info(
        f"Original Map points range: Lat[min,max]={map_points[:, 0].min(), map_points[:, 0].max()}, Long[min,max]={map_points[:, 1].min(), map_points[:, 1].max()}"
    )

    # logging.info(f"cctv_points: {cctv_points}")

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

    flipped_cctv_points = torch.flip(cctv_points, [1])
    # logging.info(f"cctv_points: {cctv_points}")
    # logging.info(f"flipped_cctv_points: {flipped_cctv_points}")

    # flip the CCTV points because they are X,Y and the map points are latitude, longitude which is Y,X
    cctv_points = flipped_cctv_points

    # logging.info(f"cctv_points: {cctv_points}")
    # logging.info(f"map_points: {map_points}")

    geojson = create_geojson_of_image_borders_in_map_space(
        image_extents_as_geographic_coordinates
    )

    image_width = 1920
    image_height = 1080

    # Normalizing cctv_points
    normalized_cctv_points = cctv_points / torch.tensor([image_height, image_width])

    # Determine min and max for latitude and longitude
    lat_min, lat_max = map_points[:, 0].min(), map_points[:, 0].max()
    long_min, long_max = map_points[:, 1].min(), map_points[:, 1].max()

    # Normalize map_points
    normalized_map_points = torch.empty_like(map_points)
    normalized_map_points[:, 0] = (map_points[:, 0] - lat_min) / (
        lat_max - lat_min
    )  # Normalize latitude
    normalized_map_points[:, 1] = (map_points[:, 1] - long_min) / (
        long_max - long_min
    )  # Normalize longitude

    # logging.info(
    #     f"Map points normalization: Lat Min={lat_min}, Lat Max={lat_max}, Long Min={long_min}, Long Max={long_max}"
    # )
    # logging.info(
    #     f"Normalized map points (extended sample): {normalized_map_points[:10]}"
    # )  # Shows first 10 points

    # logging.info(
    #     f"Normalized CCTV points post-adjustment range: X[min,max]={normalized_cctv_points[:, 0].min(), normalized_cctv_points[:, 0].max()}, Y[min,max]={normalized_cctv_points[:, 1].min(), normalized_cctv_points[:, 1].max()}"
    # )

    # logging.info(f"Normalized CCTV points (sample): {normalized_cctv_points[:5]}")
    # logging.info(f"Normalized Map points (sample): {normalized_map_points[:5]}")

    tps = ThinPlateSpline(0.5)

    # Fit the surfaces
    tps.fit(normalized_cctv_points, normalized_map_points)
    logging.info(f"TPS fitting completed")

    # Extracted corner coordinates from TPS
    corner_coords = image_extents_as_geographic_coordinates

    logging.info(f"corner_coords: {corner_coords}")

    # Combine corner_coords and map_points to find overall min and max
    all_latitudes = torch.cat((corner_coords[:, 0], map_points[:, 0]))
    all_longitudes = torch.cat((corner_coords[:, 1], map_points[:, 1]))

    overall_lat_min, overall_lat_max = all_latitudes.min(), all_latitudes.max()
    overall_lon_min, overall_lon_max = all_longitudes.min(), all_longitudes.max()

    # Normalize corner_coords
    normalized_corner_coords = torch.empty_like(corner_coords)
    normalized_corner_coords[:, 0] = (corner_coords[:, 0] - overall_lat_min) / (
        overall_lat_max - overall_lat_min
    )
    normalized_corner_coords[:, 1] = (corner_coords[:, 1] - overall_lon_min) / (
        overall_lon_max - overall_lon_min
    )

    logging.info(f"Normalized corner_coords: {normalized_corner_coords}")
    logging.info(f"Normalized Map points (sample): {normalized_map_points[:5]}")

    # Generate latitude and longitude values using normalized_corner_coords
    lat_min, lat_max = (
        normalized_corner_coords[:, 0].min(),
        normalized_corner_coords[:, 0].max(),
    )
    lon_min, lon_max = (
        normalized_corner_coords[:, 1].min(),
        normalized_corner_coords[:, 1].max(),
    )

    # Number of points in each dimension
    num_points_lat = 1000  # Adjust as needed
    num_points_lon = 1000  # Adjust as needed

    # Generate latitude and longitude values
    latitudes = torch.linspace(lat_min, lat_max, num_points_lat)
    longitudes = torch.linspace(lon_min, lon_max, num_points_lon)

    logging.info(f"Generated latitudes range: {latitudes.min()}, {latitudes.max()}")
    logging.info(f"Generated longitudes range: {longitudes.min()}, {longitudes.max()}")

    # Create a meshgrid
    lat_grid, lon_grid = torch.meshgrid(latitudes, longitudes, indexing="ij")

    # Flatten the grid
    geo_grid = torch.stack([lat_grid.flatten(), lon_grid.flatten()], dim=1)

    logging.info(f"Geo grid shape: {geo_grid.shape}")
    logging.info(f"Geo grid sample points: {geo_grid[:5]}")

    transformed_geo_grid = tps.transform(geo_grid)
    logging.info(f"Transformed geo grid shape: {transformed_geo_grid.shape}")
    logging.info(f"Transformed geo grid sample points: {transformed_geo_grid[:5]}")

    # Normalize the transformed grid to [-1, 1]
    max_height, max_width = (
        image_height,
        image_width,
    )

    transformed_geo_grid_normalized = (
        transformed_geo_grid * 2 / torch.tensor([max_width, max_height])
    ) - 1

    transformed_geo_grid_np = transformed_geo_grid.detach().numpy()

    plt.figure(figsize=(8, 8))  # Set the figure size as required
    plt.scatter(transformed_geo_grid_np[:, 0], transformed_geo_grid_np[:, 1], s=1)
    plt.title("Scatter plot of transformed geo grid points")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.xlim(-2, 2)  # Set limits to see all points
    plt.ylim(-2, 2)
    plt.grid(True)

    # Save the figure
    plt.savefig("matplotlib.jpg", dpi=300)

    # Close the plot to free up memory
    plt.close()

    transformed_geo_grid_normalized = torch.flip(
        transformed_geo_grid_normalized, [1]
    )  # Flip to match grid_sample's x, y format

    logging.info(
        f"Transformed geo grid range before clamping: X[min,max]={transformed_geo_grid[:, 0].min(), transformed_geo_grid[:, 0].max()}, Y[min,max]={transformed_geo_grid[:, 1].min(), transformed_geo_grid[:, 1].max()}"
    )

    # Clamp the normalized grid values to be between -1 and 1
    transformed_geo_grid_normalized.clamp_(-1, 1)
    logging.info(
        f"Clamped transformed_geo_grid_normalized: {transformed_geo_grid_normalized}"
    )

    logging.info(f"transformed_geo_grid_normalized: {transformed_geo_grid_normalized}")

    # Load and prepare the image
    image = Image.open(BytesIO(image_content))
    image_tensor = (
        torch.tensor(np.array(image), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    )

    # Assuming you are using PyTorch's grid_sample or similar function
    warped_image_tensor = F.grid_sample(
        image_tensor,  # your original image tensor
        transformed_geo_grid_normalized.view(1, num_points_lat, num_points_lon, 2),
        mode="bilinear",  # or 'nearest', etc.
        padding_mode="zeros",
        align_corners=True,
    )

    # Log some sample values from the warped image
    logging.info(
        f"Warped image tensor sample values: {warped_image_tensor[0, :, :5, :5]}"
    )

    # Convert back to PIL image for viewing
    warped_image = (
        warped_image_tensor.squeeze(0).permute(1, 2, 0) * 255
    ).byte()  # Rescaling back to [0, 255]
    warped_image_pil = Image.fromarray(warped_image.numpy())
    warped_image_pil.save("warped_image.jpg")

    # logging.info(f"status_code: {status_code}")
    # logging.info(f"image_content length: {len(image_content)}")
    # logging.info(f"camera: {camera}")
    # logging.info(f"cctv_points (before flip): {cctv_points}")
    # logging.info(f"map_points: {map_points}")
    # logging.info(f"normalized_cctv_points: {normalized_cctv_points}")
    # logging.info(f"normalized_map_points: {normalized_map_points}")
    # logging.info(f"corner_coords: {corner_coords}")
    # logging.info(
    #     f"geo_grid (sample points): {geo_grid[:5]}"
    # )  # Sample first 5 points for brevity
    # logging.info(f"transformed_geo_grid (sample points): {transformed_geo_grid[:5]}")
    # logging.info(
    #     f"transformed_geo_grid_normalized (sample points): {transformed_geo_grid_normalized[:5]}"
    # )
    # logging.info(f"warped_image_tensor shape: {warped_image_tensor.shape}")

    # Create a BytesIO object and save the image to it
    byte_io = io.BytesIO()
    warped_image_pil.save(byte_io, "JPEG")

    # Go back to the beginning of the BytesIO object
    byte_io.seek(0)
    return send_file(byte_io, mimetype="image/jpeg")

    value = f"<pre>{geojson}</pre>"
    return value
