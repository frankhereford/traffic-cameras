import torch
from torch_tps import ThinPlateSpline
import logging
import json

torch.set_printoptions(precision=10)

logging.basicConfig(level=logging.INFO)


def extract_points(locations):
    cctv_points = torch.tensor([[location.x, location.y] for location in locations])
    map_points = torch.tensor(
        [[location.latitude, location.longitude] for location in locations]
    )
    return cctv_points, map_points


def thin_plate_spline(id, db, redis):
    camera = db.camera.find_first(where={"coaId": id}, include={"Location": True})

    cctv_points, map_points = extract_points(camera.Location)

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

    objects_to_transform = ["person", "car"]
    points_to_transform = torch.tensor(
        [
            [(d.xMin + d.xMax) / 2, d.yMin]
            for d in image.detections
            if d.label in objects_to_transform
        ]
    ).float()
    # test_points = torch.tensor([[d["x"], d["y"]] for d in data["labels"]]).float()
    transformed_xy = tps.transform(points_to_transform)
    transformed_xy_list = transformed_xy.tolist()
    transformed_objects = [
        {"id": d.id, "latitude": xy[0], "longitude": xy[1]}
        for d, xy in zip(image.detections, transformed_xy_list)
        if d.label in objects_to_transform
    ]
    logging.info(transformed_objects)
    # logging.info(json.dumps(transformed_points, indent=2))

    # logging.info(f"cctv_points: {cctv_points}")
    # logging.info(f"cctv_points: {map_points}")

    return "hi"
