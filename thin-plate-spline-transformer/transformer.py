import argparse

import torch
import json
from torch_tps import ThinPlateSpline

torch.set_printoptions(precision=10)


def read_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def extract_points(data):
    cctv_points = torch.tensor(
        [[d["cctvPoint"]["x"], d["cctvPoint"]["y"]] for d in data["points"]]
    )
    map_points = torch.tensor(
        [[d["mapPoint"]["lat"], d["mapPoint"]["lng"]] for d in data["points"]]
    )
    return cctv_points, map_points


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("uuid", help="The UUID to process")
    args = parser.parse_args()

    # Read data from file
    data = read_data(f"/tmp/transformations/{args.uuid}/points.json")
    # print(data)
    # Extract points and train TPS model
    cctv_points, map_points = extract_points(data)
    # print("cctv_points", cctv_points)
    # print("map_points", map_points)
    tps = ThinPlateSpline(0.5)

    # Convert the tensors to float
    cctv_points = cctv_points.float()
    map_points = map_points.float()

    # Fit the surfaces
    tps.fit(cctv_points, map_points)

    test_points = torch.tensor([[d["x"], d["y"]] for d in data["labels"]]).float()
    transformed_xy = tps.transform(test_points)
    transformed_xy_list = transformed_xy.tolist()
    transformed_points = [{"lat": xy[0], "lng": xy[1]} for xy in transformed_xy_list]
    print(json.dumps(transformed_points, indent=2))


if __name__ == "__main__":
    main()
