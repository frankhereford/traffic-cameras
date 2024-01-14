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
        [[d["cctvPoint"]["x"], d["cctvPoint"]["y"]] for d in data]
    )
    map_points = torch.tensor(
        [[d["mapPoint"]["lat"], d["mapPoint"]["lng"]] for d in data]
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

    test_point = torch.tensor([[1018, 426]]).float()
    transformed_xy = tps.transform(test_point)
    print("transformed_xy", transformed_xy)


if __name__ == "__main__":
    main()