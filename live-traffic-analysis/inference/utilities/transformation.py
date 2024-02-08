import csv
import torch


def read_points_file(file_path):
    image_coordinates = []
    map_coordinates = []

    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            if row[0] == "mapX":  # Skip if it's the header row
                continue
            mapX, mapY, sourceX, sourceY = row[:4]
            sourceY = -1 * float(sourceY)  # Invert Y axis
            map_coordinates.append((float(mapX), float(mapY)))
            image_coordinates.append((float(sourceX), float(sourceY)))

    # Convert lists to PyTorch tensors
    image_coordinates = torch.tensor(image_coordinates)
    map_coordinates = torch.tensor(map_coordinates)

    return {"image_coordinates": image_coordinates, "map_coordinates": map_coordinates}
