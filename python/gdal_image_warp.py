import argparse
import redis
from prisma import Prisma
import os


def generate_gdal_commands(control_points, image_path):
    gdal_translate_command = (
        "/Applications/QGIS.app/Contents/MacOS/bin/gdal_translate -of GTiff "
    )
    for point in control_points:
        gdal_translate_command += "-gcp {} {} {} {} ".format(
            point[0][0], point[0][1], point[1][0], point[1][1]
        )
    gdal_translate_command += "{} intermediate.tif".format(image_path)

    gdalwarp_command = "/Applications/QGIS.app/Contents/MacOS/bin/gdalwarp -r near -tps -co COMPRESS= -dstalpha -t_srs EPSG:4326 intermediate.tif {}_modified.tif".format(
        os.path.splitext(os.path.basename(image_path))[0]
    )

    return gdal_translate_command, gdalwarp_command


def main(db, redis, camera_id, image_path):
    camera = db.camera.find_first(
        where={"coaId": camera_id}, include={"Location": True}
    )

    control_points = []
    for location in camera.Location:
        control_points.append(
            (
                (location.x, location.y),
                (location.longitude, location.latitude),
            )
        )

    print(control_points)
    gdal_translate_command, gdalwarp_command = generate_gdal_commands(
        control_points, image_path
    )
    os.system(gdal_translate_command)
    os.system(gdalwarp_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--camera", type=int, default=406, help="Image ID")
    parser.add_argument(
        "-i", "--image", type=str, required=True, help="Image file path"
    )

    args = parser.parse_args()

    redis = redis.Redis(host="localhost", port=6379, db=0)

    db = Prisma()
    db.connect()

    main(db, redis, args.camera, args.image)
