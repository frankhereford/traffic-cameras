import argparse
import redis
from prisma import Prisma


def main(db, redis, camera_id):
    camera = db.camera.find_first(where={"coaId": camera_id})
    print(camera)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--camera", type=int, default=406, help="Image ID")

    args = parser.parse_args()

    redis = redis.Redis(host="localhost", port=6379, db=0)

    db = Prisma()
    db.connect()

    main(db, redis, args.camera)
