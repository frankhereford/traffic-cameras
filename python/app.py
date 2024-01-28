import argparse
from flask import Flask
import redis

from routes.status import status
from routes.vision import vision
from routes.image import image

app = Flask(__name__)


@app.route("/")
def status_route():
    return status()


@app.route("/image/<int:id>", methods=["GET"])
def image_route(id):
    return image(id, db, redis)


def main(mode):
    if mode == "flask":
        app.run(host="0.0.0.0", debug=True)
    elif mode == "detector":
        vision(db, redis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="flask",
        choices=["flask", "detector"],
        required=True,
    )
    args = parser.parse_args()

    hostname = "localhost" if args.mode == "detector" else "redis"
    redis = redis.Redis(host=hostname, port=6379, db=0)
    from prisma import Prisma

    db = Prisma()
    db.connect()

    main(args.mode)
