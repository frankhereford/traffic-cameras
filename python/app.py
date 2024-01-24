import argparse
from flask import Flask, request, jsonify, send_file
import redis

from routes.status import status
from routes.vision import vision_flask_response, vision_request_processor
from routes.image import image

redis = redis.Redis(host="redis", port=6379, db=0)
from prisma import Prisma

db = Prisma()
db.connect()

app = Flask(__name__)


@app.route("/")
def status_route():
    return status()


@app.route("/image/<int:id>", methods=["GET"])
def image_route(id):
    return image(id, db, redis)


@app.route("/vision", methods=["POST"])
def vision_route():
    return vision_flask_response(db, redis)


def main(mode):
    if mode == "flask":
        app.run(host="0.0.0.0", debug=True)
    elif mode == "detector":
        vision_request_processor(db, redis)


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
    main(args.mode)
