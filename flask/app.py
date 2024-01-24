from flask import Flask, request, jsonify, send_file
import redis


from routes.status import status
from routes.vision import vision
from routes.image import image


# from queries import *

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


# remember to change this, think about this, make it a useQuery
@app.route("/vision", methods=["POST"])
def vision_route():
    return vision(db, redis)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
