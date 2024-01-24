from flask import Flask, request, jsonify, send_file
import requests
import datetime
import logging
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
import base64
import tempfile
import json
from io import BytesIO
from PIL import Image
import hashlib
import redis
import pickle
from xml.dom.minidom import parseString

logging.basicConfig(level=logging.INFO)


def pretty_print_xml(xml_string):
    # logging.info(xml_string)
    try:
        dom = parseString(xml_string)
        pretty_xml = dom.toprettyxml(indent="    ")  # Use 4 spaces for indentation
        return pretty_xml
    except Exception as e:
        logging.error(f"Failed to parse XML: {e}")
        return xml_string


def image(id, db, redis):
    # Construct the URL for the image
    # timestamp = datetime.datetime.now().isoformat()
    # image_url = f"https://cctv.austinmobility.io/image/{id}.jpg?{timestamp}"
    image_url = f"https://cctv.austinmobility.io/image/{id}.jpg"

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
        redis.setex(image_url, 300, pickle.dumps((status_code, image_content)))

    if status_code != 200:
        with db.tx() as transaction:
            status = db.status.upsert(
                where={"name": "404"},
                data={"create": {"name": "404"}, "update": {}},
            )
            camera = db.camera.upsert(
                where={"coaId": id},
                data={
                    "create": {"coaId": id, "statusId": status.id},
                    "update": {"statusId": status.id},
                },
            )
        img = Image.new("RGB", (1920, 1080), color="black")
        d = ImageDraw.Draw(img)
        font_path = "/usr/share/fonts/truetype/MonaspaceNeon-Light.otf"
        font = ImageFont.truetype(font_path, 150)
        small_font = ImageFont.truetype(font_path, 24)
        d.text((10, 10), str(status_code), font=font, fill=(255, 255, 255))

        d.text(
            (10, 200),
            pretty_print_xml(str(image_content.decode())),
            font=small_font,
            fill=(255, 255, 255),
        )
        img_io = BytesIO()
        img.save(img_io, "JPEG", quality=70)
        img_io.seek(0)
        return send_file(img_io, mimetype="image/jpeg")

    image = BytesIO(image_content)

    sha256_hash = hashlib.sha256()
    sha256_hash.update(image_content)
    image_hash = sha256_hash.hexdigest()

    logging.info(f"SHA256 hash of the image: {image_hash}")

    if (
        image_hash == "58da0d53512030c5748d6ecf8337419586ab95d91e1ca2f9d6347cb8879ea960"
    ):  # unavailable
        with db.tx() as transaction:
            status = db.status.upsert(
                where={"name": "unavailable"},
                data={"create": {"name": "unavailable"}, "update": {}},
            )
            camera = db.camera.upsert(
                where={"coaId": id},
                data={
                    "create": {"coaId": id, "statusId": status.id},
                    "update": {"statusId": status.id},
                },
            )
            db.image.upsert(
                where={"hash": image_hash, "cameraId": camera.id},
                data={
                    "create": {
                        "hash": image_hash,
                        "cameraId": camera.id,
                        "statusId": status.id,
                    },
                    "update": {"statusId": status.id},
                },
            )
    else:
        with db.tx() as transaction:
            status = db.status.upsert(
                where={"name": "ok"},
                data={"create": {"name": "ok"}, "update": {}},
            )
            camera = db.camera.upsert(
                where={"coaId": id},
                data={
                    "create": {"coaId": id, "statusId": status.id},
                    "update": {"statusId": status.id},
                },
            )
            db.image.upsert(
                where={"hash": image_hash, "cameraId": camera.id},
                data={
                    "create": {
                        "hash": image_hash,
                        "cameraId": camera.id,
                        "statusId": status.id,
                    },
                    "update": {"statusId": status.id},
                },
            )

    return send_file(image, mimetype="image/jpeg")
