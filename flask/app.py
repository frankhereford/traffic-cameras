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


# from queries import *

redis = redis.Redis(host="redis", port=6379, db=0)
from prisma import Prisma

db = Prisma()
db.connect()

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-101", revision="no_timm"
)
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-101", revision="no_timm"
)


@app.route("/")
def status():
    # Get the current time
    current_time = datetime.datetime.now().isoformat()

    # Fetch weather data from NOAA API for Austin, TX (using station KAUS)
    response = requests.get("https://api.weather.gov/stations/KAUS/observations/latest")
    weather_data = response.json()

    # Get weather description and corresponding emoji
    weather_description = weather_data["properties"]["textDescription"]

    # Return the current time and brief weather description
    return jsonify(
        {
            "status": "nominal",
            "description": "traffic-cameras backend python processor",
            "current_time": current_time,
            "current_weather_in_austin": weather_description,
        }
    )


@app.route("/image/<int:id>", methods=["GET"])
def image(id):
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

    return send_file(image, mimetype="image/jpeg")


# remember to change this, think about this, make it a useQuery
@app.route("/vision", methods=["POST"])
def vision():
    logging.info("")
    data = request.get_json()
    base64_image = data.get("image")
    logging.info(f"Received image data: {base64_image[:100]}...")

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Decode the base64 image
        img_data = base64.b64decode(base64_image)

        logging.info("base64_image")
        logging.info(base64_image[:50])

        # Remove the prefix if present
        if base64_image.startswith("data:image/jpeg;base64,"):
            base64_image = base64_image.replace("data:image/jpeg;base64,", "")

        decoded_data = base64.b64decode(base64_image)
        byte_stream = BytesIO(decoded_data)
        image = Image.open(byte_stream)

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9
        )[0]

        detected_objects = []

        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box = [round(i, 2) for i in box.tolist()]
            logging.info(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
            )

            # Add the detected object to the list
            detected_objects.append(
                {
                    "label": model.config.id2label[label.item()],
                    "confidence": round(score.item(), 3),
                    "location": box,
                }
            )

        return jsonify({"detected_objects": detected_objects})


def pretty_print_xml(xml_string):
    # logging.info(xml_string)
    try:
        dom = parseString(xml_string)
        pretty_xml = dom.toprettyxml(indent="    ")  # Use 4 spaces for indentation
        return pretty_xml
    except Exception as e:
        logging.error(f"Failed to parse XML: {e}")
        return xml_string


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
