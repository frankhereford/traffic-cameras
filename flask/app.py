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
import psycopg2


logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-101", revision="no_timm"
)
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-101", revision="no_timm"
)


params = {
    "host": "postgis",
    "user": "postgres",
    "password": "cameras",
    "dbname": "cameras",
}

conn = psycopg2.connect(**params)


@app.route("/")
def status():
    # Weather description to emoji mapping

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


from PIL import Image, ImageDraw, ImageFont


@app.route("/image/<int:id>", methods=["GET"])
def image(id):
    # Construct the URL for the image
    image_url = f"https://cctv.austinmobility.io/image/{id}.jpg"

    # Download the image
    response = requests.get(image_url)

    # Check if the request was successful
    if response.status_code != 200:
        # Create a new black image
        img = Image.new("RGB", (1280, 1920), color="black")

        # Create a draw object
        d = ImageDraw.Draw(img)

        # Define the font for the text
        fnt = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 150
        )

        # Add the text to the image
        d.text((10, 10), "404", font=fnt, fill=(255, 255, 255))

        # Save the image to a BytesIO object
        img_io = BytesIO()
        img.save(img_io, "JPEG", quality=70)
        img_io.seek(0)

        # Return the image
        return send_file(img_io, mimetype="image/jpeg")

    # Create a BytesIO object from the response content
    image = BytesIO(response.content)

    # Return the image as a response
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
