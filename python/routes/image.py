from flask import send_file
import requests
import logging
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
from PIL import Image
import hashlib
import pickle
from xml.dom.minidom import parseString
from io import BytesIO
import boto3
import os
from datetime import datetime

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
    image_url = f"https://cctv.austinmobility.io/image/{id}.jpg"
    image_key = f"requests:{image_url[8:]}"

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
        redis.setex(image_key, 300, pickle.dumps((status_code, image_content)))

    # Try to detect image dimensions if possible
    detected_width, detected_height = 1920, 1080  # Default fallback
    if status_code == 200:
        try:
            with Image.open(BytesIO(image_content)) as img_dim:
                detected_width, detected_height = img_dim.size
        except Exception as e:
            logging.warning(f"Could not detect image dimensions: {e}")

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
        # Use detected dimensions if available, otherwise fallback
        img = Image.new("RGB", (detected_width, detected_height), color="black")
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
            image_record = db.image.upsert(
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

        image.seek(0)
        if not redis.exists(f"images:{image_hash}"):
            serialized_image = pickle.dumps(image)
            redis.setex(f"images:{image_hash}", 86400 * 1, serialized_image)

        image.seek(0)
        pillow_image = Image.open(image)
        d = ImageDraw.Draw(pillow_image)
        font_path = "/usr/share/fonts/truetype/MonaspaceNeon-Light.otf"
        font = ImageFont.truetype(font_path, 24)

        # Draw the black stroke
        for i in range(-3, 4):
            for j in range(-3, 4):
                d.text(
                    (1720 + i, 10 + j),
                    f"{image_hash[:8]}/{id}",
                    font=font,
                    fill=(0, 0, 0),  # Black color
                )

        # Draw the white text
        d.text(
            (1720, 10),
            f"{image_hash[:8]}/{id}",
            font=font,
            fill=(255, 255, 255),  # White color
        )

        img_io = BytesIO()
        pillow_image.save(img_io, "JPEG", quality=100)
        image = img_io

        # --- S3 Upload as PNG using createdAt ---
        try:
            image_type = 'jpg'

            # Prepare image in memory based on image_type
            if image_type == 'png':
                img_io_s3 = BytesIO()
                pillow_image.save(img_io_s3, "PNG")
                content_type = "image/png"
                ext = "png"
            elif image_type == 'jpg':
                # Use the original image bytes received from the source
                img_io_s3 = BytesIO(image_content)
                content_type = "image/jpeg"
                ext = "jpg"
            else:
                raise ValueError(f"Unsupported image_type: {image_type}")
            img_io_s3.seek(0)

            # S3 key: cameras/<id>/<hash>.<ext> (no timestamp)
            s3_key = f"cameras/{id}/{image_hash}.{ext}"
            logging.info(f"Uploading {ext.upper()} to S3 with key: {s3_key}")

            s3 = boto3.client(
                "s3",
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
            )

            # Check if file exists
            try:
                s3.head_object(Bucket="atx-traffic-cameras", Key=s3_key)
                logging.info(f"File already exists in S3: {s3_key}, skipping upload.")
            except s3.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    # File does not exist, upload
                    s3.upload_fileobj(
                        img_io_s3,
                        "atx-traffic-cameras",
                        s3_key,
                        ExtraArgs={"ContentType": content_type},
                    )
                    logging.info(f"Uploaded {ext.upper()} to s3://atx-traffic-cameras/{s3_key}")
                else:
                    logging.error(f"Error checking S3 for {s3_key}: {e}")

        except Exception as e:
            logging.error(f"Failed to upload PNG to S3: {e}")

    image.seek(0)  # Rewind to the start of the stream

    return send_file(image, mimetype="image/jpeg")
