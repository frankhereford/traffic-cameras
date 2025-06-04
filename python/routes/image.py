from flask import send_file
import requests
import logging
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import hashlib
import pickle
from xml.dom.minidom import parseString
import boto3
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)

# Helper functions for resolution-independent rendering
def _get_scaled_font(font_path, base_font_size_at_1080p, current_image_height):
    """Calculates font size scaled relative to a 1080p reference height."""
    if current_image_height <= 0: # Avoid division by zero or negative heights
        current_image_height = 1080 # Default to base height if invalid
    scale_factor = current_image_height / 1080.0
    scaled_font_size = max(1, round(base_font_size_at_1080p * scale_factor))
    try:
        return ImageFont.truetype(font_path, scaled_font_size)
    except IOError:
        logging.warning(f"Font not found at {font_path}. Using default font.")
        # Attempt to load a basic default font if custom font fails
        try:
            return ImageFont.load_default(size=scaled_font_size) # Pillow 9.5.0+
        except AttributeError: # older Pillow or if size arg not supported
             return ImageFont.load_default()


def _get_scaled_coords(x_on_1920_canvas, y_on_1080_canvas, current_image_width, current_image_height):
    """Calculates coordinates scaled from a 1920x1080 reference canvas."""
    if current_image_width <= 0 or current_image_height <= 0:
        logging.warning("Invalid image dimensions for coordinate scaling. Defaulting to (0,0).")
        return (0,0)
        
    x_percent = x_on_1920_canvas / 1920.0
    y_percent = y_on_1080_canvas / 1080.0
    scaled_x = round(x_percent * current_image_width)
    scaled_y = round(y_percent * current_image_height)
    return int(scaled_x), int(scaled_y)

def _get_scaled_offset(base_offset_at_1080p_height, current_image_height):
    """Calculates an offset value scaled relative to a 1080p reference height."""
    if current_image_height <= 0:
        current_image_height = 1080 # Default to base height
    scale_factor = current_image_height / 1080.0
    if base_offset_at_1080p_height == 0:
        return 0
    return max(1, round(base_offset_at_1080p_height * scale_factor))


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

    # Try fetching the image content and status code from the cache
    cached_response_data = redis.get(image_key)
    if cached_response_data:
        logging.info(f"Image response found in cache for key: {image_key}")
        status_code, image_content = pickle.loads(cached_response_data)
    else:
        logging.info(f"Image not found in cache for key: {image_key}, downloading from {image_url}")
        response = requests.get(image_url)
        status_code, image_content = response.status_code, response.content
        # Cache the status code and image content
        redis.setex(image_key, 300, pickle.dumps((status_code, image_content)))

    font_path = "/usr/share/fonts/truetype/MonaspaceNeon-Light.otf"

    if status_code != 200:
        with db.tx() as transaction:
            status_obj = db.status.upsert(
                where={"name": "404"},
                data={"create": {"name": "404"}, "update": {}},
            )
            camera = db.camera.upsert(
                where={"coaId": id},
                data={
                    "create": {"coaId": id, "statusId": status_obj.id},
                    "update": {"statusId": status_obj.id},
                },
            )
        
        error_img_base_w, error_img_base_h = 1920, 1080
        img = Image.new("RGB", (error_img_base_w, error_img_base_h), color="black")
        d = ImageDraw.Draw(img)
        
        base_font_size_status = 150
        status_font = _get_scaled_font(font_path, base_font_size_status, error_img_base_h)
        status_x, status_y = _get_scaled_coords(10, 10, error_img_base_w, error_img_base_h)
        d.text((status_x, status_y), str(status_code), font=status_font, fill=(255, 255, 255))

        base_font_size_xml = 24
        xml_font = _get_scaled_font(font_path, base_font_size_xml, error_img_base_h)
        xml_x, xml_y = _get_scaled_coords(10, 200, error_img_base_w, error_img_base_h)
        
        try:
            decoded_content = image_content.decode()
        except (UnicodeDecodeError, AttributeError): # AttributeError if image_content is None
            decoded_content = "Error: Content not available or undecodable."
            logging.error(f"Failed to decode image_content for error image: {type(image_content)}")


        d.text(
            (xml_x, xml_y),
            pretty_print_xml(decoded_content),
            font=xml_font,
            fill=(255, 255, 255),
        )
        img_io = BytesIO()
        img.save(img_io, "JPEG", quality=70)
        img_io.seek(0)
        return send_file(img_io, mimetype="image/jpeg")

    # Process successful response
    original_image_stream_for_caching_and_processing = BytesIO(image_content)

    sha256_hash = hashlib.sha256()
    sha256_hash.update(image_content) # Hash the original raw bytes
    image_hash = sha256_hash.hexdigest()

    logging.info(f"SHA256 hash of the image: {image_hash}")

    final_image_data_for_response: BytesIO

    if (
        image_hash == "58da0d53512030c5748d6ecf8337419586ab95d91e1ca2f9d6347cb8879ea960"
    ):  # unavailable placeholder image
        with db.tx() as transaction:
            status_obj = db.status.upsert(
                where={"name": "unavailable"},
                data={"create": {"name": "unavailable"}, "update": {}},
            )
            camera = db.camera.upsert(
                where={"coaId": id},
                data={
                    "create": {"coaId": id, "statusId": status_obj.id},
                    "update": {"statusId": status_obj.id},
                },
            )
        original_image_stream_for_caching_and_processing.seek(0)
        final_image_data_for_response = original_image_stream_for_caching_and_processing
    else:  # Image is "ok", apply overlay
        with db.tx() as transaction:
            status_obj = db.status.upsert(
                where={"name": "ok"},
                data={"create": {"name": "ok"}, "update": {}},
            )
            camera = db.camera.upsert(
                where={"coaId": id},
                data={
                    "create": {"coaId": id, "statusId": status_obj.id},
                    "update": {"statusId": status_obj.id},
                },
            )
            image_record = db.image.upsert(
                where={"hash": image_hash, "cameraId": camera.id},
                data={
                    "create": {
                        "hash": image_hash,
                        "cameraId": camera.id,
                        "statusId": status_obj.id,
                    },
                    "update": {"statusId": status_obj.id},
                },
            )

        # Cache the ORIGINAL image stream in Redis if not already present
        if not redis.exists(f"images:{image_hash}"):
            original_image_stream_for_caching_and_processing.seek(0)
            serialized_original_image = pickle.dumps(original_image_stream_for_caching_and_processing)
            redis.setex(f"images:{image_hash}", 86400 * 1, serialized_original_image)

        original_image_stream_for_caching_and_processing.seek(0)
        pillow_image = Image.open(original_image_stream_for_caching_and_processing)
        img_w, img_h = pillow_image.size
        d = ImageDraw.Draw(pillow_image)
        
        base_font_size_overlay = 24
        overlay_font = _get_scaled_font(font_path, base_font_size_overlay, img_h)

        text_x_base_on_1920, text_y_base_on_1080 = 1720, 10
        text_x, text_y = _get_scaled_coords(text_x_base_on_1920, text_y_base_on_1080, img_w, img_h)
        
        text_to_draw = f"{image_hash[:8]}/{id}"

        base_stroke_pixel_offset = 3
        scaled_stroke_offset = _get_scaled_offset(base_stroke_pixel_offset, img_h)

        for i_offset in range(-scaled_stroke_offset, scaled_stroke_offset + 1):
            for j_offset in range(-scaled_stroke_offset, scaled_stroke_offset + 1):
                d.text(
                    (text_x + i_offset, text_y + j_offset),
                    text_to_draw,
                    font=overlay_font,
                    fill=(0, 0, 0),
                )

        d.text(
            (text_x, text_y),
            text_to_draw,
            font=overlay_font,
            fill=(255, 255, 255),
        )

        modified_image_io = BytesIO()
        # Determine output format; original code saved as JPEG quality 100 for response
        # S3 PNG used pillow_image.save(..., "PNG")
        # S3 JPG used original image_content.
        # For response, stick to JPEG 100 as before.
        pillow_image.save(modified_image_io, "JPEG", quality=100)
        modified_image_io.seek(0)
        final_image_data_for_response = modified_image_io
        
        # --- S3 Upload section ---
        # This preserves the original behavior: PNGs to S3 get the overlay, JPGs to S3 get the original image.
        try:
            image_type_for_s3 = 'jpg' # This was hardcoded in the original snippet for S3 logic section

            if image_type_for_s3 == 'png':
                img_io_s3 = BytesIO()
                pillow_image.save(img_io_s3, "PNG") # pillow_image has the overlay
                content_type_s3 = "image/png"
                ext_s3 = "png"
            elif image_type_for_s3 == 'jpg':
                # Use the original image bytes received from the source for S3 JPG
                img_io_s3 = BytesIO(image_content) # image_content is original, without overlay
                content_type_s3 = "image/jpeg"
                ext_s3 = "jpg"
            else:
                raise ValueError(f"Unsupported image_type_for_s3: {image_type_for_s3}")
            img_io_s3.seek(0)

            s3_key = f"cameras/{id}/{image_hash}.{ext_s3}"
            logging.info(f"Attempting to upload {ext_s3.upper()} to S3 with key: {s3_key}")

            s3 = boto3.client(
                "s3",
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
            )

            try:
                s3.head_object(Bucket="atx-traffic-cameras", Key=s3_key)
                logging.info(f"File already exists in S3: {s3_key}, skipping upload.")
            except s3.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    s3.upload_fileobj(
                        img_io_s3,
                        "atx-traffic-cameras",
                        s3_key,
                        ExtraArgs={"ContentType": content_type_s3},
                    )
                    logging.info(f"Uploaded {ext_s3.upper()} to s3://atx-traffic-cameras/{s3_key}")
                else:
                    logging.error(f"Error checking S3 for {s3_key}: {e}")
        except Exception as e:
            logging.error(f"Failed during S3 processing: {e}")

    final_image_data_for_response.seek(0)
    return send_file(final_image_data_for_response, mimetype="image/jpeg")
