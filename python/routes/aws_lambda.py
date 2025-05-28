import time
import logging
import requests
import json
import os  # new import for environment variables
import boto3  # added boto3
import io
import base64              # new import for base64 encoding
from PIL import Image, ImageDraw   # new import for drawing on images
import numpy as np         # new import for numerical operations
from matplotlib.path import Path   # new import for convex hull check
from scipy.spatial import ConvexHull  # new import for convex hull check

# Supporting functions from rekognition.py
def is_point_in_hull(hull, point):
    hull_path = Path(hull.points[hull.vertices])
    return hull_path.contains_point(point)

def check_point_in_camera_location(camera, point):
    if not camera:
        return False
    locations = camera.Location
    points = np.array([[location.x, location.y] for location in locations])
    if len(points) >= 3 and points.shape[1] >= 2:
        hull = ConvexHull(points)
        return is_point_in_hull(hull, point)
    else:
        logging.info("Cannot create a convex hull due to insufficient or invalid points.")
        return False

def aws_lambda(db, redis):
    sqs = boto3.client('sqs', region_name='us-east-1')  # specify AWS region
    queue_url = sqs.get_queue_url(QueueName="camera-detections")['QueueUrl']
    
    while True:
        response = sqs.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=20
        )
        if 'Messages' in response:
            for message in response['Messages']:
                logging.info("Received message: " + message.get('Body', ''))
                # New: Extract S3 object key, coaId, and file hash, then download file
                body = message.get('Body', '')
                try:
                    data = json.loads(body)
                    if "Records" in data and data["Records"]:
                        for record in data["Records"]:
                            object_key = record["s3"]["object"]["key"]
                            parts = object_key.split('/')
                            if len(parts) >= 3:
                                coa_id = parts[1]
                                file_name = parts[2]
                                file_hash = file_name.split('.')[0]
                                # Initialize S3 client with credentials from env vars
                                s3_client_args = {
                                    'service_name': 's3',
                                    'region_name': os.environ.get('AWS_REGION', 'us-east-1')
                                }
                                AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
                                AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
                                if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
                                    s3_client_args.update({
                                        'aws_access_key_id': AWS_ACCESS_KEY_ID,
                                        'aws_secret_access_key': AWS_SECRET_ACCESS_KEY,
                                    })
                                s3_client = boto3.client(**s3_client_args)
                                bucket_name = os.environ.get('S3_BUCKET', 'atx-traffic-cameras')
                                expected_key = f"cameras/{coa_id}/{file_hash}.jpg"
                                s3_address = f"s3://{bucket_name}/{expected_key}"
                                logging.info(f"Attempting to download from S3 location: {s3_address}")
                                try:
                                    response = s3_client.get_object(Bucket=bucket_name, Key=expected_key)
                                except Exception as e:
                                    logging.error(f"Error downloading file: {e}")

                                file_data = response['Body'].read()
                                img = Image.open(io.BytesIO(file_data))
                                logging.info(f"Successfully downloaded and loaded image from {s3_address} into memory")

                                db_image = db.image.find_first(where={"hash": file_hash})
                                logging.info(f"Found image in database: {db_image}") 

                                camera_record = db.camera.find_first(
                                    where={"id": db_image.cameraId},
                                    include={"Location": True}
                                )
                                logging.info(f"Found camera in database: {camera_record}")

                                
                                # New: Process detections from message if present
                                if "detections" in data:
                                    img_width, img_height = img.size
                                    for detection in data["detections"]:
                                        box = detection.get("box", {})
                                        xMin = box.get("xMin")
                                        yMin = box.get("yMin")
                                        xMax = box.get("xMax")
                                        yMax = box.get("yMax")
                                        label = detection.get("label")
                                        confidence = detection.get("confidence")
                                        
                                        # Calculate width, height, and padding (20%)
                                        width = xMax - xMin
                                        height = yMax - yMin
                                        padding_width = width * 0.2
                                        padding_height = height * 0.2
                                        
                                        # Compute a padded bounding box
                                        new_box = [
                                            max(0, xMin - padding_width),
                                            max(0, yMin - padding_height),
                                            min(img_width, xMax + padding_width),
                                            min(img_height, yMax + padding_height)
                                        ]
                                        
                                        # Crop the image using the new bounding box
                                        detected_object = img.crop(new_box)
                                        
                                        # Calculate relative bounding box coordinates for drawing
                                        relative_box = [
                                            xMin - new_box[0],
                                            yMin - new_box[1],
                                            xMax - new_box[0],
                                            yMax - new_box[1]
                                        ]
                                        
                                        # Draw the bounding box on the cropped image
                                        draw = ImageDraw.Draw(detected_object)
                                        draw.rectangle(relative_box, outline="red", width=1)
                                        
                                        # Convert the cropped object to base64 encoded JPEG image
                                        byte_stream = io.BytesIO()
                                        detected_object.save(byte_stream, format="JPEG")
                                        byte_stream.seek(0)
                                        encoded_image = base64.b64encode(byte_stream.getvalue()).decode("utf-8")
                                        
                                        logging.info(f"Processed detection: {label} with confidence {confidence:.3f}")
                                        
                                        if camera_record:
                                            center_x = (xMin + xMax) / 2
                                            center_y = (yMin + yMax) / 2
                                            point = (center_x, center_y)
                                            is_in_hull = check_point_in_camera_location(camera_record, point)
                                        else:
                                            is_in_hull = False
                                        
                                        # NEW: Create detection record in DB
                                        db.detection.create(
                                            data={
                                                "label": label,
                                                "confidence": round(confidence, 3),
                                                "xMin": int(xMin),
                                                "yMin": int(yMin),
                                                "xMax": int(xMax),
                                                "yMax": int(yMax),
                                                "imageId": db_image.id if db_image else "",
                                                "picture": encoded_image,
                                                "isInsideConvexHull": is_in_hull,
                                            }
                                        )
                                db.image.update(
                                    where={"id": db_image.id},
                                    data={"detectionsProcessed": True}
                                )

                    else:
                        logging.error("No Records found in message")
                except Exception as e:
                    logging.error(f"Error processing message: {e}")
                sqs.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=message['ReceiptHandle']
                )
        time.sleep(1)

