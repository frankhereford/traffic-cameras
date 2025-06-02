import sys
import json
import boto3
import os
from botocore.exceptions import ClientError
import os.path
import torch
import logging
from PIL import Image
from io import BytesIO
from transformers import DetrImageProcessor, DetrForObjectDetection

# Set up logging
logging.basicConfig(level=logging.INFO)

CACHE_DB_PATH = '/tmp/detection_cache.db'  # SQLite cache database path
S3_BUCKET = 'atx-traffic-cameras'  # S3 bucket name

# Initialize the object detection model and processor
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm")

def detect_objects(image_path):
    """
    Perform object detection on an image
    
    Parameters:
        image_path: String path to the image file
    Returns:
        List of detected objects with their labels, confidence scores, and bounding boxes
    """
    # Open the image
    image = Image.open(image_path)
    
    # Process the image for object detection
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    
    # Convert outputs to COCO API format and filter with threshold
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.9
    )[0]
    
    # Prepare detection results
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]
        confidence = round(score.item(), 3)
        
        logging.info(f"Detected {label_name} with confidence {confidence} at location {box}")
        
        detections.append({
            "label": label_name,
            "confidence": confidence,
            "box": {
                "xMin": box[0],
                "yMin": box[1],
                "xMax": box[2],
                "yMax": box[3]
            }
        })
    
    return detections

def handler(event, context):
    """
    Main Lambda handler function
    Parameters:
        event: Dict containing the Lambda function event data
        context: Lambda runtime context
    Returns:
        Dict containing status message
    """
    try:
        print("Received event: " + json.dumps(event))
        
        # Extract S3 information from the event and parse coaId and hash
        record = event["Records"][0]
        s3_info = record["s3"]
        bucket_name = s3_info["bucket"]["name"]
        object_key = s3_info["object"]["key"]
        parts = object_key.split("/")
        if len(parts) < 3:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid S3 object key format'})
            }
        coa_id = parts[1]
        filename = parts[2]
        file_hash, file_ext = os.path.splitext(filename)
        file_ext = file_ext.lstrip('.').lower()  # remove dot and normalize
        
        # Initialize S3 client with configuration constants
        # If AWS credentials are None, boto3 will use the default AWS credential chain
        s3_client_args = {'service_name': 's3', 'region_name': 'us-east-1'}
        # if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        #     s3_client_args.update({
        #         'aws_access_key_id': AWS_ACCESS_KEY_ID,
        #         'aws_secret_access_key': AWS_SECRET_ACCESS_KEY,
        #     })
        s3_client = boto3.client(**s3_client_args)
        
        # Use the configured bucket name
        bucket_name = S3_BUCKET
        object_key = f"cameras/{coa_id}/{file_hash}.{file_ext}"
        
        # Log the S3 address
        s3_address = f"s3://{bucket_name}/{object_key}"
        print(f"Attempting to download from S3 location: {s3_address}")
        
        # Download file from S3 to /tmp (Lambda's writable directory)
        local_file_path = f"/tmp/{file_hash}.{file_ext}"
        print(f"Local file path for download: {local_file_path}")
        
        try:
            # Download the file
            s3_client.download_file(bucket_name, object_key, local_file_path)
            print(f"Successfully downloaded file from {s3_address} to {local_file_path}")
            
            # For demonstration purposes - get file info
            file_info = os.stat(local_file_path)
            file_size = file_info.st_size
            
            # Perform object detection on the downloaded image
            detections = detect_objects(local_file_path)

            print(f"Detected {len(detections)} objects in the image.")
            # print(f"Detections: {json.dumps(detections, indent=2)}")

            input = {
                'coaId': coa_id,
                'hash': file_hash,
                'bucket': bucket_name,
                'objectKey': object_key,
                'localPath': local_file_path,
                'fileSize': file_size,
                'detections': detections
            }
            
            # Process the message and return success
            response = {
                'statusCode': 200,
                'body': json.dumps(input)
            }
            
            # Send message to SQS with the full input event and status
            try:
                sqs_client = boto3.client('sqs')
                message_payload = event.copy()
                message_payload["status"] = "processed"
                message_payload["detections"] = detections  # include detections in the payload
                print(f"Sending message to SQS: {json.dumps(message_payload)}")
                sqs_client.send_message(
                    QueueUrl="https://sqs.us-east-1.amazonaws.com/969346816767/camera-detections",
                    MessageBody=json.dumps(message_payload)
                )
            except Exception as sqs_error:
                print(f"Failed to send SQS message: {sqs_error}")
            
            return response
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey' or e.response['Error']['Code'] == 'NoSuchBucket':
                return {
                    'statusCode': 404,
                    'body': json.dumps({
                        'error': f"File not found: s3://{bucket_name}/{object_key}"
                    })
                }
            else:
                raise

    except Exception as e:
        # Handle any exceptions that occur
        print(f"Error processing event: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }


