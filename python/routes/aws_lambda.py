import time
import logging
import requests
import json
import os  # new import for environment variables
import boto3  # added boto3
import io
from PIL import Image

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
                                    file_data = response['Body'].read()
                                    img = Image.open(io.BytesIO(file_data))
                                    logging.info(f"Successfully downloaded and loaded image from {s3_address} into memory")
                                except Exception as e:
                                    logging.error(f"Error downloading file: {e}")
                except Exception as e:
                    logging.error(f"Error processing message: {e}")
                # ...existing code...
                sqs.delete_message(
                    QueueUrl=queue_url,
                    ReceiptHandle=message['ReceiptHandle']
                )
        time.sleep(5)

