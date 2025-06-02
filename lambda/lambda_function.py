import sys
import json
import boto3
import os
from botocore.exceptions import ClientError
import os.path

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
        # Log the event for debugging
        print("Received event: " + json.dumps(event))
        
        # Extract parameters from event
        coa_id = event.get("coaId")
        file_hash = event.get("hash")
        
        if not coa_id or not file_hash:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'error': 'Missing required parameters: coaId or hash'
                })
            }
        
        # Initialize S3 client with configuration constants
        # If AWS credentials are None, boto3 will use the default AWS credential chain
        s3_client_args = {'service_name': 's3', 'region_name': AWS_REGION}
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            s3_client_args.update({
                'aws_access_key_id': AWS_ACCESS_KEY_ID,
                'aws_secret_access_key': AWS_SECRET_ACCESS_KEY,
            })
        s3_client = boto3.client(**s3_client_args)
        
        # Use the configured bucket name
        bucket_name = S3_BUCKET
        object_key = f"cameras/{coa_id}/{file_hash}.jpg"
        
        # Log the S3 address
        s3_address = f"s3://{bucket_name}/{object_key}"
        print(f"Attempting to download from S3 location: {s3_address}")
        
        # Download file from S3 to /tmp (Lambda's writable directory)
        local_file_path = f"/tmp/{file_hash}.jpg"
        
        try:
            # Download the file
            s3_client.download_file(bucket_name, object_key, local_file_path)
            print(f"Successfully downloaded file from {s3_address} to {local_file_path}")
            
            # For demonstration purposes - get file info
            file_info = os.stat(local_file_path)
            file_size = file_info.st_size
            
            input = {
                'coaId': coa_id,
                'hash': file_hash,
                'bucket': bucket_name,
                'objectKey': object_key,
                'localPath': local_file_path,
                'fileSize': file_size
            }
            
            # Process the message and return success
            response = {
                'statusCode': 200,
                'body': json.dumps(input)
            }
            
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


