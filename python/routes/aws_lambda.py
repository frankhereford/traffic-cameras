import time
import logging
import requests
import json

def aws_lambda(db, redis):
    while True:
        job = db.image.find_first(
            where={
                "detectionsProcessed": False,
            },
            include={"camera": True},
        )
        if job is None:
            time.sleep(10)
            continue

        logging.info(job.hash)

        key = f"images:{job.hash}"

        camera = db.camera.find_first(
            where={"id": job.camera.id}, include={"Location": True}
        )
        
        # Extract the hash from the job
        image_hash = job.hash
        
        # Extract the coaId from the camera
        coa_id = camera.coaId
        
        # Define the Lambda function URL
        lambda_url = "https://enzcekpb53uebdi4uj62ul5ium0hjjhq.lambda-url.us-east-1.on.aws"
        # lambda_url = "http://localhost:9000/2015-03-31/functions/function/invocations"
        
        # Prepare the payload
        payload = {
            "coaId": coa_id,
            "hash": image_hash
        }

        try:
            # Make the request to the Lambda function
            logging.info(f"Calling Lambda function with payload: {payload}")
            response = requests.post(lambda_url, json=payload)
            
            # Check if the request was successful
            if response.status_code == 200:
                logging.info("Lambda function call successful")
                # Process the response if needed
                # response_data = response.json()
            else:
                logging.error(f"Lambda function call failed with status code: {response.status_code}")
                raise Exception(f"Response: {response.text}")
                
        except Exception as e:
            logging.error(f"Error calling Lambda function: {str(e)}")
            # Mark the image as processed to prevent retrying
            db.image.update(
                where={"hash": image_hash},
                data={"detectionsProcessed": True}
            )

        with db.tx() as transaction:
            status = db.status.upsert(
                where={"name": "ok"},
                data={"create": {"name": "ok"}, "update": {}},
            )
            camera = db.camera.upsert(
                where={"coaId": coa_id},  # Changed from id to coa_id
                data={
                    "create": {"coaId": coa_id, "statusId": status.id},  # Changed from id to coa_id
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

        
        time.sleep(.5)
        
        # Continue with the rest of the processing for this image
        # (You might want to add more code here based on your application requirements)

