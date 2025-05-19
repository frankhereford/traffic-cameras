import time
import logging
import boto3
import io
import pickle
from PIL import Image, ImageDraw


logging.basicConfig(level=logging.INFO)


def throttle(func):
    throttle._last_called = None

    def wrapper(*args, **kwargs):
        if throttle._last_called is not None:
            time_since_last_call = time.time() - throttle._last_called
            sleep_time = int(round(max(0, 1 - time_since_last_call), 0))
            logging.info(f"sleep time: {sleep_time} ")
            time.sleep(sleep_time)
        throttle._last_called = time.time()
        return func(*args, **kwargs)

    return wrapper

def rekognition(db, redis):
    while True:
        process_one_image(db, redis)

@throttle
def process_one_image(db, redis):
    logging.info("rekognition processing one image")
    job = db.image.find_first(
        where={
            "detectionsProcessed": False,
        },
        include={"camera": True},
    )
    if job is None:
        return

    logging.info(job.hash)

    key = f"images:{job.hash}"

    camera = db.camera.find_first(
        where={"id": job.camera.id}, include={"Location": True}
    )

    # print(camera.Location)

    if redis.exists(key):
        serialized_data = redis.get(key)
        image_byte_stream = pickle.loads(serialized_data)
    else:
        db.image.update(
            where={
                "id": job.id,
            },
            data={
                "detectionsProcessed": True,
            },
        )
        return

    image = Image.open(image_byte_stream)

    # 1. Convert PIL Image to raw bytes
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    img_bytes = buffer.getvalue()

    # 2. Call Rekognition
    client = boto3.client('rekognition', region_name='us-east-1')  # adjust region
    response = client.detect_labels(
        Image={'Bytes': img_bytes},
        MaxLabels=10,
        MinConfidence=75
    )

    # 3. Inspect results
    for lbl in response['Labels']:
        logging.info(f"{lbl['Name']} ({lbl['Confidence']:.1f}%)")
        for inst in lbl.get('Instances', []):
            logging.info("  BoundingBox:", inst['BoundingBox'])
