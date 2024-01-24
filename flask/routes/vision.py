from flask import Flask, request, jsonify, send_file
import logging
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import base64
import tempfile
from io import BytesIO
from PIL import Image

# import redis
# import pickle

logging.basicConfig(level=logging.INFO)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-101", revision="no_timm"
)
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-101", revision="no_timm"
)


def vision(db, redis):
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
