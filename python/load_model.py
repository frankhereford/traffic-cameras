from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont
import requests

url = "https://cctv.austinmobility.io/image/105.jpg"  # oltof / s 1st
# url = "https://cctv.austinmobility.io/image/326.jpg"
# url = "https://cctv.austinmobility.io/image/89.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-101", revision="no_timm"
)
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-101", revision="no_timm"
)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
