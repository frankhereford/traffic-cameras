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

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.9
)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )


draw = ImageDraw.Draw(image)
font = ImageFont.truetype("MonaspaceArgon-Medium.otf", 18)


for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )
    draw.rectangle(box, outline="red", width=3)
    draw.text(
        (box[0], box[1]), model.config.id2label[label.item()], fill="red", font=font
    )

image.show()
