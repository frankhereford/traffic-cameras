import requests
from PIL import Image, ImageDraw, ImageFont
import torch

# Import the required library
from PIL import ImageFont


from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
url = "https://cctv.austinmobility.io/image/71.jpg"
image = Image.open(requests.get(url, stream=True).raw)
texts = [
    [
        "car",
        "bus",
        "person",
        # "sign",
        # "traffic light",
        # "zebra crossing",
        # "road",
        "utility pole",
    ]
]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions [batch_size, 2]
target_sizes = torch.Tensor([image.size[::-1]])
# Convert outputs (bounding boxes and class logits) to COCO API
results = processor.post_process_object_detection(
    outputs=outputs, threshold=0.1, target_sizes=target_sizes
)

i = 0  # Retrieve predictions for the first image for the corresponding text queries
text = texts[i]
boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

# Create a draw object
draw = ImageDraw.Draw(image)

# Load a font
font = ImageFont.truetype("MonaspaceArgon-Medium.otf", 18)

# Loop over each box
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}"
    )

    # Draw the bounding box
    draw.rectangle(box, outline="red", width=3)

    # Draw the label with the specified font
    draw.text(
        (box[0], box[1] - 20),
        str(round(score.item(), 3)) + ": " + text[label],
        fill="red",
        font=font,
    )

# Display the image
image.show()
