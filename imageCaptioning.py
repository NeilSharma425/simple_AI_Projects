from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Initialize the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load image
image = Image.open("space.jpg")

# prep image
inputs = processor(image, return_tensors="pt")