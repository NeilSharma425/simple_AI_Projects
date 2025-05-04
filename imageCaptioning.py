from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Initialize the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load image
image = Image.open("space.jpg")

# prep image
inputs = processor(image, return_tensors="pt")

# gen captions
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0],skip_special_tokens=True)
print("Generated Caption:", caption)