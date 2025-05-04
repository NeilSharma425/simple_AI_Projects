import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Initialize the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
def generate_caption(image): 

    # prep image
    inputs = processor(image, return_tensors="pt")
    # gen captions
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0],skip_special_tokens=True)
    return caption
def caption_img(img):
    try:
        caption = generate_caption(img)
        return caption
    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(
    fn = caption_img,
    inputs = gr.Image(type="pil"),
    outputs = "text",
    title = "BLIP Image Caption",
    description = "Upload image to generate a caption"
)
iface.launch(share=True)