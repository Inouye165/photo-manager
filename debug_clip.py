from transformers import pipeline
from PIL import Image, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()

classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

labels = [
    "a photo of a person", 
    "a photo of a dog", 
    "a photo of a bunny", 
    "a photo of a bird", 
    "a photo of a cat",
    "a photo of an animal",
    "a photo of a landscape",
    "a blurry photo",
    "a digital graphic or screenshot",
    "a photo of a room"
]

path = r"C:\Users\inouy\OneDrive\Pictures\Projects\IMG_5657.HEIC"
pil_img = Image.open(path)
pil_img = ImageOps.exif_transpose(pil_img)
if pil_img.mode != "RGB":
    pil_img = pil_img.convert("RGB")

results = classifier(pil_img, candidate_labels=labels)
for r in results:
    print(f"{r['score']:.4f}: {r['label']}")
