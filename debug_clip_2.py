import os
from transformers import pipeline
from PIL import Image, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()
classifier = pipeline("zero-shot-image-classification", model="openai/clip-vit-base-patch32")

labels = [
    "person", "dog", "bunny", "bird", "cat", "animal",
    "screenshot", "landscape", "object", "room"
]

folder = r"C:\Users\inouy\OneDrive\Pictures\Projects"
all_files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.heic'))][:5]

for f in all_files:
    path = os.path.join(folder, f)
    try:
        pil_img = ImageOps.exif_transpose(Image.open(path))
        if pil_img.mode != "RGB": pil_img = pil_img.convert("RGB")
        res = classifier(pil_img, candidate_labels=labels)
        print(f"--- {f} ---")
        for r in res[:3]:
            print(f"  {r['score']:.3f}: {r['label']}")
    except:
        pass
