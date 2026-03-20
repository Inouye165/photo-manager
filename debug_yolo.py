import os
import cv2
import random
from ultralytics import YOLO

def debug_yolo():
    folder = r"C:\Users\inouy\OneDrive\Pictures\Projects"
    model = YOLO("yolov8m.pt")
    
    # Get all jpg/png/heic files
    all_files = []
    for f in os.listdir(folder):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
            all_files.append(os.path.join(folder, f))
            
    # Check 15 files
    sample = all_files[:15]
    
    for path in sample:
        print(f"--- Analyzing {os.path.basename(path)} ---")
        try:
            results = model(path, verbose=False, conf=0.1)
            for r in results:
                if r.boxes:
                    for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
                        class_id = int(cls.item())
                        class_name = model.names[class_id]
                        print(f"   Found: {class_name} ({conf.item():.2f})")
                else:
                    print("   Found: nothing")
        except Exception as e:
            print(f"   Error: {e}")

if __name__ == "__main__":
    debug_yolo()
