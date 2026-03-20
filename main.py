import os
import shutil
import sys

from src.image_processor import ImageProcessor
from src.detection_metadata import DetectionMetadata


def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <input_directory> <output_directory>")
        print("Example: python main.py ./input_photos ./true_photos")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):
        print(f"Error: Source directory '{input_dir}' does not exist.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    people_dir = os.path.join(output_dir, "people")
    animals_dir = os.path.join(output_dir, "animals")
    others_dir = os.path.join(output_dir, "others")
    debug_dir = os.path.join(output_dir, "_debug_boxes")
    training_crops_dir = os.path.join(output_dir, "_training_crops")
    people_crops_dir = os.path.join(training_crops_dir, "people")
    animals_crops_dir = os.path.join(training_crops_dir, "animals")
    
    # Clear subfolders before each run (but keep working_dir root untouched)
    for subfolder in [people_dir, animals_dir, others_dir, debug_dir, people_crops_dir, animals_crops_dir]:
        if os.path.exists(subfolder):
            for file in os.listdir(subfolder):
                file_path = os.path.join(subfolder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        os.makedirs(subfolder, exist_ok=True)

    processor = ImageProcessor()
    metadata_manager = DetectionMetadata(output_dir)

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".heic", ".heif"}

    copied_count = 0
    total_images = 0

    print(f"Scanning directory: {input_dir}")
    print("-" * 50)

    for root, dirs, files in os.walk(input_dir):
        # Prevent recursing into output directories if input_dir == output_dir
        dirs[:] = [d for d in dirs if d.lower() not in ("people", "animals", "others", "_debug_boxes")]

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                file_path = os.path.join(root, file)
                total_images += 1
                try:
                    # Always run subject detection first since the user specifically wants people and animals
                    subjects = processor.detect_subjects(file_path)
                    has_person = subjects.get("has_person", False)
                    has_animal = subjects.get("has_animal", False)

                    # If it has a person or animal, it's definitively a photo of interest
                    if has_person or has_animal:
                        is_photo = True
                        status = "SUBJECT_DETECTED"
                        print(f"[{status}] {file} (Person: {has_person}, Animal: {has_animal})")
                    else:
                        # Fallback to the heuristic
                        is_photo, metrics = processor.is_true_photo(file_path)
                        status = "PHOTO" if is_photo else "GRAPHIC"
                        lap = metrics.get("laplacian_variance", 0)
                        ent = metrics.get("color_entropy", 0)
                        print(f"[{status}] {file} (Laplacian: {lap:.2f}, Entropy: {ent:.2f})")

                    if is_photo:
                        target_dirs = []
                        if has_person:
                            target_dirs.append(people_dir)
                        if has_animal:
                            target_dirs.append(animals_dir)
                        if not has_person and not has_animal:
                            # if neither, put it in the others folder
                            target_dirs.append(others_dir)

                        for target_dir in target_dirs:
                            out_path = os.path.join(target_dir, file)
                            # Handle duplicate filenames
                            counter = 1
                            while os.path.exists(out_path):
                                name = os.path.splitext(file)[0]
                                out_path = os.path.join(target_dir, f"{name}_{counter}{ext}")
                                counter += 1

                            shutil.copy2(file_path, out_path)

                        # Save debug annotation
                        debug_filename = f"{os.path.splitext(file)[0]}_debug.jpg"
                        debug_path = os.path.join(debug_dir, debug_filename)
                        debug_res = processor.annotate_detections(file_path, debug_path)
                        if debug_res.get("saved"):
                            print(f"   -> Saved debug annotation: {debug_path}")

                        # Get detailed detections and generate crops
                        detailed_detections = processor.get_detailed_detections(file_path)
                        if detailed_detections:
                            print(f"   -> Found {len(detailed_detections)} detections for labeling")
                            
                            # Generate crops for each detection
                            for i, detection in enumerate(detailed_detections):
                                class_name = detection["class_name"]
                                bbox = detection["bbox"]
                                
                                # Determine crop directory
                                if class_name == "person":
                                    crop_dir = people_crops_dir
                                elif class_name in {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}:
                                    crop_dir = animals_crops_dir
                                else:
                                    continue  # Skip other classes
                                
                                # Generate unique crop filename
                                base_name = os.path.splitext(file)[0]
                                crop_filename = f"{base_name}_crop_{i}_{class_name}.jpg"
                                crop_path = os.path.join(crop_dir, crop_filename)
                                
                                # Generate crop
                                if processor.generate_crop(file_path, bbox, crop_path):
                                    detection["crop_path"] = os.path.relpath(crop_path, output_dir)
                                    print(f"   -> Saved crop: {crop_path}")
                            
                            # Save detection metadata
                            metadata_manager.add_detections(file_path, detailed_detections, debug_path if debug_res.get("saved") else None)

                        copied_count += 1

                except Exception as e:
                    print(f"[ERROR] Could not process {file}: {e}")

    print("-" * 50)
    print(f"Done! Found {copied_count} true photos out of {total_images} scanned images.")
    print(f"Copied true photos to: {output_dir}")


if __name__ == "__main__":
    main()
