#!/usr/bin/env python3
"""
Identity model training script for PhotoFinder.
Trains face/animal recognition models using labeled data from the labeling workflow.

This is a foundation script that can be extended with actual model training
implementations as needed.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List

from src.label_manager import LabelManager


def prepare_training_data(output_dir: str, min_samples_per_label: int = 5) -> Dict:
    """
    Prepare training data by reading confirmed labels and organizing by assigned label.
    
    Args:
        output_dir: Directory containing the labeled data
        min_samples_per_label: Minimum number of samples required per label
        
    Returns:
        Dictionary organized by label with list of training examples
    """
    print("Preparing training data...")
    
    # Initialize label manager
    label_manager = LabelManager(output_dir)
    
    # Export confirmed labels for training
    training_data = label_manager.export_for_training(status="confirmed")
    
    # Filter labels with insufficient samples
    filtered_data = {}
    for label, examples in training_data.items():
        if len(examples) >= min_samples_per_label:
            filtered_data[label] = examples
            print(f"  {label}: {len(examples)} samples")
        else:
            print(f"  {label}: {len(examples)} samples (skipped - < {min_samples_per_label} minimum)")
    
    print(f"\nTotal labels for training: {len(filtered_data)}")
    total_samples = sum(len(examples) for examples in filtered_data.values())
    print(f"Total training samples: {total_samples}")
    
    return filtered_data


def create_dataset_structure(output_dir: str, training_data: Dict) -> str:
    """
    Create a standard dataset structure for training.
    
    Args:
        output_dir: Base output directory
        training_data: Dictionary of labeled training examples
        
    Returns:
        Path to created dataset directory
    """
    dataset_dir = Path(output_dir) / "_training_dataset"
    dataset_dir.mkdir(exist_ok=True)
    
    print(f"\nCreating dataset structure in {dataset_dir}...")
    
    for label, examples in training_data.items():
        label_dir = dataset_dir / label
        label_dir.mkdir(exist_ok=True)
        
        print(f"  {label}/: {len(examples)} files")
        
        for i, example in enumerate(examples):
            # Prefer crop files, fall back to original images
            source_path = None
            
            if example.get("crop_path"):
                crop_path = Path(output_dir) / example["crop_path"]
                if crop_path.exists():
                    source_path = crop_path
            
            if not source_path:
                # Try original image
                orig_path = Path(output_dir) / example["image_path"]
                if orig_path.exists():
                    source_path = orig_path
            
            if source_path:
                # Copy to dataset with standardized naming
                target_path = label_dir / f"{i:04d}_{source_path.name}"
                import shutil
                shutil.copy2(source_path, target_path)
            else:
                print(f"    Warning: Could not find source file for {example}")
    
    print(f"Dataset created with {len(training_data)} classes")
    return str(dataset_dir)


def train_face_recognition_model(dataset_dir: str):
    """
    Train a face recognition model using the prepared dataset.
    
    This is a placeholder implementation. In a full implementation, you would:
    1. Use a library like face_recognition, insightface, or deepface
    2. Extract face embeddings from training images
    3. Train a classifier (SVM, neural network, etc.)
    4. Save the trained model
    
    Args:
        dataset_dir: Directory containing labeled face images
    """
    print("\n" + "="*50)
    print("FACE RECOGNITION TRAINING (Placeholder)")
    print("="*50)
    
    print("TODO: Implement face recognition training")
    print("Suggested libraries:")
    print("  - face_recognition (simple, good for getting started)")
    print("  - insightface (more advanced, better accuracy)")
    print("  - deepface (multiple backends, easy to use)")
    
    print("\nExample training pipeline:")
    print("1. Detect and align faces in training images")
    print("2. Extract face embeddings (128-D or 512-D vectors)")
    print("3. Train classifier (SVM, Random Forest, or Neural Network)")
    print("4. Save model and label mapping")
    
    # For now, just analyze the dataset structure
    dataset_path = Path(dataset_dir)
    if dataset_path.exists():
        classes = [d.name for d in dataset_path.iterdir() if d.is_dir()]
        print(f"\nFound {len(classes)} identity classes:")
        for cls in classes:
            class_dir = dataset_path / cls
            num_images = len([f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            print(f"  {cls}: {num_images} images")


def train_animal_recognition_model(dataset_dir: str):
    """
    Train an animal recognition model using the prepared dataset.
    
    This is a placeholder implementation. In a full implementation, you would:
    1. Use a fine-tuning approach with pre-trained models
    2. Or train from scratch if you have enough data
    3. Use transfer learning for better performance with limited data
    
    Args:
        dataset_dir: Directory containing labeled animal images
    """
    print("\n" + "="*50)
    print("ANIMAL RECOGNITION TRAINING (Placeholder)")
    print("="*50)
    
    print("TODO: Implement animal recognition training")
    print("Suggested approaches:")
    print("  - Fine-tune pre-trained ResNet/EfficientNet on animal images")
    print("  - Use YOLO for animal detection + classification")
    print("  - Train separate models for different animal types")
    
    print("\nExample training pipeline:")
    print("1. Preprocess images (resize, normalize)")
    print("2. Split data into train/validation sets")
    print("3. Set up data augmentation")
    print("4. Fine-tune pre-trained model")
    print("5. Evaluate and save best model")
    
    # For now, just analyze the dataset structure
    dataset_path = Path(dataset_dir)
    if dataset_path.exists():
        classes = [d.name for d in dataset_path.iterdir() if d.is_dir()]
        print(f"\nFound {len(classes)} animal classes:")
        for cls in classes:
            class_dir = dataset_path / cls
            num_images = len([f for f in class_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            print(f"  {cls}: {num_images} images")


def export_training_summary(output_dir: str, training_data: Dict):
    """
    Export a summary of training data for reference.
    
    Args:
        output_dir: Output directory
        training_data: Dictionary of labeled training examples
    """
    summary_file = Path(output_dir) / "_training_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("PhotoFinder Training Data Summary\n")
        f.write("="*40 + "\n\n")
        
        total_examples = sum(len(examples) for examples in training_data.values())
        f.write(f"Total Labels: {len(training_data)}\n")
        f.write(f"Total Examples: {total_examples}\n\n")
        
        f.write("Labels and Examples:\n")
        f.write("-" * 20 + "\n")
        
        # Sort by number of examples (descending)
        sorted_labels = sorted(training_data.items(), key=lambda x: len(x[1]), reverse=True)
        
        for label, examples in sorted_labels:
            f.write(f"{label}: {len(examples)} examples\n")
            
            # List source files for this label
            for example in examples[:5]:  # Show first 5 examples
                source = example.get("crop_path", example.get("image_path", "Unknown"))
                f.write(f"  - {source}\n")
            
            if len(examples) > 5:
                f.write(f"  ... and {len(examples) - 5} more\n")
            f.write("\n")
    
    print(f"Training summary saved to: {summary_file}")


def main():
    """Main training pipeline"""
    if len(sys.argv) < 2:
        print("Usage: python train_identity_model.py <output_directory>")
        print("Example: python train_identity_model.py ./sorted_output")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    if not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist.")
        sys.exit(1)
    
    print("PhotoFinder Identity Model Training")
    print("="*50)
    
    # Step 1: Prepare training data
    training_data = prepare_training_data(output_dir)
    
    if not training_data:
        print("No training data found. Please run the labeling workflow first.")
        sys.exit(1)
    
    # Step 2: Create dataset structure
    dataset_dir = create_dataset_structure(output_dir, training_data)
    
    # Step 3: Export training summary
    export_training_summary(output_dir, training_data)
    
    # Step 4: Placeholder training implementations
    train_face_recognition_model(dataset_dir)
    train_animal_recognition_model(dataset_dir)
    
    print("\n" + "="*50)
    print("TRAINING PIPELINE COMPLETE")
    print("="*50)
    print(f"Dataset created: {dataset_dir}")
    print("Ready for model training implementation!")
    print("\nNext steps:")
    print("1. Choose your training approach (face_recognition, insightface, etc.)")
    print("2. Implement the actual training functions")
    print("3. Save trained models for use in PhotoFinder")


if __name__ == "__main__":
    main()
