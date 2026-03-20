#!/usr/bin/env python3
"""
Create a zip archive of all files except images.
Saves the archive to the Downloads folder.
"""

import zipfile
import os
from pathlib import Path

def create_archive():
    # Define source and destination
    source_dir = Path('c:/Users/inouy/photofinder')
    downloads_dir = Path.home() / 'Downloads'
    zip_path = downloads_dir / 'photofinder_archive.zip'
    
    # Common image file extensions to exclude
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.heic', '.HEIC', '.webp', '.svg'}
    
    # Large model files to exclude (58 MB total)
    model_files = {'yolov8m.pt', 'yolov8n.pt'}
    
    # Create zip file
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Skip .git directory
            if '.git' in root:
                continue
                
            for file in files:
                file_path = Path(root) / file
                file_ext = file_path.suffix.lower()
                
                # Skip image files
                if file_ext in image_extensions:
                    continue
                    
                # Skip model files
                if file in model_files:
                    continue
                    
                # Skip the zip file itself if it exists in source
                if file == 'photofinder_archive.zip':
                    continue
                    
                # Calculate relative path for zip
                arcname = str(file_path.relative_to(source_dir))
                zipf.write(file_path, arcname)
    
    print(f'Archive created: {zip_path}')
    print(f'Size: {zip_path.stat().st_size / (1024*1024):.2f} MB')

if __name__ == '__main__':
    create_archive()
