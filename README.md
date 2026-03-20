# PhotoFinder

AI-powered photo organization with person and animal detection, featuring debug annotations, incremental labeling, a Flask backend, and a React frontend for atlas and lab views.

## Features

- **Smart Detection**: Uses YOLOv8 to detect people and animals in photos
- **Labeling Workflow**: Web interface to assign names to detected people and animals
- **Training Preparation**: Generates crops and metadata for identity model training
- **Debug Annotations**: Creates bounding box visualizations around detected objects
- **Web UI**: Browser-based interface to view organized photos and manage labels
- **HEIC/HEIF Support**: Full compatibility with Apple image formats
- **Security-Focused**: Implements proper input validation and file access controls
- **Modular Architecture**: Clean separation of concerns with comprehensive testing

## Architecture

This project follows Google's engineering best practices with a modular architecture:
- `src/`: Contains core logic (`ImageProcessor`, `DetectionMetadata`, `LabelManager` classes)
- `src/mirror_manager.py`: Blueprint for immutable `source_originals` handling with a separate `mirror_workspace`
- `src/intelligence_core.py`: Blueprint for GPU-backed embeddings, vector search, and active-learning verification
- `tests/`: Contains PyTest test cases with mocking
- `main.py`: Command-line photo processing with crop generation
- `web_ui.py`: Flask backend serving APIs, image routes, and the built React frontend
- `frontend/`: React + Vite frontend for the Identity Atlas and Identity Lab
- `templates/`: Legacy server-rendered templates kept as a fallback during migration
- `train_identity_model.py`: Training pipeline stub for identity recognition models
- `run_pipeline.py`: Complete pipeline runner

## Detection Logic

The system uses a combination of computer vision and AI:

### Object Detection (YOLOv8)
- Detects people and common animals (bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe)
- Confidence thresholds: 0.5 for people, 0.4 for animals
- Creates debug annotations with colored bounding boxes:
  - **Blue**: People
  - **Red**: Animals  
  - **Black**: Other detected objects

### Photo Quality Analysis
- **Laplacian Variance**: Measures image sharpness and focus
- **Color Entropy**: Analyzes texture complexity and noise
- Combined heuristics distinguish true photos from digital graphics

## Installation

```bash
# Set up virtual environment
python -m venv .venv

# Activate and install dependencies (Windows)
.venv\Scripts\activate
pip install -r requirements.txt

# Install dependencies (macOS/Linux)
source .venv/bin/activate
pip install -r requirements.txt
```

### Optional GPU Learning Stack

The new blueprint modules are import-safe by default, but the SOTA learning path
needs additional packages that depend on your GPU and vector-store choice.

```bash
# Core active-learning stack
pip install transformers accelerate safetensors

# Qdrant client if you want a persistent vector DB instead of in-memory search
pip install qdrant-client

# Install the GPU build of torch that matches your CUDA runtime
# Example only; pick the correct index URL for your driver/toolkit version.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

The blueprint classes to start from are:
- `MirrorManager` in `src/mirror_manager.py`
- `IntelligenceCore` in `src/intelligence_core.py`

## Usage

### Quick Start (Recommended)

```bash
# 1. Process photos and generate detections
python main.py <input_directory> <output_directory>

# Example:
python main.py ./working_dir ./working_dir

# 2. Build the React frontend
cd frontend
npm install
npm run build
cd ..

# 3. Start the unified app
python web_ui.py ./working_dir ./working_dir
```

### Development Modes

#### Unified Production-Like Mode

```bash
cd frontend
npm run build
cd ..
python web_ui.py ./working_dir ./working_dir
```

Flask serves:
- `/` → React Identity Atlas
- `/lab` → React Identity Lab
- `/api/dashboard` → Atlas JSON data
- `/api/lab` → Lab JSON data

#### Split Development Mode

```bash
# Terminal 1
python web_ui.py ./working_dir ./working_dir

# Terminal 2
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/api`, `/image`, and `/working_dir` requests to Flask.

### Individual Components

```bash
# 1. Process photos with detection and annotations
python main.py <input_directory> <output_directory>

# Example:
python main.py ./working_dir ./sorted_output

# 2. Start the Flask backend separately
python web_ui.py <input_directory> <output_directory>

# 3. Start the React frontend in development
cd frontend
npm run dev

# 4. Run tests
python -m pytest tests/ -v
```

## Output Structure

The processing creates the following directory structure:

```
output_directory/
├── people/                    # Images with people detected
├── animals/                   # Images with animals detected
├── _debug_boxes/              # Annotated images with bounding boxes
├── _training_crops/           # Crops of detected people/animals
│   ├── people/               # Person crops for training
│   └── animals/              # Animal crops for training
├── _detections.json          # Detection metadata (bbox, confidence, class)
├── _labels.json             # Assigned labels and training data
└── [other images]           # Photos without people/animals
```

## Labeling Workflow

PhotoFinder now includes a complete labeling workflow for identity recognition:

### 1. Process Images
```bash
python main.py <input_directory> <output_directory>
```
This generates:
- Detection metadata with bounding boxes
- Training crops for each detected person/animal
- Debug annotations for verification

### 2. Label Detections
Access the React Identity Lab at `/lab` to:
- Review detected people and animals
- Assign names (e.g., "Ron", "Dobby", "Hazel")
- Confirm or reject detections
- Filter by status, class, or assigned label
- Accept likely-name suggestions from the lightweight identity engine
- Build named folders incrementally in `_sorted_by_name/`

### 3. Browse Named Collections
Access the React Identity Atlas at `/` to:
- See named identity cards with progress stages
- Track which identities are warming up or stabilizing
- Browse people, animals, and mixed-scene lanes
- Jump back into the lab to continue strengthening weak identities

### 4. Prepare Training Data
```bash
python train_identity_model.py <output_directory>
```
This creates a structured dataset for model training and provides:
- Dataset organized by assigned labels
- Training statistics and summaries
- Placeholder implementations for face/animal recognition models

## Web UI Categories

The web interface organizes photos into:
- **Working Directory**: Original input images
- **People Only**: Images containing people but no animals
- **Animals Only**: Images containing animals but no people
- **Both**: Images containing both people and animals
- **None**: Photos without detected people or animals

## Security Features

- **Input Validation**: All file paths are validated and sanitized
- **File Type Restrictions**: Only allowed image extensions are processed
- **Path Traversal Protection**: Prevents directory traversal attacks
- **MIME Type Validation**: Ensures only image files are served
- **Size Limits**: Configurable maximum file sizes
- **Safe File Serving**: Uses Flask's `safe_join` for secure file access

## Debug Annotations

The system creates debug images showing:
- Bounding boxes around all detected objects
- Class labels and confidence scores
- Color-coded boxes (blue=people, red=animals, black=others)
- Saved as JPG format in `_debug_boxes/` directory

Example debug filename: `IMG_1234_debug.jpg`

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test class
python -m pytest tests/test_image_processor.py::TestAnnotateDetections -v

# Run with coverage
python -m pytest tests/ --cov=src/
```

The test suite includes:
- Mock-based unit tests (no real model downloads required)
- Color assignment validation
- HEIC compatibility testing
- Edge case handling

## Configuration

### Detection Thresholds

You can modify confidence thresholds in `src/image_processor.py`:

```python
# In detect_subjects() method
if class_name == "person" and confidence > 0.5:  # People threshold
elif class_name in animal_classes and confidence > 0.4:  # Animal threshold

# In annotate_detections() method  
results = self.yolo_model(image, verbose=False, conf=0.10)  # Debug threshold
```

### Supported Image Formats

- **Input**: JPG, JPEG, PNG, BMP, TIFF, HEIC, HEIF
- **Debug Output**: JPG (for compatibility)
- **Web UI**: All supported formats

## Troubleshooting

### Common Issues

1. **YOLO Model Download**: First run downloads YOLOv8 models (~50MB)
2. **HEIC Processing**: Ensure `pillow-heif` is installed
3. **Memory Usage**: Large image collections may require more RAM
4. **Web UI Port**: Default is 5000, change if occupied

### Performance Tips

- Use YOLOv8n (nano) for faster processing
- Process images in batches for large collections
- SSD storage improves I/O performance
- Consider GPU acceleration for very large datasets

## Development

### Adding New Animal Classes

Update the `animal_classes` set in `src/image_processor.py`:

```python
animal_classes = {
    "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "new_animal"
}
```

### Extending the Web UI

The Flask app (`web_ui.py`) uses Jinja2 templates. Modify `templates/index.html` for UI changes.

## License

This project is provided as-is for educational and personal use.
