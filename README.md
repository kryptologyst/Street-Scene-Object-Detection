# Street Scene Object Detection

A comprehensive object detection system for street scenes using YOLOv8. This project provides multiple interfaces (CLI, Web UI, Python API) for detecting vehicles, pedestrians, traffic signs, and other objects in urban environments.

## Features

- **State-of-the-art Detection**: Uses YOLOv8 (You Only Look Once version 8) for fast and accurate object detection
- **Multiple Interfaces**: Command-line interface, Streamlit web app, and Python API
- **Street Scene Optimized**: Focuses on street-relevant objects (vehicles, pedestrians, traffic elements)
- **Configurable**: Adjustable confidence thresholds, model sizes, and display options
- **Batch Processing**: Process multiple images at once
- **Modern Architecture**: Type hints, comprehensive logging, configuration management
- **Easy to Use**: Simple setup and intuitive interfaces

## Requirements

- Python 3.8+
- PyTorch 1.13+
- CUDA (optional, for GPU acceleration)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kryptologyst/Street-Scene-Object-Detection.git
   cd Street-Scene-Object-Detection
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "from src.object_detector import StreetSceneDetector; print('Installation successful!')"
   ```

## Quick Start

### Web Interface (Recommended for beginners)

```bash
streamlit run web_app/app.py
```

Then open your browser to `http://localhost:8501` and:
1. Upload an image or use the sample image
2. Adjust settings in the sidebar
3. Click "Detect Objects"

### Command Line Interface

```bash
# Detect objects in a single image
python cli.py detect --input street_scene.jpg --output results/

# Process sample image
python cli.py sample --output sample_results/

# Batch process multiple images
python cli.py batch --input-dir images/ --output-dir results/
```

### Python API

```python
from src.object_detector import StreetSceneDetector

# Initialize detector
detector = StreetSceneDetector(model_path="yolov8n.pt")

# Detect objects
results = detector.detect_objects("street_scene.jpg")

# Print summary
print(detector.get_detection_summary(results))
```

## Usage Examples

### Basic Detection

```python
from src.object_detector import StreetSceneDetector

# Create detector instance
detector = StreetSceneDetector(
    model_path="yolov8n.pt",  # or yolov8s.pt for better accuracy
    confidence_threshold=0.5
)

# Detect objects in an image
results = detector.detect_objects("path/to/image.jpg")

# Get human-readable summary
summary = detector.get_detection_summary(results)
print(summary)
```

### Custom Configuration

```python
from src.config import config_manager

# Update configuration
config_manager.update_config(
    model_confidence_threshold=0.7,
    model_model_path="yolov8s.pt"
)

# Save configuration
config_manager.save_config("my_config.yaml")
```

### Batch Processing

```python
import os
from pathlib import Path

detector = StreetSceneDetector()

# Process all images in a directory
input_dir = Path("images/")
for image_path in input_dir.glob("*.jpg"):
    results = detector.detect_objects(str(image_path))
    print(f"Processed {image_path.name}: {len(results['detections'])} objects")
```

## Configuration

The system uses YAML configuration files. Key settings:

```yaml
model:
  model_path: "yolov8n.pt"  # Model size (n/s/m/l/x)
  confidence_threshold: 0.5  # Detection confidence
  device: null  # Auto-detect or specify 'cpu'/'cuda'

data:
  input_dir: "data/input"
  output_dir: "data/output"
  supported_formats: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

ui:
  show_confidence: true
  show_labels: true
  max_file_size_mb: 10
```

## Project Structure

```
street-scene-object-detection/
├── src/                    # Source code
│   ├── object_detector.py  # Main detection class
│   └── config.py          # Configuration management
├── web_app/               # Streamlit web interface
│   └── app.py
├── tests/                 # Unit tests
├── data/                  # Data directories
│   ├── input/             # Input images
│   └── output/            # Detection results
├── config/                # Configuration files
│   └── config.yaml
├── models/                # Model files (downloaded automatically)
├── cli.py                 # Command-line interface
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🔧 Advanced Usage

### Custom Model Training

While this project uses pre-trained models, you can integrate custom training:

```python
from ultralytics import YOLO

# Load a custom model
model = YOLO("path/to/custom_model.pt")

# Use with our detector
detector = StreetSceneDetector(model_path="path/to/custom_model.pt")
```

### Performance Optimization

```python
# Use GPU acceleration
detector = StreetSceneDetector(device="cuda")

# Use larger model for better accuracy
detector = StreetSceneDetector(model_path="yolov8l.pt")

# Adjust confidence threshold
detector = StreetSceneDetector(confidence_threshold=0.3)
```

### Integration with Other Systems

```python
# Process video frames
import cv2

cap = cv2.VideoCapture("video.mp4")
detector = StreetSceneDetector()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = detector.detect_objects(frame)
    # Process results...
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_object_detector.py
```

## Model Performance

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv8n | 6.2MB | Fastest | Good | Real-time applications |
| YOLOv8s | 21.5MB | Fast | Better | Balanced performance |
| YOLOv8m | 49.7MB | Medium | Good | High accuracy needs |
| YOLOv8l | 83.7MB | Slow | Better | Maximum accuracy |
| YOLOv8x | 136.7MB | Slowest | Best | Research/offline processing |

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Use a smaller model or CPU
   ```python
   detector = StreetSceneDetector(model_path="yolov8n.pt", device="cpu")
   ```

2. **Model download fails**: Check internet connection or download manually
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   ```

3. **Import errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

### Performance Tips

- Use GPU acceleration when available
- Choose appropriate model size for your needs
- Adjust confidence threshold based on requirements
- Use batch processing for multiple images

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 implementation
- [Streamlit](https://streamlit.io/) for the web interface framework
- [OpenCV](https://opencv.org/) for computer vision utilities

## Future Enhancements

- [ ] Real-time video processing
- [ ] Custom model training pipeline
- [ ] Advanced visualization features
- [ ] REST API interface
- [ ] Docker containerization
- [ ] Mobile app integration
- [ ] Multi-camera support
- [ ] Object tracking capabilities


# Street-Scene-Object-Detection
