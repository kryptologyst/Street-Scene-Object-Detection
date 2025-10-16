#!/usr/bin/env python3
"""
Demo script for the Street Scene Object Detection system.

This script demonstrates the main features of the modernized object detection system.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.object_detector import StreetSceneDetector
from src.synthetic_data import SyntheticStreetSceneGenerator
from src.config import config_manager


def demo_basic_detection():
    """Demonstrate basic object detection functionality."""
    print("=" * 60)
    print("DEMO: Basic Object Detection")
    print("=" * 60)
    
    # Initialize detector
    print("Initializing detector...")
    detector = StreetSceneDetector(
        model_path="yolov8n.pt",
        confidence_threshold=0.5
    )
    print("✓ Detector initialized successfully")
    
    # Download sample image
    print("\nDownloading sample image...")
    try:
        sample_path = detector.download_sample_image("demo_sample.jpg")
        print(f"✓ Sample image downloaded: {sample_path}")
    except Exception as e:
        print(f"✗ Failed to download sample image: {e}")
        return
    
    # Detect objects
    print("\nDetecting objects...")
    try:
        results = detector.detect_objects(sample_path, save_results=True)
        print("✓ Object detection completed")
        
        # Print summary
        print("\nDetection Summary:")
        print("-" * 30)
        print(detector.get_detection_summary(results))
        
    except Exception as e:
        print(f"✗ Detection failed: {e}")


def demo_synthetic_data():
    """Demonstrate synthetic data generation."""
    print("\n" + "=" * 60)
    print("DEMO: Synthetic Data Generation")
    print("=" * 60)
    
    # Create generator
    generator = SyntheticStreetSceneGenerator(width=640, height=480)
    print("✓ Synthetic data generator created")
    
    # Generate single image
    print("\nGenerating synthetic street scene...")
    image, annotations = generator.generate_street_scene(
        objects=['car', 'person', 'bicycle', 'traffic_light'],
        background_type='road'
    )
    
    print(f"✓ Generated image with {len(annotations)} objects")
    print("Objects in scene:")
    for i, ann in enumerate(annotations, 1):
        print(f"  {i}. {ann['class_name']} (confidence: {ann['confidence']:.3f})")
    
    # Save synthetic image
    import cv2
    cv2.imwrite("demo_synthetic.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print("✓ Synthetic image saved as 'demo_synthetic.jpg'")


def demo_configuration():
    """Demonstrate configuration management."""
    print("\n" + "=" * 60)
    print("DEMO: Configuration Management")
    print("=" * 60)
    
    # Show current config
    config = config_manager.get_config()
    print("Current Configuration:")
    print(f"  Model: {config.model.model_path}")
    print(f"  Confidence Threshold: {config.model.confidence_threshold}")
    print(f"  Input Directory: {config.data.input_dir}")
    print(f"  Output Directory: {config.data.output_dir}")
    
    # Update configuration
    print("\nUpdating configuration...")
    config_manager.update_config(
        model_confidence_threshold=0.7,
        ui_show_labels=True
    )
    
    updated_config = config_manager.get_config()
    print("Updated Configuration:")
    print(f"  Confidence Threshold: {updated_config.model.confidence_threshold}")
    print(f"  Show Labels: {updated_config.ui.show_labels}")
    
    # Save configuration
    config_manager.save_config("demo_config.yaml")
    print("✓ Configuration saved to 'demo_config.yaml'")


def demo_cli_interface():
    """Demonstrate CLI interface."""
    print("\n" + "=" * 60)
    print("DEMO: CLI Interface")
    print("=" * 60)
    
    print("The CLI interface provides the following commands:")
    print("  python cli.py detect --input image.jpg --output results/")
    print("  python cli.py sample --output sample_results/")
    print("  python cli.py batch --input-dir images/ --output-dir results/")
    print("  python cli.py config --show")
    
    print("\nTo test the CLI, run:")
    print("  python cli.py sample --output demo_cli_results/")


def demo_web_interface():
    """Demonstrate web interface."""
    print("\n" + "=" * 60)
    print("DEMO: Web Interface")
    print("=" * 60)
    
    print("To launch the web interface, run:")
    print("  streamlit run web_app/app.py")
    print("\nThen open your browser to: http://localhost:8501")
    print("\nFeatures available in the web interface:")
    print("  • Upload images for detection")
    print("  • Use sample images")
    print("  • Adjust confidence threshold")
    print("  • Choose different model sizes")
    print("  • View detection results with visualizations")


def main():
    """Run all demonstrations."""
    print("🚗 Street Scene Object Detection - Demo")
    print("=" * 60)
    print("This demo showcases the modernized object detection system")
    print("with multiple interfaces and advanced features.")
    
    try:
        # Run demonstrations
        demo_basic_detection()
        demo_synthetic_data()
        demo_configuration()
        demo_cli_interface()
        demo_web_interface()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Try the web interface: streamlit run web_app/app.py")
        print("2. Use the CLI: python cli.py sample")
        print("3. Check the generated files:")
        print("   - demo_sample.jpg (sample image)")
        print("   - demo_synthetic.jpg (synthetic image)")
        print("   - demo_config.yaml (configuration)")
        print("   - detections/ (detection results)")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        print("Please check your installation and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
