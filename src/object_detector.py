"""
Modern Object Detection Module for Street Scenes

This module provides a comprehensive object detection system using YOLOv8
with modern Python practices, type hints, and extensible architecture.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreetSceneDetector:
    """
    A modern object detection class for street scenes using YOLOv8.
    
    This class provides comprehensive object detection capabilities with
    support for multiple model sizes, custom datasets, and visualization.
    """
    
    def __init__(
        self, 
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize the StreetSceneDetector.
        
        Args:
            model_path: Path to YOLO model file or model name
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on ('cpu', 'cuda', etc.)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Load the model
        self.model = self._load_model()
        
        # Define street scene relevant classes
        self.street_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus',
            7: 'truck', 9: 'traffic light', 11: 'stop sign', 13: 'bench'
        }
        
        logger.info(f"Initialized StreetSceneDetector with model: {model_path}")
    
    def _load_model(self) -> YOLO:
        """Load the YOLO model with error handling."""
        try:
            model = YOLO(self.model_path)
            if self.device:
                model.to(self.device)
            logger.info(f"Successfully loaded model: {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {self.model_path}: {e}")
            raise
    
    def detect_objects(
        self, 
        image_path: Union[str, Path, np.ndarray],
        save_results: bool = False,
        output_dir: str = "detections"
    ) -> Dict:
        """
        Detect objects in a street scene image.
        
        Args:
            image_path: Path to image file or numpy array
            save_results: Whether to save detection results
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing detection results and metadata
        """
        try:
            # Run inference
            results = self.model(image_path, conf=self.confidence_threshold)
            
            # Extract detection data
            detection_data = self._extract_detection_data(results[0])
            
            # Add metadata
            detection_data.update({
                'model_path': self.model_path,
                'confidence_threshold': self.confidence_threshold,
                'image_path': str(image_path) if isinstance(image_path, (str, Path)) else 'numpy_array'
            })
            
            # Save results if requested
            if save_results:
                self._save_detection_results(results[0], output_dir)
            
            logger.info(f"Detected {len(detection_data['detections'])} objects")
            return detection_data
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise
    
    def _extract_detection_data(self, result) -> Dict:
        """Extract structured data from YOLO results."""
        detections = []
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                detection = {
                    'id': i,
                    'class_id': int(class_id),
                    'class_name': self.model.names[int(class_id)],
                    'confidence': float(conf),
                    'bbox': {
                        'x1': float(box[0]),
                        'y1': float(box[1]),
                        'x2': float(box[2]),
                        'y2': float(box[3])
                    },
                    'is_street_relevant': int(class_id) in self.street_classes
                }
                detections.append(detection)
        
        return {
            'detections': detections,
            'total_detections': len(detections),
            'street_relevant_detections': sum(1 for d in detections if d['is_street_relevant'])
        }
    
    def _save_detection_results(self, result, output_dir: str) -> None:
        """Save detection results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save annotated image
        result.save(output_dir)
        
        # Save detection data as JSON
        import json
        detection_data = self._extract_detection_data(result)
        with open(f"{output_dir}/detection_data.json", 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    def visualize_detections(
        self, 
        image_path: Union[str, Path],
        detection_data: Dict,
        show_labels: bool = True,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Create a visualization of detections on the image.
        
        Args:
            image_path: Path to the original image
            detection_data: Detection results from detect_objects
            show_labels: Whether to show class labels
            show_confidence: Whether to show confidence scores
            
        Returns:
            Annotated image as numpy array
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path.copy()
        
        # Draw bounding boxes
        for detection in detection_data['detections']:
            bbox = detection['bbox']
            x1, y1, x2, y2 = int(bbox['x1']), int(bbox['y1']), int(bbox['x2']), int(bbox['y2'])
            
            # Choose color based on street relevance
            color = (0, 255, 0) if detection['is_street_relevant'] else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            if show_labels:
                label = detection['class_name']
                if show_confidence:
                    label += f" ({detection['confidence']:.2f})"
                
                # Get text size for background
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # Draw background rectangle
                cv2.rectangle(
                    image, (x1, y1 - text_height - 10), 
                    (x1 + text_width, y1), color, -1
                )
                
                # Draw text
                cv2.putText(
                    image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
        
        return image
    
    def download_sample_image(self, output_path: str = "sample_street_scene.jpg") -> str:
        """
        Download a sample street scene image for testing.
        
        Args:
            output_path: Path to save the downloaded image
            
        Returns:
            Path to the downloaded image
        """
        url = 'https://images.pexels.com/photos/167832/pexels-photo-167832.jpeg'
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded sample image to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to download sample image: {e}")
            raise
    
    def get_detection_summary(self, detection_data: Dict) -> str:
        """
        Generate a human-readable summary of detection results.
        
        Args:
            detection_data: Detection results from detect_objects
            
        Returns:
            Formatted summary string
        """
        summary = f"Detection Summary:\n"
        summary += f"Total objects detected: {detection_data['total_detections']}\n"
        summary += f"Street-relevant objects: {detection_data['street_relevant_detections']}\n\n"
        
        # Group by class
        class_counts = {}
        for detection in detection_data['detections']:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        summary += "Objects by class:\n"
        for class_name, count in sorted(class_counts.items()):
            summary += f"  {class_name}: {count}\n"
        
        return summary


def main():
    """Example usage of the StreetSceneDetector."""
    # Initialize detector
    detector = StreetSceneDetector(model_path="yolov8n.pt")
    
    # Download sample image
    image_path = detector.download_sample_image()
    
    # Detect objects
    results = detector.detect_objects(image_path, save_results=True)
    
    # Print summary
    print(detector.get_detection_summary(results))
    
    # Visualize results
    annotated_image = detector.visualize_detections(image_path, results)
    
    # Display results
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.title("Street Scene Object Detection")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
