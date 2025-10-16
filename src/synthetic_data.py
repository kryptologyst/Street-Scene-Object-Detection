"""
Synthetic data generator for testing and demonstration purposes.

This module creates synthetic street scene images with known objects
for testing the object detection system.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class SyntheticStreetSceneGenerator:
    """
    Generate synthetic street scene images for testing and demonstration.
    
    This class creates realistic-looking street scenes with various objects
    that can be used to test the object detection system.
    """
    
    def __init__(self, width: int = 640, height: int = 480):
        """
        Initialize the synthetic data generator.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
        """
        self.width = width
        self.height = height
        
        # Define colors for different objects
        self.colors = {
            'car': (0, 0, 255),      # Red
            'truck': (0, 100, 200),   # Blue
            'bus': (0, 200, 100),     # Green
            'person': (255, 0, 0),    # Blue
            'bicycle': (255, 255, 0), # Yellow
            'motorcycle': (255, 0, 255), # Magenta
            'traffic_light': (0, 255, 255), # Cyan
            'stop_sign': (128, 0, 128), # Purple
        }
        
        # Object templates (simple shapes)
        self.templates = self._create_object_templates()
    
    def _create_object_templates(self) -> dict:
        """Create simple object templates."""
        templates = {}
        
        # Car template
        car_template = np.zeros((40, 80, 3), dtype=np.uint8)
        cv2.rectangle(car_template, (5, 10), (75, 30), self.colors['car'], -1)
        cv2.rectangle(car_template, (10, 5), (70, 35), self.colors['car'], -1)
        templates['car'] = car_template
        
        # Truck template
        truck_template = np.zeros((50, 100, 3), dtype=np.uint8)
        cv2.rectangle(truck_template, (5, 15), (95, 35), self.colors['truck'], -1)
        cv2.rectangle(truck_template, (10, 10), (90, 40), self.colors['truck'], -1)
        templates['truck'] = truck_template
        
        # Person template
        person_template = np.zeros((60, 20, 3), dtype=np.uint8)
        cv2.circle(person_template, (10, 10), 8, self.colors['person'], -1)  # Head
        cv2.rectangle(person_template, (8, 18), (12, 50), self.colors['person'], -1)  # Body
        cv2.rectangle(person_template, (5, 20), (8, 35), self.colors['person'], -1)   # Left arm
        cv2.rectangle(person_template, (12, 20), (15, 35), self.colors['person'], -1) # Right arm
        cv2.rectangle(person_template, (8, 50), (10, 60), self.colors['person'], -1) # Left leg
        cv2.rectangle(person_template, (10, 50), (12, 60), self.colors['person'], -1) # Right leg
        templates['person'] = person_template
        
        # Bicycle template
        bike_template = np.zeros((30, 60, 3), dtype=np.uint8)
        cv2.circle(bike_template, (15, 15), 12, self.colors['bicycle'], 2)  # Front wheel
        cv2.circle(bike_template, (45, 15), 12, self.colors['bicycle'], 2)  # Back wheel
        cv2.line(bike_template, (15, 15), (45, 15), self.colors['bicycle'], 2)  # Frame
        cv2.line(bike_template, (15, 15), (30, 5), self.colors['bicycle'], 2)   # Handlebar
        templates['bicycle'] = bike_template
        
        # Traffic light template
        light_template = np.zeros((80, 30, 3), dtype=np.uint8)
        cv2.rectangle(light_template, (10, 5), (20, 75), (100, 100, 100), -1)  # Pole
        cv2.circle(light_template, (15, 20), 8, (0, 255, 0), -1)  # Green light
        cv2.circle(light_template, (15, 35), 8, (255, 255, 0), -1)  # Yellow light
        cv2.circle(light_template, (15, 50), 8, (255, 0, 0), -1)  # Red light
        templates['traffic_light'] = light_template
        
        return templates
    
    def generate_street_scene(
        self, 
        objects: List[str] = None,
        background_type: str = "road"
    ) -> Tuple[np.ndarray, List[dict]]:
        """
        Generate a synthetic street scene image.
        
        Args:
            objects: List of object types to include
            background_type: Type of background ("road", "park", "city")
            
        Returns:
            Tuple of (image, object_annotations)
        """
        # Create background
        image = self._create_background(background_type)
        
        # Default objects if none specified
        if objects is None:
            objects = ['car', 'person', 'bicycle', 'traffic_light']
        
        annotations = []
        
        # Add objects to the scene
        for obj_type in objects:
            if obj_type in self.templates:
                template = self.templates[obj_type]
                
                # Random position (avoid edges)
                x = random.randint(20, self.width - template.shape[1] - 20)
                y = random.randint(20, self.height - template.shape[0] - 20)
                
                # Place object on image
                self._place_object(image, template, x, y)
                
                # Create annotation
                annotation = {
                    'class_name': obj_type,
                    'bbox': [x, y, x + template.shape[1], y + template.shape[0]],
                    'confidence': random.uniform(0.7, 0.95)
                }
                annotations.append(annotation)
        
        return image, annotations
    
    def _create_background(self, background_type: str) -> np.ndarray:
        """Create background for the street scene."""
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if background_type == "road":
            # Road scene
            image[:, :] = (50, 50, 50)  # Dark gray road
            
            # Add road markings
            cv2.line(image, (0, self.height//2), (self.width, self.height//2), (255, 255, 255), 2)
            
            # Add sidewalk
            cv2.rectangle(image, (0, 0), (self.width, self.height//4), (100, 100, 100), -1)
            cv2.rectangle(image, (0, 3*self.height//4), (self.width, self.height), (100, 100, 100), -1)
            
            # Add sky
            cv2.rectangle(image, (0, 0), (self.width, self.height//6), (135, 206, 235), -1)
            
        elif background_type == "park":
            # Park scene
            image[:, :] = (34, 139, 34)  # Forest green
            
            # Add path
            cv2.rectangle(image, (self.width//4, 0), (3*self.width//4, self.height), (139, 69, 19), -1)
            
        elif background_type == "city":
            # City scene
            image[:, :] = (70, 70, 70)  # Dark gray
            
            # Add buildings in background
            for i in range(3):
                x = i * self.width // 3
                height = random.randint(self.height//3, self.height//2)
                cv2.rectangle(image, (x, self.height-height), (x + self.width//3, self.height), (50, 50, 50), -1)
        
        return image
    
    def _place_object(self, image: np.ndarray, template: np.ndarray, x: int, y: int) -> None:
        """Place an object template on the image."""
        h, w = template.shape[:2]
        
        # Ensure object fits within image bounds
        if x + w > self.width or y + h > self.height:
            return
        
        # Blend object with background
        roi = image[y:y+h, x:x+w]
        mask = template > 0
        
        # Simple blending
        image[y:y+h, x:x+w][mask] = template[mask]
    
    def generate_dataset(
        self, 
        num_images: int = 10,
        output_dir: str = "data/synthetic",
        background_types: List[str] = None
    ) -> None:
        """
        Generate a dataset of synthetic street scenes.
        
        Args:
            num_images: Number of images to generate
            output_dir: Directory to save images and annotations
            background_types: List of background types to use
        """
        if background_types is None:
            background_types = ["road", "park", "city"]
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        images_dir = output_path / "images"
        annotations_dir = output_path / "annotations"
        images_dir.mkdir(exist_ok=True)
        annotations_dir.mkdir(exist_ok=True)
        
        # Available objects
        all_objects = list(self.templates.keys())
        
        for i in range(num_images):
            # Random selection of objects
            num_objects = random.randint(2, min(5, len(all_objects)))
            selected_objects = random.sample(all_objects, num_objects)
            
            # Random background
            background = random.choice(background_types)
            
            # Generate image
            image, annotations = self.generate_street_scene(selected_objects, background)
            
            # Save image
            image_filename = f"synthetic_scene_{i:03d}.jpg"
            image_path = images_dir / image_filename
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # Save annotations
            annotation_filename = f"synthetic_scene_{i:03d}.txt"
            annotation_path = annotations_dir / annotation_filename
            
            with open(annotation_path, 'w') as f:
                f.write(f"# Synthetic street scene {i}\n")
                f.write(f"# Background: {background}\n")
                f.write(f"# Objects: {', '.join(selected_objects)}\n\n")
                
                for j, ann in enumerate(annotations):
                    f.write(f"Object {j+1}:\n")
                    f.write(f"  Class: {ann['class_name']}\n")
                    f.write(f"  Bbox: {ann['bbox']}\n")
                    f.write(f"  Confidence: {ann['confidence']:.3f}\n\n")
        
        print(f"Generated {num_images} synthetic images in {output_dir}")
        print(f"Images saved to: {images_dir}")
        print(f"Annotations saved to: {annotations_dir}")


def main():
    """Example usage of the synthetic data generator."""
    generator = SyntheticStreetSceneGenerator(width=800, height=600)
    
    # Generate a single image
    image, annotations = generator.generate_street_scene(
        objects=['car', 'person', 'bicycle', 'traffic_light'],
        background_type='road'
    )
    
    # Save the image
    cv2.imwrite('synthetic_street_scene.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    
    # Print annotations
    print("Generated synthetic street scene with annotations:")
    for i, ann in enumerate(annotations):
        print(f"Object {i+1}: {ann['class_name']} at {ann['bbox']} (confidence: {ann['confidence']:.3f})")
    
    # Generate a small dataset
    generator.generate_dataset(
        num_images=5,
        output_dir="data/synthetic",
        background_types=["road", "city"]
    )


if __name__ == "__main__":
    main()
