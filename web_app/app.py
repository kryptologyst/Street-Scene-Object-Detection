"""
Streamlit web interface for Street Scene Object Detection.

This module provides a user-friendly web interface for the object detection system.
"""

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import our modules
from src.object_detector import StreetSceneDetector
from src.config import config_manager, config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Street Scene Object Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .detection-info {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = None
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None


def create_detector() -> StreetSceneDetector:
    """Create and cache the detector instance."""
    if st.session_state.detector is None:
        with st.spinner("Loading YOLOv8 model..."):
            try:
                st.session_state.detector = StreetSceneDetector(
                    model_path=config.model.model_path,
                    confidence_threshold=config.model.confidence_threshold,
                    device=config.model.device
                )
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                return None
    
    return st.session_state.detector


def display_detection_results(results: dict, image: np.ndarray):
    """Display detection results in a formatted way."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Detection Visualization")
        
        # Create visualization
        annotated_image = st.session_state.detector.visualize_detections(
            image, results, 
            show_labels=config.ui.show_labels,
            show_confidence=config.ui.show_confidence
        )
        
        st.image(annotated_image, caption="Detected Objects", use_column_width=True)
    
    with col2:
        st.subheader("Detection Summary")
        
        # Display metrics
        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.metric("Total Objects", results['total_detections'])
        with col2_2:
            st.metric("Street Relevant", results['street_relevant_detections'])
        
        # Display detailed results
        st.markdown("### Detected Objects")
        
        for i, detection in enumerate(results['detections']):
            with st.expander(f"{detection['class_name']} (Confidence: {detection['confidence']:.2f})"):
                st.write(f"**Class:** {detection['class_name']}")
                st.write(f"**Confidence:** {detection['confidence']:.3f}")
                st.write(f"**Bounding Box:** ({detection['bbox']['x1']:.0f}, {detection['bbox']['y1']:.0f}) to ({detection['bbox']['x2']:.0f}, {detection['bbox']['y2']:.0f})")
                st.write(f"**Street Relevant:** {'Yes' if detection['is_street_relevant'] else 'No'}")


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🚗 Street Scene Object Detection</h1>', unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; font-size: 1.2rem;'>{config.ui.description}</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Model settings
        st.subheader("Model Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=config.model.confidence_threshold,
            step=0.05
        )
        
        model_size = st.selectbox(
            "Model Size",
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
            index=0
        )
        
        # Display settings
        st.subheader("Display Settings")
        show_labels = st.checkbox("Show Labels", value=config.ui.show_labels)
        show_confidence = st.checkbox("Show Confidence", value=config.ui.show_confidence)
        
        # Update config
        config_manager.update_config(
            model_confidence_threshold=confidence_threshold,
            model_model_path=model_size,
            ui_show_labels=show_labels,
            ui_show_confidence=show_confidence
        )
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Upload Image", "Sample Image", "About"])
    
    with tab1:
        st.header("Upload Your Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload a street scene image for object detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert to numpy array for processing
            image_array = np.array(image)
            
            # Create detector
            detector = create_detector()
            
            if detector is not None:
                # Update detector with new confidence threshold
                detector.confidence_threshold = confidence_threshold
                
                # Detect objects
                if st.button("Detect Objects", type="primary"):
                    with st.spinner("Detecting objects..."):
                        try:
                            results = detector.detect_objects(
                                image_array, 
                                save_results=False
                            )
                            st.session_state.detection_results = results
                            st.session_state.uploaded_image = image_array
                            
                            # Display results
                            display_detection_results(results, image_array)
                            
                        except Exception as e:
                            st.error(f"Detection failed: {e}")
                            logger.error(f"Detection error: {e}")
    
    with tab2:
        st.header("Try with Sample Image")
        
        detector = create_detector()
        
        if detector is not None:
            if st.button("Load Sample Street Scene", type="primary"):
                with st.spinner("Downloading sample image..."):
                    try:
                        # Download sample image
                        sample_path = detector.download_sample_image("sample_image.jpg")
                        
                        # Load and display image
                        sample_image = Image.open(sample_path)
                        st.image(sample_image, caption="Sample Street Scene", use_column_width=True)
                        
                        # Convert to numpy array
                        sample_array = np.array(sample_image)
                        
                        # Detect objects
                        with st.spinner("Detecting objects..."):
                            results = detector.detect_objects(
                                sample_array,
                                save_results=True,
                                output_dir="data/output"
                            )
                            
                            # Display results
                            display_detection_results(results, sample_array)
                            
                    except Exception as e:
                        st.error(f"Failed to process sample image: {e}")
                        logger.error(f"Sample image error: {e}")
    
    with tab3:
        st.header("About This Application")
        
        st.markdown("""
        ### Street Scene Object Detection
        
        This application uses **YOLOv8** (You Only Look Once version 8) to detect objects in street scenes.
        
        #### Features:
        - **Real-time object detection** using state-of-the-art YOLOv8 models
        - **Multiple model sizes** from nano (fastest) to extra-large (most accurate)
        - **Street-relevant object filtering** focusing on vehicles, pedestrians, and traffic elements
        - **Interactive visualization** with bounding boxes and confidence scores
        - **Configurable parameters** for confidence threshold and display options
        
        #### Detected Object Classes:
        - **Vehicles**: Cars, trucks, buses, motorcycles, bicycles
        - **People**: Pedestrians
        - **Traffic Elements**: Traffic lights, stop signs
        - **Street Furniture**: Benches, etc.
        
        #### Technical Details:
        - **Model**: YOLOv8 (Ultralytics implementation)
        - **Framework**: PyTorch
        - **Interface**: Streamlit
        - **Visualization**: OpenCV + Matplotlib
        
        #### Usage:
        1. Upload an image or use the sample image
        2. Adjust confidence threshold and model size in the sidebar
        3. Click "Detect Objects" to run the detection
        4. View results with bounding boxes and detailed information
        
        This application is perfect for:
        - **Autonomous vehicle development**
        - **Traffic monitoring systems**
        - **Smart city applications**
        - **Surveillance and security**
        - **Urban planning and analysis**
        """)


if __name__ == "__main__":
    main()
