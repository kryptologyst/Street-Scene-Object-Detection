"""
Command-line interface for Street Scene Object Detection.

This module provides a command-line interface for batch processing
and automated object detection tasks.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

# Import our modules
from src.object_detector import StreetSceneDetector
from src.config import config_manager, config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Street Scene Object Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect objects in a single image
  python cli.py detect --input street_scene.jpg --output results/

  # Batch process multiple images
  python cli.py batch --input-dir images/ --output-dir results/

  # Download and process sample image
  python cli.py sample --output sample_results/

  # Process with custom confidence threshold
  python cli.py detect --input image.jpg --confidence 0.7 --model yolov8s.pt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect objects in a single image')
    detect_parser.add_argument('--input', '-i', required=True, help='Input image path')
    detect_parser.add_argument('--output', '-o', default='detections/', help='Output directory')
    detect_parser.add_argument('--confidence', '-c', type=float, default=0.5, 
                              help='Confidence threshold (0.0-1.0)')
    detect_parser.add_argument('--model', '-m', default='yolov8n.pt', 
                              help='YOLO model path or name')
    detect_parser.add_argument('--device', '-d', help='Device to use (cpu, cuda, etc.)')
    detect_parser.add_argument('--save-image', action='store_true', 
                              help='Save annotated image')
    detect_parser.add_argument('--save-data', action='store_true', 
                              help='Save detection data as JSON')
    detect_parser.add_argument('--verbose', '-v', action='store_true', 
                              help='Verbose output')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch process multiple images')
    batch_parser.add_argument('--input-dir', '-i', required=True, help='Input directory')
    batch_parser.add_argument('--output-dir', '-o', default='batch_results/', 
                            help='Output directory')
    batch_parser.add_argument('--confidence', '-c', type=float, default=0.5,
                            help='Confidence threshold (0.0-1.0)')
    batch_parser.add_argument('--model', '-m', default='yolov8n.pt',
                            help='YOLO model path or name')
    batch_parser.add_argument('--recursive', '-r', action='store_true',
                            help='Process subdirectories recursively')
    batch_parser.add_argument('--extensions', '-e', nargs='+', 
                            default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
                            help='File extensions to process')
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Download and process sample image')
    sample_parser.add_argument('--output', '-o', default='sample_results/', 
                              help='Output directory')
    sample_parser.add_argument('--confidence', '-c', type=float, default=0.5,
                              help='Confidence threshold (0.0-1.0)')
    sample_parser.add_argument('--model', '-m', default='yolov8n.pt',
                              help='YOLO model path or name')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    config_parser.add_argument('--save', help='Save configuration to file')
    
    return parser


def detect_single_image(args) -> None:
    """Detect objects in a single image."""
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    logger.info(f"Initializing detector with model: {args.model}")
    detector = StreetSceneDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        device=args.device
    )
    
    # Process image
    logger.info(f"Processing image: {input_path}")
    try:
        results = detector.detect_objects(
            str(input_path),
            save_results=args.save_image or args.save_data,
            output_dir=str(output_dir)
        )
        
        # Print results
        print("\n" + "="*50)
        print("DETECTION RESULTS")
        print("="*50)
        print(detector.get_detection_summary(results))
        
        if args.verbose:
            print("\nDetailed Results:")
            for detection in results['detections']:
                print(f"  {detection['class_name']}: {detection['confidence']:.3f} "
                      f"[{detection['bbox']['x1']:.0f}, {detection['bbox']['y1']:.0f}, "
                      f"{detection['bbox']['x2']:.0f}, {detection['bbox']['y2']:.0f}]")
        
        # Save additional data if requested
        if args.save_data:
            import json
            data_path = output_dir / "detection_data.json"
            with open(data_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Detection data saved to: {data_path}")
        
        logger.info(f"Processing completed. Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        sys.exit(1)


def batch_process(args) -> None:
    """Batch process multiple images."""
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Find image files
    image_files = []
    if args.recursive:
        for ext in args.extensions:
            image_files.extend(input_dir.rglob(f"*{ext}"))
    else:
        for ext in args.extensions:
            image_files.extend(input_dir.glob(f"*{ext}"))
    
    if not image_files:
        logger.error(f"No image files found in {input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = StreetSceneDetector(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    # Process images
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(image_files, 1):
        logger.info(f"Processing {i}/{len(image_files)}: {image_path.name}")
        
        try:
            # Create subdirectory for this image
            image_output_dir = output_dir / image_path.stem
            image_output_dir.mkdir(exist_ok=True)
            
            # Process image
            results = detector.detect_objects(
                str(image_path),
                save_results=True,
                output_dir=str(image_output_dir)
            )
            
            # Save summary
            summary_path = image_output_dir / "summary.txt"
            with open(summary_path, 'w') as f:
                f.write(detector.get_detection_summary(results))
            
            successful += 1
            
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            failed += 1
    
    # Print batch summary
    print("\n" + "="*50)
    print("BATCH PROCESSING SUMMARY")
    print("="*50)
    print(f"Total images: {len(image_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Results saved to: {output_dir}")


def process_sample(args) -> None:
    """Download and process sample image."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    detector = StreetSceneDetector(
        model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    # Download sample image
    logger.info("Downloading sample image...")
    sample_path = detector.download_sample_image(str(output_dir / "sample_image.jpg"))
    
    # Process image
    logger.info("Processing sample image...")
    results = detector.detect_objects(
        sample_path,
        save_results=True,
        output_dir=str(output_dir)
    )
    
    # Print results
    print("\n" + "="*50)
    print("SAMPLE IMAGE DETECTION RESULTS")
    print("="*50)
    print(detector.get_detection_summary(results))
    
    logger.info(f"Sample processing completed. Results saved to: {output_dir}")


def handle_config(args) -> None:
    """Handle configuration commands."""
    if args.show:
        print("\nCurrent Configuration:")
        print(f"Model Path: {config.model.model_path}")
        print(f"Confidence Threshold: {config.model.confidence_threshold}")
        print(f"Device: {config.model.device}")
        print(f"Input Directory: {config.data.input_dir}")
        print(f"Output Directory: {config.data.output_dir}")
        print(f"Log Level: {config.log_level}")
        print(f"Debug Mode: {config.debug}")
    
    if args.save:
        config_manager.save_config(args.save)
        print(f"Configuration saved to: {args.save}")


def main():
    """Main CLI entry point."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Set logging level
    if hasattr(args, 'verbose') and args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.command == 'detect':
            detect_single_image(args)
        elif args.command == 'batch':
            batch_process(args)
        elif args.command == 'sample':
            process_sample(args)
        elif args.command == 'config':
            handle_config(args)
        else:
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
