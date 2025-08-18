#!/usr/bin/env python3
"""
YOLO Object Detection Script
Detects objects in historical photos using pre-trained YOLOv8 models
"""

import cv2
import numpy as np
import json
import random
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

class YOLODetector:
    def __init__(self, model_size='n'):
        """
        Initialize YOLO detector
        model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
        """
        self.model_size = model_size
        self.model = YOLO(f'yolov8{model_size}.pt')
        
        # COCO class names for 20th century European historical photos
        self.relevant_classes = {
            # People and animals - highest priority for historical context
            0: 'person',
            14: 'bird',
            15: 'cat',
            16: 'dog',
            17: 'horse',
            18: 'sheep',
            19: 'cow',
            
            # Transportation - crucial for dating and context
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            6: 'train',
            7: 'truck',
            
            # Military and ceremonial items (20th century Europe)
            # Note: YOLO may detect some as similar objects
            32: 'sports ball',  # May detect cannonballs, military equipment
            77: 'teddy bear',   # May detect uniforms, military gear
            
            # Household and period items
            24: 'backpack',     # Military packs, luggage
            25: 'umbrella',
            26: 'handbag',      # Period bags, satchels
            41: 'cup',          # Period tableware
            43: 'fork',
            44: 'knife',
            46: 'banana',       # Food items in historical context
            47: 'apple',
            56: 'chair',        # Period furniture
            57: 'couch',        # Period furniture
            58: 'potted plant', # Garden/decorative items
            59: 'bed',          # Period furniture
            60: 'dining table', # Period furniture
            73: 'book',         # Documents, reading materials
            74: 'clock',        # Period timepieces
            75: 'vase',         # Decorative items
            76: 'scissors',     # Tools and implements
            
            # Musical and ceremonial
            # Note: YOLO doesn't have specific trumpet/horn classes
            # These objects might be detected as similar items
            33: 'kite',         # May detect flags, banners
        }
        
        # 20th century European historical priority objects
        self.priority_objects = [
            'person',           # People - highest priority for genealogy
            'horse',            # Transportation, military, agriculture
            'train',            # Major transportation method
            'car',              # Automotive development
            'bicycle',          # Common transportation
            'chair',            # Period furniture for dating
            'book',             # Documents, education, culture
            'clock',            # Period timepieces
            'cup',              # Daily life items
            'umbrella',         # Common accessories
            'dog',              # Pets and working animals
            'vase',             # Decorative arts
            'bird'              # Wildlife and domestic
        ]
        
        # Additional context mapping for better historical interpretation
        self.historical_context = {
            'person': 'People in period clothing, military uniforms, formal wear',
            'horse': 'Transportation, military cavalry, agricultural work',
            'train': 'Railway transportation, steam engines, passenger cars',
            'car': 'Early automobiles, period vehicles for dating photos',
            'bicycle': 'Personal transportation, recreational activity',
            'chair': 'Period furniture styles, social gatherings',
            'book': 'Education, literacy, documents, religious texts',
            'clock': 'Timepieces, household items, technology level',
            'cup': 'Daily life, social customs, tableware',
            'umbrella': 'Fashion accessories, weather protection',
            'dog': 'Pets, working dogs, family companions',
            'vase': 'Decorative arts, household ornamentation',
            'bird': 'Wildlife, domestic birds, natural environment'
        }
    
    def detect_objects(self, image_path, confidence_threshold=0.5):
        """Detect objects in a single image"""
        try:
            # Run YOLO detection
            results = self.model(str(image_path), conf=confidence_threshold)
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                        if cls in self.relevant_classes:
                            x1, y1, x2, y2 = box
                            detection = {
                                'class_id': int(cls),
                                'class_name': self.relevant_classes[cls],
                                'confidence': float(conf),
                                'bbox': {
                                    'x1': float(x1),
                                    'y1': float(y1),
                                    'x2': float(x2),
                                    'y2': float(y2),
                                    'width': float(x2 - x1),
                                    'height': float(y2 - y1)
                                }
                            }
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Error detecting objects in {image_path}: {e}")
            return []
    
    def analyze_photo(self, image_path, confidence_threshold=0.5):
        """Analyze a single photo for objects"""
        try:
            detections = self.detect_objects(image_path, confidence_threshold)
            
            # Count objects by class
            object_counts = {}
            for detection in detections:
                class_name = detection['class_name']
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
            # Calculate priority score (higher for historically relevant objects)
            priority_score = 0
            for obj_name, count in object_counts.items():
                if obj_name in self.priority_objects:
                    priority_score += count * (len(self.priority_objects) - self.priority_objects.index(obj_name))
            
            return {
                'filename': Path(image_path).name,
                'total_objects': len(detections),
                'object_counts': object_counts,
                'priority_score': priority_score,
                'detections': detections
            }
            
        except Exception as e:
            print(f"Error analyzing {image_path}: {e}")
            return None
    
    def create_detection_visualization(self, image_path, detections, output_path=None, show_image=False):
        """Create visualization with bounding boxes"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Create matplotlib figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image)
            
            # Add bounding boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(self.relevant_classes)))
            class_colors = {cls_id: colors[i % len(colors)] for i, cls_id in enumerate(self.relevant_classes.keys())}
            
            for detection in detections:
                bbox = detection['bbox']
                class_name = detection['class_name']
                confidence = detection['confidence']
                class_id = detection['class_id']
                
                # Create rectangle
                rect = patches.Rectangle(
                    (bbox['x1'], bbox['y1']),
                    bbox['width'],
                    bbox['height'],
                    linewidth=2,
                    edgecolor=class_colors.get(class_id, 'red'),
                    facecolor='none'
                )
                ax.add_patch(rect)
                
                # Add label
                label = f"{class_name}: {confidence:.2f}"
                ax.text(
                    bbox['x1'],
                    bbox['y1'] - 5,
                    label,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=class_colors.get(class_id, 'red'), alpha=0.7),
                    fontsize=10,
                    color='white',
                    weight='bold'
                )
            
            ax.set_title(f"YOLO Detection - {Path(image_path).name}", fontsize=14, weight='bold')
            ax.axis('off')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            if show_image:
                plt.show()
            else:
                plt.close()
            
            return True
            
        except Exception as e:
            print(f"Error creating visualization for {image_path}: {e}")
            return False
    
    def analyze_directory(self, input_dir='../sample_photos', output_file='yolo_detections.json', 
                         confidence_threshold=0.5, create_visualizations=True):
        """Analyze all photos in a directory"""
        input_dir = Path(input_dir)
        results = []
        
        # Create visualizations directory
        if create_visualizations:
            viz_dir = Path('detection_visualizations')
            viz_dir.mkdir(exist_ok=True)
        
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_paths = []
        
        # Find all image files
        for ext in extensions:
            image_paths.extend(list(input_dir.glob(f'*{ext}')))
            image_paths.extend(list(input_dir.glob(f'*{ext.upper()}')))
        
        print(f"Found {len(image_paths)} images to analyze with YOLO")
        print(f"Using YOLOv8{self.model_size} model")
        
        # Analyze each image
        for i, image_path in enumerate(image_paths):
            print(f"Analyzing {i+1}/{len(image_paths)}: {image_path.name}")
            
            result = self.analyze_photo(image_path, confidence_threshold)
            if result:
                results.append(result)
                
                # Create visualization if requested
                if create_visualizations and result['total_objects'] > 0:
                    viz_path = viz_dir / f"{Path(image_path).stem}_detections.png"
                    self.create_detection_visualization(image_path, result['detections'], viz_path)
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        # Save results to JSON
        output_data = {
            'model_info': {
                'model_name': f'YOLOv8{self.model_size}',
                'confidence_threshold': confidence_threshold,
                'total_photos_analyzed': len(results)
            },
            'summary': summary,
            'photos': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nYOLO analysis complete! Results saved to {output_file}")
        if create_visualizations:
            print(f"Visualizations saved to detection_visualizations/ directory")
        
        return results
    
    def _calculate_summary(self, results):
        """Calculate summary statistics"""
        if not results:
            return {}
        
        # Count all object types
        all_objects = {}
        photos_with_objects = 0
        total_objects = 0
        
        for result in results:
            if result['total_objects'] > 0:
                photos_with_objects += 1
                total_objects += result['total_objects']
                
                for obj_name, count in result['object_counts'].items():
                    all_objects[obj_name] = all_objects.get(obj_name, 0) + count
        
        # Sort objects by frequency
        sorted_objects = dict(sorted(all_objects.items(), key=lambda x: x[1], reverse=True))
        
        # Find most common objects
        top_objects = dict(list(sorted_objects.items())[:10])
        
        return {
            'photos_with_objects': photos_with_objects,
            'photos_without_objects': len(results) - photos_with_objects,
            'total_objects_detected': total_objects,
            'average_objects_per_photo': round(total_objects / len(results), 2),
            'unique_object_types': len(all_objects),
            'most_common_objects': top_objects,
            'all_object_counts': sorted_objects
        }
    
    def analyze_single_photo(self, image_path, confidence_threshold=0.5, show_image=True, save_result=True):
        """Analyze a single photo and optionally display it"""
        print(f"Analyzing single photo: {image_path}")
        
        # Check if file exists
        if not Path(image_path).exists():
            print(f"Error: File not found - {image_path}")
            return None
        
        # Analyze the photo
        result = self.analyze_photo(image_path, confidence_threshold)
        if not result:
            print("Failed to analyze photo")
            return None
        
        # Print results
        print(f"\n{'='*50}")
        print(f"ANALYSIS RESULTS: {result['filename']}")
        print(f"{'='*50}")
        print(f"Total objects detected: {result['total_objects']}")
        print(f"Priority score: {result['priority_score']}")
        
        if result['object_counts']:
            print(f"Objects found:")
            for obj_name, count in result['object_counts'].items():
                context = self.historical_context.get(obj_name, 'General object')
                print(f"  {obj_name}: {count} - {context}")
        else:
            print("No objects detected")
        
        # Show image if requested
        if show_image and result['detections']:
            print(f"\nDisplaying image with detections...")
            self.create_detection_visualization(image_path, result['detections'], show_image=True)
        elif show_image:
            # Show original image without detections
            img = Image.open(image_path)
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.title(f"No objects detected - {Path(image_path).name}")
            plt.axis('off')
            plt.show()
        
        # Save individual result if requested
        if save_result:
            output_file = f"single_photo_result_{Path(image_path).stem}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    'model_info': {
                        'model_name': f'YOLOv8{self.model_size}',
                        'confidence_threshold': confidence_threshold
                    },
                    'photo': result
                }, f, indent=2)
            print(f"Result saved to: {output_file}")
        
        return result
    
    def analyze_random_photos(self, input_dir='../sample_photos', num_photos=5, 
                            confidence_threshold=0.5, show_images=True):
        """Analyze random photos from directory"""
        input_dir = Path(input_dir)
        
        # Find all image files
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_paths = []
        
        for ext in extensions:
            image_paths.extend(list(input_dir.glob(f'*{ext}')))
            image_paths.extend(list(input_dir.glob(f'*{ext.upper()}')))
        
        if len(image_paths) == 0:
            print(f"No images found in {input_dir}")
            return []
        
        # Select random photos
        num_photos = min(num_photos, len(image_paths))
        selected_photos = random.sample(image_paths, num_photos)
        
        print(f"Analyzing {num_photos} random photos from {len(image_paths)} total images")
        
        results = []
        for i, image_path in enumerate(selected_photos, 1):
            print(f"\n--- Random Photo {i}/{num_photos} ---")
            result = self.analyze_single_photo(image_path, confidence_threshold, 
                                             show_image=show_images, save_result=False)
            if result:
                results.append(result)
        
        # Save combined results
        output_file = f"random_{num_photos}_photos_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'model_info': {
                    'model_name': f'YOLOv8{self.model_size}',
                    'confidence_threshold': confidence_threshold,
                    'num_photos': len(results)
                },
                'photos': results
            }, f, indent=2)
        
        print(f"\nRandom analysis complete! Results saved to: {output_file}")
        return results
    
    def interactive_mode(self):
        """Interactive mode for choosing analysis type"""
        print("\n" + "="*60)
        print("YOLO OBJECT DETECTION - INTERACTIVE MODE")
        print("="*60)
        print("Choose analysis mode:")
        print("1. Analyze all photos in directory")
        print("2. Analyze 5 random photos") 
        print("3. Analyze specific photo (enter path)")
        print("4. Exit")
        
        while True:
            try:
                choice = input("\nEnter choice (1-4): ").strip()
                
                if choice == '1':
                    print("\nAnalyzing all photos...")
                    self.analyze_directory()
                    break
                
                elif choice == '2':
                    print("\nAnalyzing 5 random photos...")
                    self.analyze_random_photos(num_photos=5, show_images=True)
                    break
                
                elif choice == '3':
                    photo_path = input("Enter photo path: ").strip()
                    if photo_path:
                        self.analyze_single_photo(photo_path, show_image=True)
                    else:
                        print("Invalid path")
                    break
                
                elif choice == '4':
                    print("Exiting...")
                    break
                
                else:
                    print("Invalid choice. Please enter 1-4.")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                break

def main():
    # Default settings
    model_size = 'n'  # nano model for faster inference
    
    print("Initializing YOLO detector...")
    detector = YOLODetector(model_size=model_size)
    
    print(f"Relevant object classes ({len(detector.relevant_classes)}):")
    for class_name in list(detector.relevant_classes.values())[:10]:
        print(f"  {class_name}")
    print("  ... and more")
    
    # Run interactive mode
    detector.interactive_mode()

if __name__ == "__main__":
    main()