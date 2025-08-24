#!/usr/bin/env python3
"""
YOLO Object Detection Script
Detects objects in historical photos using pre-trained YOLOv8 models
"""

from typing import Any, cast
import cv2 as cv2_module  # type: ignore
cv2 = cast(Any, cv2_module)
import numpy as np
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageOps

class YOLODetector:
    def __init__(self, model_size='s', imgsz=1280, base_confidence=0.4, iou=0.65, max_det=300,
                 enable_preprocessing=True, augment=False, min_box_area_ratio=0.003,
                 enable_opencv_aux=True, aux_fast_mode=True, hog_only_if_no_yolo_person=True):
        """
        Initialize YOLO detector
        model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
        """
        self.model_size = model_size
        # Lazy import to avoid linter/env errors when ultralytics isn't installed
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Ultralytics YOLO is required. Install with 'pip install ultralytics' ") from exc
        self.model = YOLO(f'yolov8{model_size}.pt')
        # Prefer Apple MPS or CUDA when available for speed
        try:
            import torch  # type: ignore
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
            # Move model
            try:
                self.model.to(self.device)
            except Exception:
                pass
        except Exception:
            self.device = 'cpu'
        
        # Inference configuration (tuned for M1 8GB friendly performance)
        self.imgsz = imgsz
        self.global_confidence = base_confidence
        self.iou = iou
        self.max_det = max_det
        self.augment = augment
        self.enable_preprocessing = enable_preprocessing
        self.min_box_area_ratio = min_box_area_ratio
        self.enable_opencv_aux = enable_opencv_aux
        self.aux_fast_mode = aux_fast_mode
        self.hog_only_if_no_yolo_person = hog_only_if_no_yolo_person
        
        # Per-class confidence thresholds (post-filter). Lower for 'person' to boost recall
        self.per_class_conf_thresholds = {
            'person': 0.35
        }

        # OpenCV auxiliary detectors (faces, cats, HOG people) if enabled
        if self.enable_opencv_aux:
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            except Exception:
                self.face_cascade = cv2.CascadeClassifier()
            try:
                self.cat_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalcatface.xml')
            except Exception:
                self.cat_cascade = cv2.CascadeClassifier()
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
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
            'person': 'People',
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
    
    def _preprocess_image(self, image_path):
        """Load image, fix EXIF orientation, apply CLAHE and mild sharpening. Returns BGR np.array."""
        img = Image.open(image_path)
        # Respect EXIF orientation
        img = ImageOps.exif_transpose(img)
        img = img.convert('RGB')
        img_np = np.array(img)  # RGB
        
        # Apply CLAHE on luminance channel (LAB) only if needed for speed
        try:
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            if np.std(gray) < 45:
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_eq = clahe.apply(l)
                lab_eq = cv2.merge((l_eq, a, b))
                img_np = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
        except Exception:
            # Fallback without failing
            pass
        
        # Mild unsharp mask for edge enhancement
        try:
            blurred = cv2.GaussianBlur(img_np, (0, 0), sigmaX=0.8)
            img_np = cv2.addWeighted(img_np, 1.25, blurred, -0.25, 0)
        except Exception:
            pass
        
        # Convert to BGR for YOLO OpenCV pipeline
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_bgr

    # --------------- OpenCV auxiliary methods ---------------
    def _opencv_count_faces(self, image_bgr):
        if not self.enable_opencv_aux or self.face_cascade.empty():
            return 0, []
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
        )
        return len(faces), faces.tolist() if len(faces) > 0 else []

    def _opencv_detect_cat_faces(self, image_bgr):
        if not self.enable_opencv_aux or self.cat_cascade.empty():
            return 0, []
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        cats = self.cat_cascade.detectMultiScale(
            gray, scaleFactor=1.02, minNeighbors=3, minSize=(24, 24), flags=cv2.CASCADE_SCALE_IMAGE
        )
        return len(cats), cats.tolist() if len(cats) > 0 else []

    def _opencv_detect_people_hog(self, image_bgr, max_side=640):
        if not self.enable_opencv_aux:
            return 0, []
        h, w = image_bgr.shape[:2]
        scale = 1.0
        if max(h, w) > max_side:
            scale = max_side / float(max(h, w))
            resized = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image_bgr
        rects, weights = self.hog.detectMultiScale(resized, winStride=(8, 8), padding=(8, 8), scale=1.08)
        rects = np.array(rects)
        boxes = []
        if len(rects) > 0:
            inv = 1.0 / scale
            for (x, y, bw, bh) in rects:
                boxes.append([int(x * inv), int(y * inv), int(bw * inv), int(bh * inv)])
        return len(boxes), boxes

    def _opencv_detect_lines(self, image_bgr, canny_low=50, canny_high=150, min_line_length_ratio=0.05, max_line_gap=10):
        if not self.enable_opencv_aux:
            return [], []
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, canny_low, canny_high)
        h, w = gray.shape[:2]
        min_len = int(min(h, w) * min_line_length_ratio)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=min_len, maxLineGap=max_line_gap)
        line_list = []
        angles = []
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                line_list.append([int(x1), int(y1), int(x2), int(y2)])
                angle = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
                angles.append(float(angle))
        return line_list, angles

    def _opencv_detect_vegetation_and_trees(self, image_bgr):
        if not self.enable_opencv_aux:
            return {'green_ratio': 0.0, 'vegetation_present': False, 'trees_present': False}
        h, w = image_bgr.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        green_mask = ((hue >= 35) & (hue <= 85) & (sat > 40) & (val > 40))
        green_ratio = float(np.sum(green_mask)) / float(max(1, h * w))
        vegetation_present = green_ratio > 0.10
        # Trees: look for vertical lines overlapping green areas
        lines, angles = self._opencv_detect_lines(image_bgr)
        verticals = []
        for (x1, y1, x2, y2), a in zip(lines, angles):
            if abs(abs(a) - 90) < 12:
                # Sample mid-point color to see if near green area
                mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if 0 <= my < h and 0 <= mx < w and green_mask[my, mx]:
                    verticals.append((x1, y1, x2, y2))
        trees_present = vegetation_present and len(verticals) >= 2
        return {'green_ratio': round(green_ratio, 4), 'vegetation_present': vegetation_present, 'trees_present': trees_present}

    def _opencv_detect_building_structure(self, image_bgr):
        if not self.enable_opencv_aux:
            return {'building_present': False, 'num_lines': 0, 'rectilinear_ratio': 0.0}
        lines, angles = self._opencv_detect_lines(image_bgr)
        num_lines = len(lines)
        if num_lines == 0:
            return {'building_present': False, 'num_lines': 0, 'rectilinear_ratio': 0.0}
        rectilinear = sum(1 for a in angles if (abs(a) < 10) or (abs(abs(a) - 90) < 10))
        rect_ratio = rectilinear / float(num_lines)
        building_present = num_lines >= 12 and rect_ratio > 0.5
        return {'building_present': building_present, 'num_lines': num_lines, 'rectilinear_ratio': round(rect_ratio, 3)}

    def _opencv_detect_frame_like(self, image_bgr):
        if not self.enable_opencv_aux:
            return {'frame_like_present': False, 'frames': []}
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape[:2]
        frames = []
        for cnt in contours:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                x, y, bw, bh = cv2.boundingRect(approx)
                area = bw * bh
                if area < 0.02 * h * w or area > 0.75 * h * w:
                    continue
                # Aspect sanity and near-rectangle shape
                aspect = max(bw, bh) / max(1, min(bw, bh))
                if aspect > 6.0:
                    continue
                frames.append([int(x), int(y), int(bw), int(bh)])
        return {'frame_like_present': len(frames) > 0, 'frames': frames[:10]}


    def detect_objects(self, image_path, confidence_threshold=None):
        """Detect objects in a single image"""
        try:
            # Prepare image (optional preprocessing, EXIF-aware)
            if self.enable_preprocessing:
                image_bgr = self._preprocess_image(image_path)
            else:
                # Load raw image via OpenCV (no EXIF auto-rotation)
                image_bgr = cv2.imread(str(image_path))
                if image_bgr is None:
                    # Fallback to PIL if OpenCV fails
                    image_bgr = self._preprocess_image(image_path)
            
            img_h, img_w = image_bgr.shape[:2]
            img_area = float(img_h * img_w)
            min_area = self.min_box_area_ratio * img_area
            
            # Effective base confidence for model call (collect more boxes, filter later per-class)
            effective_conf = self.global_confidence if confidence_threshold is None else confidence_threshold
            
            # Run YOLO detection with tuned inference params
            results = self.model(
                image_bgr,
                conf=effective_conf,
                iou=self.iou,
                imgsz=self.imgsz,
                classes=list(self.relevant_classes.keys()),
                max_det=self.max_det,
                augment=self.augment
            )
            
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
                            width = float(x2 - x1)
                            height = float(y2 - y1)
                            box_area = width * height
                            class_name = self.relevant_classes[cls]
                            
                            # Post-filter: per-class confidence and minimum area ratio
                            per_cls_thr = self.per_class_conf_thresholds.get(class_name, effective_conf)
                            if float(conf) < per_cls_thr:
                                continue
                            if box_area < min_area:
                                continue
                            
                            detection = {
                                'class_id': int(cls),
                                'class_name': class_name,
                                'confidence': float(conf),
                                'bbox': {
                                    'x1': float(x1),
                                    'y1': float(y1),
                                    'x2': float(x2),
                                    'y2': float(y2),
                                    'width': width,
                                    'height': height
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
            opencv_aux = None
            if self.enable_opencv_aux:
                # Use preprocessed image to be EXIF-correct
                try:
                    image_bgr = self._preprocess_image(image_path)
                    # Faces and cats
                    face_count, face_locations = self._opencv_count_faces(image_bgr)
                    cat_count, cat_locations = self._opencv_detect_cat_faces(image_bgr)
                    # HOG people
                    people_count, people_boxes = self._opencv_detect_people_hog(image_bgr)
                    # Lines
                    lines, angles = self._opencv_detect_lines(image_bgr)
                    # Heuristic cues: vegetation/trees, buildings, frames
                    vegetation = self._opencv_detect_vegetation_and_trees(image_bgr)
                    building = self._opencv_detect_building_structure(image_bgr)
                    frames = self._opencv_detect_frame_like(image_bgr)
                    opencv_aux = {
                        'face_count': int(face_count),
                        'face_locations': face_locations,
                        'cat_face_count': int(cat_count),
                        'cat_face_locations': cat_locations,
                        'people_hog_count': int(people_count),
                        'people_hog_boxes': people_boxes,
                        'lines_count': len(lines),
                        'lines': lines[:500],
                        'line_angles': [float(a) for a in angles[:500]],
                        'vegetation': vegetation,
                        'building': building,
                        'frames': frames
                    }
                except Exception:
                    opencv_aux = None
            
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
                'detections': detections,
                **({'opencv_aux': opencv_aux} if opencv_aux is not None else {})
            }
            
        except Exception as e:
            print(f"Error analyzing {image_path}: {e}")
            return None
    
    def create_detection_visualization(self, image_path, detections, output_path=None, show_image=False,
                                       opencv_aux=None):
        """Create visualization with bounding boxes"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Create matplotlib figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image)
            
            # Add bounding boxes
            set3 = plt.cm.get_cmap('Set3', 12)
            colors = [set3(i) for i in range(12)]
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
            
            # Overlay OpenCV auxiliary detections if provided
            if opencv_aux:
                # Faces (blue), HOG people (green), Cats (orange), Lines (red)
                # Draw with matplotlib patches for consistency
                for (x, y, w, h) in opencv_aux.get('faces', []):
                    rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='blue', facecolor='none', linestyle='--')
                    ax.add_patch(rect)
                for (x, y, w, h) in opencv_aux.get('people_hog', []):
                    rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='green', facecolor='none', linestyle=':')
                    ax.add_patch(rect)
                for (x, y, w, h) in opencv_aux.get('cat_faces', []):
                    rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor='orange', facecolor='none', linestyle='--')
                    ax.add_patch(rect)
                for (x1, y1, x2, y2) in opencv_aux.get('lines', [])[:200]:
                    ax.plot([x1, x2], [y1, y2], color='red', linewidth=0.6, alpha=0.7)
            # Title
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
                         confidence_threshold=None, create_visualizations=True):
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
        print(f"Using YOLOv8{self.model_size} model on device={getattr(self, 'device', 'cpu')}")
        print(f"Inference params: imgsz={self.imgsz}, conf={self.global_confidence if confidence_threshold is None else confidence_threshold}, iou={self.iou}, max_det={self.max_det}, preprocess={self.enable_preprocessing}, aux_fast_mode={self.aux_fast_mode}")
        
        # Analyze each image
        for i, image_path in enumerate(image_paths):
            print(f"Analyzing {i+1}/{len(image_paths)}: {image_path.name}")
            
            result = self.analyze_photo(image_path, confidence_threshold if confidence_threshold is not None else self.global_confidence)
            if result:
                results.append(result)
                
                # Create visualization if requested
                if create_visualizations and result['total_objects'] > 0:
                    viz_path = viz_dir / f"{Path(image_path).stem}_detections.png"
                    opencv_aux = None
                    if self.enable_opencv_aux and 'opencv_aux' in result:
                        aux = result['opencv_aux']
                        opencv_aux = {
                            'faces': aux.get('face_locations', []),
                            'people_hog': aux.get('people_hog_boxes', []),
                            'cat_faces': aux.get('cat_face_locations', []),
                            'lines': aux.get('lines', []),
                            'vegetation': aux.get('vegetation', {}),
                            'building': aux.get('building', {}),
                            'frames': aux.get('frames', {})
                        }
                    self.create_detection_visualization(image_path, result['detections'], viz_path, opencv_aux=opencv_aux)
        
        # Calculate summary statistics
        summary = self._calculate_summary(results)
        
        # Save results to JSON
        effective_conf = self.global_confidence if confidence_threshold is None else confidence_threshold
        output_data = {
            'model_info': {
                'model_name': f'YOLOv8{self.model_size}',
                'confidence_threshold': effective_conf,
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
    
    def analyze_single_photo(self, image_path, confidence_threshold=None, show_image=True, save_result=True):
        """Analyze a single photo and optionally display it"""
        print(f"Analyzing single photo: {image_path}")
        
        # Check if file exists
        if not Path(image_path).exists():
            print(f"Error: File not found - {image_path}")
            return None
        
        # Analyze the photo
        result = self.analyze_photo(image_path, confidence_threshold if confidence_threshold is not None else self.global_confidence)
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
            opencv_aux = result.get('opencv_aux') if isinstance(result, dict) else None
            # Prepare aux overlays for visualization when available
            aux_for_viz = None
            if opencv_aux:
                aux_for_viz = {
                    'faces': opencv_aux.get('face_locations', []),
                    'people_hog': opencv_aux.get('people_hog_boxes', []),
                    'cat_faces': opencv_aux.get('cat_face_locations', []),
                    'lines': opencv_aux.get('lines', []),
                    'vegetation': opencv_aux.get('vegetation', {}),
                    'building': opencv_aux.get('building', {}),
                    'frames': opencv_aux.get('frames', {})
                }
            self.create_detection_visualization(image_path, result['detections'], show_image=True, opencv_aux=aux_for_viz)
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
                        'confidence_threshold': confidence_threshold if confidence_threshold is not None else self.global_confidence
                    },
                    'photo': result
                }, f, indent=2)
            print(f"Result saved to: {output_file}")
        
        return result
    
    def analyze_random_photos(self, input_dir='../sample_photos', num_photos=5, 
                            confidence_threshold=None, show_images=True):
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
            result = self.analyze_single_photo(
                image_path,
                confidence_threshold if confidence_threshold is not None else self.global_confidence,
                show_image=show_images,
                save_result=False
            )
            if result:
                results.append(result)
        
        # Save combined results
        output_file = f"random_{num_photos}_photos_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'model_info': {
                    'model_name': f'YOLOv8{self.model_size}',
                    'confidence_threshold': confidence_threshold if confidence_threshold is not None else self.global_confidence,
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
    model_size = 's'  # small model for better accuracy on M1 8GB
    
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