#!/usr/bin/env python3
"""
Photo Processing Script for Feature Extraction
Extracts features from historical photos for unsupervised learning
"""

import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

class PhotoProcessor:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.scaler = StandardScaler()
        
    def load_image(self, image_path):
        """Load and preprocess image"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not load {image_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, self.image_size)
            
            return image
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def extract_color_features(self, image):
        """Extract color histogram features"""
        # Calculate histograms for each channel
        hist_r = cv2.calcHist([image], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [32], [0, 256])
        
        # Flatten and concatenate
        color_features = np.concatenate([hist_r.flatten(), hist_g.flatten(), hist_b.flatten()])
        return color_features
    
    def extract_texture_features(self, image):
        """Extract texture features using LBP (Local Binary Pattern)"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate LBP
        radius = 3
        n_points = 8 * radius
        
        # Simple LBP implementation
        lbp = np.zeros_like(gray)
        for i in range(radius, gray.shape[0] - radius):
            for j in range(radius, gray.shape[1] - radius):
                center = gray[i, j]
                code = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if x < gray.shape[0] and y < gray.shape[1]:
                        if gray[x, y] >= center:
                            code |= (1 << k)
                lbp[i, j] = code % 256
        
        # Calculate histogram of LBP
        hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
        return hist
    
    def extract_shape_features(self, image):
        """Extract basic shape/edge features"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density in different regions
        h, w = edges.shape
        regions = [
            edges[:h//2, :w//2],  # top-left
            edges[:h//2, w//2:],  # top-right
            edges[h//2:, :w//2],  # bottom-left
            edges[h//2:, w//2:]   # bottom-right
        ]
        
        edge_features = [np.sum(region) / region.size for region in regions]
        
        # Add overall edge density
        edge_features.append(np.sum(edges) / edges.size)
        
        return np.array(edge_features)
    
    def extract_features(self, image):
        """Extract all features from an image"""
        color_features = self.extract_color_features(image)
        texture_features = self.extract_texture_features(image)
        shape_features = self.extract_shape_features(image)
        
        # Combine all features
        features = np.concatenate([color_features, texture_features, shape_features])
        return features
    
    def process_directory(self, image_dir, output_file=None):
        """Process all images in a directory"""
        image_dir = Path(image_dir)
        image_paths = []
        features_list = []
        
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # Find all image files
        for ext in extensions:
            image_paths.extend(list(image_dir.glob(f'*{ext}')))
            image_paths.extend(list(image_dir.glob(f'*{ext.upper()}')))
        
        print(f"Found {len(image_paths)} images to process")
        
        # Process each image
        valid_paths = []
        for i, image_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {image_path.name}")
            
            image = self.load_image(image_path)
            if image is not None:
                features = self.extract_features(image)
                features_list.append(features)
                valid_paths.append(str(image_path))
        
        if not features_list:
            print("No valid images found!")
            return None, None
        
        # Convert to numpy array and normalize
        features_array = np.array(features_list)
        features_normalized = self.scaler.fit_transform(features_array)
        
        print(f"Extracted features shape: {features_normalized.shape}")
        
        # Save results if output file specified
        if output_file:
            data = {
                'features': features_normalized,
                'image_paths': valid_paths,
                'scaler': self.scaler
            }
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"Features saved to {output_file}")
        
        return features_normalized, valid_paths

def main():
    # Default settings
    input_dir = 'sample_photos'
    output_file = 'photo_features.pkl'
    image_size = (224, 224)
    
    # Create processor
    processor = PhotoProcessor(image_size=image_size)
    
    # Process images
    features, image_paths = processor.process_directory(input_dir, output_file)
    
    if features is not None:
        print(f"\nProcessing complete!")
        print(f"Processed {len(image_paths)} images")
        print(f"Feature vector size: {features.shape[1]}")

if __name__ == "__main__":
    main()