#!/usr/bin/env python3
"""
Photo Quality and Face Analysis Script
Analyzes photos for quality metrics and face detection
"""

import cv2
import numpy as np
import json
from pathlib import Path

class PhotoAnalyzer:
    def __init__(self):
        # Load face detection cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def calculate_sharpness(self, image):
        """Calculate image sharpness using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    def calculate_brightness(self, image):
        """Calculate average brightness"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def calculate_contrast(self, image):
        """Calculate contrast using standard deviation"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.std(gray)
    
    def calculate_noise_level(self, image):
        """Estimate noise level using high-frequency content"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur and subtract from original
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = cv2.absdiff(gray, blurred)
        return np.mean(noise)
    
    def calculate_exposure_quality(self, image):
        """Check for over/under exposure"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        total_pixels = gray.shape[0] * gray.shape[1]
        
        # Check for overexposure (too many bright pixels)
        overexposed = np.sum(hist[240:]) / total_pixels
        
        # Check for underexposure (too many dark pixels)
        underexposed = np.sum(hist[:16]) / total_pixels
        
        # Good exposure should have most pixels in mid-range
        well_exposed = np.sum(hist[50:200]) / total_pixels
        
        return {
            'overexposed_ratio': overexposed,
            'underexposed_ratio': underexposed,
            'well_exposed_ratio': well_exposed
        }
    
    def grade_quality(self, image):
        """Grade photo quality from A to D based on multiple metrics"""
        sharpness = self.calculate_sharpness(image)
        brightness = self.calculate_brightness(image)
        contrast = self.calculate_contrast(image)
        noise = self.calculate_noise_level(image)
        exposure = self.calculate_exposure_quality(image)
        
        # Scoring system (0-100)
        score = 0
        
        # Sharpness scoring (40% weight)
        if sharpness > 500:
            score += 40
        elif sharpness > 200:
            score += 30
        elif sharpness > 100:
            score += 20
        elif sharpness > 50:
            score += 10
        
        # Contrast scoring (20% weight)
        if 30 < contrast < 80:
            score += 20
        elif 20 < contrast < 100:
            score += 15
        elif 15 < contrast < 120:
            score += 10
        elif contrast > 10:
            score += 5
        
        # Brightness scoring (15% weight)
        if 80 < brightness < 180:
            score += 15
        elif 60 < brightness < 200:
            score += 12
        elif 40 < brightness < 220:
            score += 8
        elif brightness > 20:
            score += 4
        
        # Noise scoring (10% weight) - lower noise is better
        if noise < 5:
            score += 10
        elif noise < 10:
            score += 8
        elif noise < 15:
            score += 5
        elif noise < 25:
            score += 2
        
        # Exposure scoring (15% weight)
        if exposure['well_exposed_ratio'] > 0.7 and exposure['overexposed_ratio'] < 0.05 and exposure['underexposed_ratio'] < 0.1:
            score += 15
        elif exposure['well_exposed_ratio'] > 0.5 and exposure['overexposed_ratio'] < 0.1 and exposure['underexposed_ratio'] < 0.2:
            score += 12
        elif exposure['well_exposed_ratio'] > 0.3:
            score += 8
        else:
            score += 3
        
        # Convert score to letter grade
        if score >= 85:
            grade = 'A'
        elif score >= 70:
            grade = 'B'
        elif score >= 55:
            grade = 'C'
        else:
            grade = 'D'
        
        return {
            'grade': grade,
            'score': score,
            'metrics': {
                'sharpness': round(sharpness, 2),
                'brightness': round(brightness, 2),
                'contrast': round(contrast, 2),
                'noise_level': round(noise, 2),
                'exposure': exposure
            }
        }
    
    def count_faces(self, image):
        """Count faces in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return len(faces), faces.tolist() if len(faces) > 0 else []
    
    def analyze_photo(self, image_path):
        """Analyze a single photo for quality and face count"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Get quality metrics
            quality_data = self.grade_quality(image)
            
            # Count faces
            face_count, face_locations = self.count_faces(image)
            
            return {
                'filename': Path(image_path).name,
                'quality_grade': quality_data['grade'],
                'quality_score': quality_data['score'],
                'quality_metrics': quality_data['metrics'],
                'face_count': face_count,
                'face_locations': face_locations
            }
            
        except Exception as e:
            print(f"Error analyzing {image_path}: {e}")
            return None
    
    def analyze_directory(self, input_dir='../sample_photos', output_file='photo_analysis.json'):
        """Analyze all photos in a directory"""
        input_dir = Path(input_dir)
        results = []
        
        # Supported image extensions
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_paths = []
        
        # Find all image files
        for ext in extensions:
            image_paths.extend(list(input_dir.glob(f'*{ext}')))
            image_paths.extend(list(input_dir.glob(f'*{ext.upper()}')))
        
        print(f"Found {len(image_paths)} images to analyze")
        
        # Analyze each image
        for i, image_path in enumerate(image_paths):
            print(f"Analyzing {i+1}/{len(image_paths)}: {image_path.name}")
            
            result = self.analyze_photo(image_path)
            if result:
                results.append(result)
        
        # Save results to JSON
        with open(output_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_photos': len(results),
                    'grade_distribution': self._calculate_grade_distribution(results),
                    'average_face_count': round(np.mean([r['face_count'] for r in results]), 2) if results else 0
                },
                'photos': results
            }, f, indent=2)
        
        print(f"\nAnalysis complete! Results saved to {output_file}")
        return results
    
    def _calculate_grade_distribution(self, results):
        """Calculate distribution of quality grades"""
        grades = [r['quality_grade'] for r in results]
        distribution = {}
        for grade in ['A', 'B', 'C', 'D']:
            distribution[grade] = grades.count(grade)
        return distribution

def main():
    analyzer = PhotoAnalyzer()
    results = analyzer.analyze_directory('../sample_photos', 'photo_analysis.json')
    
    if results:
        print(f"\nSummary:")
        print(f"Total photos analyzed: {len(results)}")
        
        # Grade distribution
        grades = [r['quality_grade'] for r in results]
        for grade in ['A', 'B', 'C', 'D']:
            count = grades.count(grade)
            print(f"Grade {grade}: {count} photos")
        
        # Face statistics
        face_counts = [r['face_count'] for r in results]
        print(f"Average faces per photo: {np.mean(face_counts):.1f}")
        print(f"Photos with faces: {sum(1 for fc in face_counts if fc > 0)}")

if __name__ == "__main__":
    main()