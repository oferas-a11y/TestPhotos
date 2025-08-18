#!/usr/bin/env python3
"""
Unit tests for OpenCV photo analysis functionality
"""

import pytest
import numpy as np
import cv2
import json
import os
import sys
from pathlib import Path
from PIL import Image
import tempfile

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from opencv_analysis.photo_analyzer import PhotoAnalyzer

class TestPhotoAnalyzer:
    """Test cases for PhotoAnalyzer class"""
    
    @pytest.fixture
    def analyzer(self):
        """Create PhotoAnalyzer instance for testing"""
        return PhotoAnalyzer()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        # Create a simple test image (100x100 RGB)
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            yield tmp.name
        os.unlink(tmp.name)
    
    @pytest.fixture
    def bright_image(self):
        """Create a bright test image"""
        img = np.full((100, 100, 3), 200, dtype=np.uint8)  # Bright image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            yield tmp.name
        os.unlink(tmp.name)
    
    @pytest.fixture
    def dark_image(self):
        """Create a dark test image"""
        img = np.full((100, 100, 3), 50, dtype=np.uint8)  # Dark image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            yield tmp.name
        os.unlink(tmp.name)
    
    def test_analyzer_initialization(self, analyzer):
        """Test that PhotoAnalyzer initializes correctly"""
        assert analyzer is not None
        assert hasattr(analyzer, 'analyze_image')
        assert hasattr(analyzer, 'calculate_brightness')
        assert hasattr(analyzer, 'calculate_contrast')
    
    def test_calculate_brightness(self, analyzer, bright_image, dark_image):
        """Test brightness calculation"""
        bright_img = cv2.imread(bright_image)
        dark_img = cv2.imread(dark_image)
        
        bright_value = analyzer.calculate_brightness(bright_img)
        dark_value = analyzer.calculate_brightness(dark_img)
        
        assert isinstance(bright_value, (int, float))
        assert isinstance(dark_value, (int, float))
        assert bright_value > dark_value
        assert 0 <= bright_value <= 255
        assert 0 <= dark_value <= 255
    
    def test_calculate_contrast(self, analyzer, sample_image):
        """Test contrast calculation"""
        img = cv2.imread(sample_image)
        contrast = analyzer.calculate_contrast(img)
        
        assert isinstance(contrast, (int, float))
        assert contrast >= 0
    
    def test_get_dominant_colors(self, analyzer, sample_image):
        """Test dominant colors extraction"""
        img = cv2.imread(sample_image)
        colors = analyzer.get_dominant_colors(img, k=3)
        
        assert isinstance(colors, list)
        assert len(colors) == 3
        for color in colors:
            assert len(color) == 3  # RGB values
            assert all(0 <= c <= 255 for c in color)
    
    def test_calculate_edge_density(self, analyzer, sample_image):
        """Test edge density calculation"""
        img = cv2.imread(sample_image)
        edge_density = analyzer.calculate_edge_density(img)
        
        assert isinstance(edge_density, (int, float))
        assert 0 <= edge_density <= 1
    
    def test_calculate_texture_score(self, analyzer, sample_image):
        """Test texture score calculation"""
        img = cv2.imread(sample_image)
        texture_score = analyzer.calculate_texture_score(img)
        
        assert isinstance(texture_score, (int, float))
        assert texture_score >= 0
    
    def test_analyze_image_success(self, analyzer, sample_image):
        """Test complete image analysis"""
        result = analyzer.analyze_image(sample_image)
        
        assert result is not None
        assert isinstance(result, dict)
        
        # Check required fields
        required_fields = ['filename', 'brightness', 'contrast', 'dominant_colors', 
                          'edge_density', 'texture_score', 'image_size']
        for field in required_fields:
            assert field in result
        
        # Check data types
        assert isinstance(result['filename'], str)
        assert isinstance(result['brightness'], (int, float))
        assert isinstance(result['contrast'], (int, float))
        assert isinstance(result['dominant_colors'], list)
        assert isinstance(result['edge_density'], (int, float))
        assert isinstance(result['texture_score'], (int, float))
        assert isinstance(result['image_size'], dict)
    
    def test_analyze_image_nonexistent_file(self, analyzer):
        """Test analysis with non-existent file"""
        result = analyzer.analyze_image('/path/that/does/not/exist.jpg')
        assert result is None
    
    def test_analyze_directory_empty(self, analyzer):
        """Test directory analysis with empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = analyzer.analyze_directory(tmpdir)
            assert isinstance(results, list)
            assert len(results) == 0
    
    def test_analyze_directory_with_images(self, analyzer, sample_image):
        """Test directory analysis with sample images"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy sample image to temp directory
            temp_image = Path(tmpdir) / "test_image.jpg"
            img = cv2.imread(sample_image)
            cv2.imwrite(str(temp_image), img)
            
            results = analyzer.analyze_directory(tmpdir)
            
            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0]['filename'] == 'test_image.jpg'
    
    def test_json_serialization(self, analyzer, sample_image):
        """Test that analysis results can be serialized to JSON"""
        result = analyzer.analyze_image(sample_image)
        
        # Should not raise an exception
        json_str = json.dumps(result, indent=2)
        assert isinstance(json_str, str)
        
        # Should be able to load back
        loaded_result = json.loads(json_str)
        assert loaded_result == result
    
    def test_image_size_calculation(self, analyzer):
        """Test image size calculation"""
        # Create image with known dimensions
        img = np.zeros((150, 200, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            result = analyzer.analyze_image(tmp.name)
            
            assert result['image_size']['height'] == 150
            assert result['image_size']['width'] == 200
            assert result['image_size']['channels'] == 3
        
        os.unlink(tmp.name)
    
    def test_color_histogram(self, analyzer, sample_image):
        """Test color histogram calculation"""
        img = cv2.imread(sample_image)
        hist = analyzer.calculate_color_histogram(img)
        
        assert isinstance(hist, dict)
        assert 'red' in hist
        assert 'green' in hist  
        assert 'blue' in hist
        
        for channel_hist in hist.values():
            assert len(channel_hist) == 256  # 256 bins for each channel
            assert all(count >= 0 for count in channel_hist)


class TestPhotoAnalyzerIntegration:
    """Integration tests for PhotoAnalyzer with real workflow scenarios"""
    
    def test_batch_processing_workflow(self):
        """Test complete batch processing workflow"""
        analyzer = PhotoAnalyzer()
        
        # Create multiple test images
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                img_path = Path(tmpdir) / f"test_image_{i}.jpg"
                cv2.imwrite(str(img_path), img)
            
            # Analyze directory
            results = analyzer.analyze_directory(tmpdir, f"{tmpdir}/results.json")
            
            assert len(results) == 3
            assert Path(f"{tmpdir}/results.json").exists()
            
            # Verify JSON file content
            with open(f"{tmpdir}/results.json", 'r') as f:
                saved_data = json.load(f)
            
            assert len(saved_data['photos']) == 3
            assert 'summary' in saved_data
    
    def test_summary_statistics(self):
        """Test summary statistics calculation"""
        analyzer = PhotoAnalyzer()
        
        # Create test results
        test_results = [
            {'brightness': 100, 'contrast': 50, 'edge_density': 0.1, 'texture_score': 20},
            {'brightness': 150, 'contrast': 75, 'edge_density': 0.2, 'texture_score': 30},
            {'brightness': 200, 'contrast': 100, 'edge_density': 0.3, 'texture_score': 40}
        ]
        
        summary = analyzer._calculate_summary(test_results)
        
        assert 'average_brightness' in summary
        assert 'average_contrast' in summary
        assert 'average_edge_density' in summary
        assert 'average_texture_score' in summary
        
        assert summary['average_brightness'] == 150.0  # (100+150+200)/3
        assert summary['total_photos_analyzed'] == 3


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])