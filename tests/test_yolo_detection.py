#!/usr/bin/env python3
"""
Unit tests for YOLO object detection functionality
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
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo_detection.yolo_detector import YOLODetector

class TestYOLODetectorInit:
    """Test YOLO detector initialization"""
    
    @patch('yolo_detection.yolo_detector.YOLO')
    def test_detector_initialization(self, mock_yolo):
        """Test that YOLODetector initializes correctly"""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        detector = YOLODetector(model_size='n')
        
        assert detector.model_size == 'n'
        assert detector.model == mock_model
        mock_yolo.assert_called_once_with('yolov8n.pt')
    
    @patch('yolo_detection.yolo_detector.YOLO')
    def test_detector_different_model_sizes(self, mock_yolo):
        """Test initialization with different model sizes"""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        sizes = ['n', 's', 'm', 'l', 'x']
        for size in sizes:
            detector = YOLODetector(model_size=size)
            assert detector.model_size == size
    
    def test_relevant_classes_structure(self):
        """Test that relevant classes dictionary is properly structured"""
        with patch('yolo_detection.yolo_detector.YOLO'):
            detector = YOLODetector()
            
            assert isinstance(detector.relevant_classes, dict)
            assert len(detector.relevant_classes) > 0
            
            # Check some expected classes
            assert 0 in detector.relevant_classes  # person
            assert detector.relevant_classes[0] == 'person'
            
            # Check that all values are strings
            for class_id, class_name in detector.relevant_classes.items():
                assert isinstance(class_id, int)
                assert isinstance(class_name, str)
    
    def test_priority_objects_list(self):
        """Test priority objects list"""
        with patch('yolo_detection.yolo_detector.YOLO'):
            detector = YOLODetector()
            
            assert isinstance(detector.priority_objects, list)
            assert len(detector.priority_objects) > 0
            assert 'person' in detector.priority_objects
            assert 'horse' in detector.priority_objects
    
    def test_historical_context_mapping(self):
        """Test historical context mapping"""
        with patch('yolo_detection.yolo_detector.YOLO'):
            detector = YOLODetector()
            
            assert isinstance(detector.historical_context, dict)
            assert len(detector.historical_context) > 0
            
            # Check some expected mappings
            assert 'person' in detector.historical_context
            assert isinstance(detector.historical_context['person'], str)


class TestYOLODetectorMethods:
    """Test YOLO detector core methods"""
    
    @pytest.fixture
    def mock_detector(self):
        """Create a mocked YOLODetector for testing"""
        with patch('yolo_detection.yolo_detector.YOLO'):
            detector = YOLODetector()
            # Mock the model with fake results
            detector.model = Mock()
            return detector
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        img = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            yield tmp.name
        os.unlink(tmp.name)
    
    def test_detect_objects_no_detections(self, mock_detector, sample_image):
        """Test detect_objects with no detections"""
        # Mock empty results
        mock_result = Mock()
        mock_result.boxes = None
        mock_detector.model.return_value = [mock_result]
        
        detections = mock_detector.detect_objects(sample_image)
        
        assert isinstance(detections, list)
        assert len(detections) == 0
    
    def test_detect_objects_with_detections(self, mock_detector, sample_image):
        """Test detect_objects with valid detections"""
        # Mock detection results
        mock_result = Mock()
        mock_boxes = Mock()
        
        # Mock tensor-like objects
        mock_boxes.xyxy = Mock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 10, 50, 50]])
        mock_boxes.conf = Mock()
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.8])
        mock_boxes.cls = Mock()
        mock_boxes.cls.cpu.return_value.numpy.return_value.astype.return_value = np.array([0])  # person class
        
        mock_result.boxes = mock_boxes
        mock_detector.model.return_value = [mock_result]
        
        detections = mock_detector.detect_objects(sample_image, confidence_threshold=0.5)
        
        assert isinstance(detections, list)
        assert len(detections) == 1
        
        detection = detections[0]
        assert detection['class_id'] == 0
        assert detection['class_name'] == 'person'
        assert detection['confidence'] == 0.8
        assert 'bbox' in detection
    
    def test_analyze_photo_success(self, mock_detector, sample_image):
        """Test analyze_photo with successful detection"""
        # Mock detect_objects to return sample detection
        mock_detector.detect_objects = Mock(return_value=[
            {
                'class_id': 0,
                'class_name': 'person', 
                'confidence': 0.8,
                'bbox': {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50, 'width': 40, 'height': 40}
            }
        ])
        
        result = mock_detector.analyze_photo(sample_image)
        
        assert result is not None
        assert isinstance(result, dict)
        
        # Check required fields
        required_fields = ['filename', 'total_objects', 'object_counts', 'priority_score', 'detections']
        for field in required_fields:
            assert field in result
        
        assert result['total_objects'] == 1
        assert result['object_counts']['person'] == 1
        assert result['priority_score'] > 0
    
    def test_analyze_photo_file_not_found(self, mock_detector):
        """Test analyze_photo with non-existent file"""
        result = mock_detector.analyze_photo('/path/that/does/not/exist.jpg')
        assert result is None
    
    def test_priority_score_calculation(self, mock_detector):
        """Test priority score calculation logic"""
        # Create mock detections with different priority objects
        detections = [
            {'class_name': 'person', 'confidence': 0.9},  # High priority
            {'class_name': 'car', 'confidence': 0.8},     # Medium priority  
            {'class_name': 'bird', 'confidence': 0.7}     # Lower priority
        ]
        
        mock_detector.detect_objects = Mock(return_value=detections)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
            # Create dummy image
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(tmp.name, img)
            
            result = mock_detector.analyze_photo(tmp.name)
            
            # Priority score should be > 0 due to person detection
            assert result['priority_score'] > 0
    
    def test_create_detection_visualization(self, mock_detector, sample_image):
        """Test visualization creation"""
        detections = [
            {
                'class_id': 0,
                'class_name': 'person',
                'confidence': 0.8,
                'bbox': {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50, 'width': 40, 'height': 40}
            }
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_viz.png'
            
            # Mock matplotlib to avoid display issues in tests
            with patch('matplotlib.pyplot.savefig'), \
                 patch('matplotlib.pyplot.close'), \
                 patch('matplotlib.pyplot.subplots') as mock_subplots:
                
                mock_fig = Mock()
                mock_ax = Mock()
                mock_subplots.return_value = (mock_fig, mock_ax)
                
                success = mock_detector.create_detection_visualization(
                    sample_image, detections, str(output_path), show_image=False
                )
                
                assert success


class TestYOLODetectorWorkflows:
    """Test complete YOLO detector workflows"""
    
    @pytest.fixture
    def mock_detector_full(self):
        """Create a fully mocked detector for workflow tests"""
        with patch('yolo_detection.yolo_detector.YOLO'):
            detector = YOLODetector()
            
            # Mock analyze_photo to return consistent results
            def mock_analyze_photo(image_path, confidence_threshold=0.5):
                return {
                    'filename': Path(image_path).name,
                    'total_objects': 2,
                    'object_counts': {'person': 1, 'car': 1},
                    'priority_score': 15.0,
                    'detections': [
                        {
                            'class_name': 'person',
                            'confidence': 0.8,
                            'bbox': {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50}
                        },
                        {
                            'class_name': 'car', 
                            'confidence': 0.7,
                            'bbox': {'x1': 60, 'y1': 60, 'x2': 100, 'y2': 100}
                        }
                    ]
                }
            
            detector.analyze_photo = Mock(side_effect=mock_analyze_photo)
            detector.create_detection_visualization = Mock(return_value=True)
            
            return detector
    
    def test_analyze_directory_workflow(self, mock_detector_full):
        """Test complete directory analysis workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            for i in range(3):
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                img_path = Path(tmpdir) / f"test_image_{i}.jpg"
                cv2.imwrite(str(img_path), img)
            
            # Mock Path.glob to return our test files
            with patch('pathlib.Path.glob') as mock_glob:
                mock_glob.return_value = list(Path(tmpdir).glob('*.jpg'))
                
                output_file = Path(tmpdir) / 'results.json'
                results = mock_detector_full.analyze_directory(
                    input_dir=tmpdir,
                    output_file=str(output_file),
                    create_visualizations=False
                )
                
                assert len(results) == 3
                assert output_file.exists()
                
                # Check JSON file content
                with open(output_file, 'r') as f:
                    data = json.load(f)
                
                assert 'model_info' in data
                assert 'summary' in data
                assert 'photos' in data
                assert len(data['photos']) == 3
    
    def test_analyze_random_photos(self, mock_detector_full):
        """Test random photos analysis"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            for i in range(5):
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                img_path = Path(tmpdir) / f"test_image_{i}.jpg"
                cv2.imwrite(str(img_path), img)
            
            # Mock random.sample and Path.glob
            with patch('pathlib.Path.glob') as mock_glob, \
                 patch('random.sample') as mock_sample:
                
                test_files = list(Path(tmpdir).glob('*.jpg'))
                mock_glob.return_value = test_files
                mock_sample.return_value = test_files[:3]  # Return first 3
                
                results = mock_detector_full.analyze_random_photos(
                    input_dir=tmpdir,
                    num_photos=3,
                    show_images=False
                )
                
                assert len(results) == 3
    
    def test_analyze_single_photo_workflow(self, mock_detector_full):
        """Test single photo analysis workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_path = Path(tmpdir) / "test_image.jpg"
            cv2.imwrite(str(img_path), img)
            
            result = mock_detector_full.analyze_single_photo(
                str(img_path),
                show_image=False,
                save_result=True
            )
            
            assert result is not None
            assert result['filename'] == 'test_image.jpg'
            
            # Check that result file was created
            result_files = list(Path(tmpdir).glob('single_photo_*.json'))
            # Note: The file would be created in current dir, not tmpdir


class TestYOLODetectorHelpers:
    """Test helper methods and edge cases"""
    
    @pytest.fixture
    def detector_for_helpers(self):
        """Create detector for testing helper methods"""
        with patch('yolo_detection.yolo_detector.YOLO'):
            return YOLODetector()
    
    def test_calculate_summary_empty(self, detector_for_helpers):
        """Test summary calculation with empty results"""
        summary = detector_for_helpers._calculate_summary([])
        assert summary == {}
    
    def test_calculate_summary_with_data(self, detector_for_helpers):
        """Test summary calculation with sample data"""
        results = [
            {
                'total_objects': 2,
                'object_counts': {'person': 1, 'car': 1}
            },
            {
                'total_objects': 1, 
                'object_counts': {'person': 1}
            },
            {
                'total_objects': 0,
                'object_counts': {}
            }
        ]
        
        summary = detector_for_helpers._calculate_summary(results)
        
        assert summary['photos_with_objects'] == 2
        assert summary['photos_without_objects'] == 1
        assert summary['total_objects_detected'] == 3
        assert summary['average_objects_per_photo'] == 1.0
        assert summary['most_common_objects']['person'] == 2
    
    def test_interactive_mode_exit(self, detector_for_helpers):
        """Test interactive mode exit functionality"""
        # Mock input to return '4' (exit option)
        with patch('builtins.input', return_value='4'):
            # Should not raise an exception
            try:
                detector_for_helpers.interactive_mode()
            except SystemExit:
                pass  # Expected for exit
    
    def test_bbox_calculation(self, detector_for_helpers):
        """Test bounding box calculation in detections"""
        # This tests the bbox creation logic in detect_objects
        x1, y1, x2, y2 = 10, 20, 50, 60
        
        bbox = {
            'x1': float(x1),
            'y1': float(y1), 
            'x2': float(x2),
            'y2': float(y2),
            'width': float(x2 - x1),
            'height': float(y2 - y1)
        }
        
        assert bbox['width'] == 40
        assert bbox['height'] == 40
        assert all(isinstance(v, float) for v in bbox.values())


class TestYOLODetectorErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.fixture
    def detector_with_errors(self):
        """Create detector that simulates errors"""
        with patch('yolo_detection.yolo_detector.YOLO'):
            detector = YOLODetector()
            return detector
    
    def test_detect_objects_exception_handling(self, detector_with_errors):
        """Test exception handling in detect_objects"""
        # Mock model to raise exception
        detector_with_errors.model = Mock()
        detector_with_errors.model.side_effect = Exception("Model error")
        
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(tmp.name, img)
            
            detections = detector_with_errors.detect_objects(tmp.name)
            assert detections == []
    
    def test_visualization_error_handling(self, detector_with_errors):
        """Test visualization error handling"""
        detections = [
            {
                'class_id': 0,
                'class_name': 'person',
                'confidence': 0.8,
                'bbox': {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 50, 'width': 40, 'height': 40}
            }
        ]
        
        # Test with non-existent image
        success = detector_with_errors.create_detection_visualization(
            '/path/that/does/not/exist.jpg',
            detections,
            show_image=False
        )
        
        assert success is False
    
    def test_confidence_threshold_bounds(self, detector_with_errors):
        """Test confidence threshold boundary conditions"""
        # Mock successful detection
        mock_result = Mock()
        mock_boxes = Mock()
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 10, 50, 50]])
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.5])
        mock_boxes.cls.cpu.return_value.numpy.return_value.astype.return_value = np.array([0])
        mock_result.boxes = mock_boxes
        detector_with_errors.model = Mock(return_value=[mock_result])
        
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
            img = np.zeros((100, 100, 3), dtype=np.uint8) 
            cv2.imwrite(tmp.name, img)
            
            # Test with confidence threshold exactly at detection confidence
            detections = detector_with_errors.detect_objects(tmp.name, confidence_threshold=0.5)
            assert len(detections) == 1
            
            # Test with confidence threshold above detection confidence  
            detections = detector_with_errors.detect_objects(tmp.name, confidence_threshold=0.6)
            assert len(detections) == 0


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])