#!/usr/bin/env python3
"""
Unit tests for CLIP semantic analysis functionality
"""

import pytest
import numpy as np
import torch
import json
import os
import sys
from pathlib import Path
from PIL import Image
import tempfile
from unittest.mock import Mock, patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Skip CLIP tests if clip is not available
clip_available = True
try:
    import clip
except ImportError:
    clip_available = False

pytestmark = pytest.mark.skipif(not clip_available, reason="CLIP not available")

from clip_analysis.clip_detector import CLIPDetector

class TestCLIPDetectorInit:
    """Test CLIP detector initialization"""
    
    @patch('clip_analysis.clip_detector.clip.load')
    def test_detector_initialization(self, mock_clip_load):
        """Test that CLIPDetector initializes correctly"""
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip_load.return_value = (mock_model, mock_preprocess)
        
        detector = CLIPDetector(model_name='ViT-B/32')
        
        assert detector.model_name == 'ViT-B/32'
        assert detector.model == mock_model
        assert detector.preprocess == mock_preprocess
        mock_clip_load.assert_called_once_with('ViT-B/32', device=detector.device)
    
    @patch('clip_analysis.clip_detector.clip.load')
    def test_device_detection(self, mock_clip_load):
        """Test device detection (CPU/CUDA)"""
        mock_clip_load.return_value = (Mock(), Mock())
        
        detector = CLIPDetector()
        
        assert detector.device in ['cpu', 'cuda']
    
    @patch('clip_analysis.clip_detector.clip.load')
    def test_historical_categories_structure(self, mock_clip_load):
        """Test historical categories list structure"""
        mock_clip_load.return_value = (Mock(), Mock())
        
        detector = CLIPDetector()
        
        assert isinstance(detector.historical_categories, list)
        assert len(detector.historical_categories) > 0
        
        # Check some expected categories
        expected_categories = [
            "a person in formal clothing",
            "family portrait", 
            "steam locomotive",
            "European building"
        ]
        
        for category in expected_categories:
            assert category in detector.historical_categories
    
    @patch('clip_analysis.clip_detector.clip.load')
    def test_priority_categories_list(self, mock_clip_load):
        """Test priority categories list"""
        mock_clip_load.return_value = (Mock(), Mock())
        
        detector = CLIPDetector()
        
        assert isinstance(detector.priority_categories, list)
        assert len(detector.priority_categories) > 0
        assert "a person in formal clothing" in detector.priority_categories
    
    @patch('clip_analysis.clip_detector.clip.load')
    def test_historical_context_mapping(self, mock_clip_load):
        """Test historical context mapping"""
        mock_clip_load.return_value = (Mock(), Mock())
        
        detector = CLIPDetector()
        
        assert isinstance(detector.historical_context, dict)
        assert len(detector.historical_context) > 0
        
        # Check mapping exists for priority categories
        for category in detector.priority_categories[:5]:  # Check first 5
            assert category in detector.historical_context
            assert isinstance(detector.historical_context[category], str)
    
    @patch('clip_analysis.clip_detector.clip.load')
    @patch('clip_analysis.clip_detector.clip.tokenize')
    def test_text_features_encoding(self, mock_tokenize, mock_clip_load):
        """Test text features encoding during initialization"""
        mock_model = Mock()
        mock_preprocess = Mock()
        mock_clip_load.return_value = (mock_model, mock_preprocess)
        
        # Mock tokenize and encode_text
        mock_tokenize.return_value = torch.zeros((50, 77))  # Mock tokenized text
        mock_text_features = torch.randn(50, 512)  # Mock text features
        mock_model.encode_text.return_value = mock_text_features
        
        detector = CLIPDetector()
        
        assert detector.text_features is not None
        mock_tokenize.assert_called_once()
        mock_model.encode_text.assert_called_once()


class TestCLIPDetectorMethods:
    """Test CLIP detector core methods"""
    
    @pytest.fixture
    def mock_detector(self):
        """Create a mocked CLIPDetector for testing"""
        with patch('clip_analysis.clip_detector.clip.load') as mock_load:
            mock_model = Mock()
            mock_preprocess = Mock()
            mock_load.return_value = (mock_model, mock_preprocess)
            
            detector = CLIPDetector()
            detector.text_features = torch.randn(50, 512)  # Mock precomputed features
            
            return detector
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        img = Image.new('RGB', (224, 224), color='red')
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img.save(tmp.name)
            yield tmp.name
        os.unlink(tmp.name)
    
    def test_analyze_image_semantics_success(self, mock_detector, sample_image):
        """Test semantic analysis with successful processing"""
        # Mock image preprocessing and model inference
        mock_image_tensor = torch.randn(1, 3, 224, 224)
        mock_detector.preprocess.return_value = mock_image_tensor.squeeze(0)
        
        # Mock image features and similarities
        mock_image_features = torch.randn(1, 512)
        mock_detector.model.encode_image.return_value = mock_image_features
        
        # Create mock similarities (some above threshold)
        mock_similarities = np.array([0.8, 0.6, 0.4, 0.2, 0.1] + [0.05] * 45)  # 50 total
        
        with patch('torch.no_grad'), \
             patch.object(torch.Tensor, 'softmax') as mock_softmax:
            
            mock_softmax.return_value.cpu.return_value.numpy.return_value = [mock_similarities]
            
            detections = mock_detector.analyze_image_semantics(sample_image, confidence_threshold=0.3)
            
            assert isinstance(detections, list)
            # Should have 3 detections above 0.3 threshold
            assert len(detections) == 3
            
            # Check detection structure
            for detection in detections:
                assert 'category' in detection
                assert 'confidence' in detection
                assert 'category_id' in detection
                assert 'historical_context' in detection
                assert detection['confidence'] >= 0.3
    
    def test_analyze_image_semantics_no_detections(self, mock_detector, sample_image):
        """Test semantic analysis with no detections above threshold"""
        mock_image_tensor = torch.randn(1, 3, 224, 224)
        mock_detector.preprocess.return_value = mock_image_tensor.squeeze(0)
        mock_image_features = torch.randn(1, 512)
        mock_detector.model.encode_image.return_value = mock_image_features
        
        # All similarities below threshold
        mock_similarities = np.array([0.1] * 50)
        
        with patch('torch.no_grad'), \
             patch.object(torch.Tensor, 'softmax') as mock_softmax:
            
            mock_softmax.return_value.cpu.return_value.numpy.return_value = [mock_similarities]
            
            detections = mock_detector.analyze_image_semantics(sample_image, confidence_threshold=0.5)
            
            assert isinstance(detections, list)
            assert len(detections) == 0
    
    def test_analyze_image_semantics_error_handling(self, mock_detector):
        """Test error handling in semantic analysis"""
        # Test with non-existent file
        detections = mock_detector.analyze_image_semantics('/path/that/does/not/exist.jpg')
        assert detections == []
    
    def test_analyze_photo_success(self, mock_detector, sample_image):
        """Test complete photo analysis"""
        # Mock analyze_image_semantics
        mock_detections = [
            {'category': 'family portrait', 'confidence': 0.8, 'category_id': 0},
            {'category': 'a person in formal clothing', 'confidence': 0.6, 'category_id': 1}
        ]
        mock_detector.analyze_image_semantics = Mock(return_value=mock_detections)
        
        result = mock_detector.analyze_photo(sample_image)
        
        assert result is not None
        assert isinstance(result, dict)
        
        # Check required fields
        required_fields = ['filename', 'total_detections', 'category_counts', 
                          'priority_score', 'detections', 'top_categories']
        for field in required_fields:
            assert field in result
        
        assert result['total_detections'] == 2
        assert result['category_counts']['family portrait'] == 1
        assert result['priority_score'] > 0
        assert len(result['top_categories']) <= 5
    
    def test_priority_score_calculation(self, mock_detector):
        """Test priority score calculation"""
        # Mock detections with priority categories
        mock_detections = [
            {'category': 'a person in formal clothing', 'confidence': 0.9},  # High priority
            {'category': 'family portrait', 'confidence': 0.8},              # High priority
            {'category': 'some other category', 'confidence': 0.7}            # Not in priority
        ]
        
        mock_detector.analyze_image_semantics = Mock(return_value=mock_detections)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
            # Create dummy image
            img = Image.new('RGB', (100, 100), color='red')
            img.save(tmp.name)
            
            result = mock_detector.analyze_photo(tmp.name)
            
            # Should have higher score due to priority categories
            assert result['priority_score'] > 0
    
    def test_create_semantic_visualization(self, mock_detector, sample_image):
        """Test semantic visualization creation"""
        detections = [
            {'category': 'family portrait', 'confidence': 0.8, 'historical_context': 'Test context'},
            {'category': 'European building', 'confidence': 0.6, 'historical_context': 'Test context'}
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_viz.png'
            
            # Mock matplotlib to avoid display issues
            with patch('matplotlib.pyplot.savefig'), \
                 patch('matplotlib.pyplot.close'), \
                 patch('matplotlib.pyplot.subplots') as mock_subplots:
                
                mock_fig = Mock()
                mock_ax1 = Mock()
                mock_ax2 = Mock()
                mock_subplots.return_value = (mock_fig, [mock_ax1, mock_ax2])
                
                success = mock_detector.create_semantic_visualization(
                    sample_image, detections, str(output_path), show_image=False
                )
                
                assert success


class TestCLIPDetectorWorkflows:
    """Test complete CLIP detector workflows"""
    
    @pytest.fixture
    def mock_detector_full(self):
        """Create fully mocked detector for workflow tests"""
        with patch('clip_analysis.clip_detector.clip.load'):
            detector = CLIPDetector()
            
            # Mock analyze_photo to return consistent results
            def mock_analyze_photo(image_path, confidence_threshold=0.3):
                return {
                    'filename': Path(image_path).name,
                    'total_detections': 3,
                    'category_counts': {'family portrait': 1, 'European building': 1, 'period furniture': 1},
                    'priority_score': 12.5,
                    'detections': [
                        {
                            'category': 'family portrait',
                            'confidence': 0.8,
                            'category_id': 0,
                            'historical_context': 'Family relationships and social customs'
                        },
                        {
                            'category': 'European building',
                            'confidence': 0.6,
                            'category_id': 1,
                            'historical_context': 'Architecture for dating photos'
                        },
                        {
                            'category': 'period furniture',
                            'confidence': 0.4,
                            'category_id': 2,
                            'historical_context': 'Interior design and social status'
                        }
                    ],
                    'top_categories': ['family portrait', 'European building', 'period furniture']
                }
            
            detector.analyze_photo = Mock(side_effect=mock_analyze_photo)
            detector.create_semantic_visualization = Mock(return_value=True)
            
            return detector
    
    def test_analyze_directory_workflow(self, mock_detector_full):
        """Test complete directory analysis workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            for i in range(3):
                img = Image.new('RGB', (100, 100), color='red')
                img_path = Path(tmpdir) / f"test_image_{i}.jpg"
                img.save(str(img_path))
            
            # Mock Path.glob
            with patch('pathlib.Path.glob') as mock_glob:
                mock_glob.return_value = list(Path(tmpdir).glob('*.jpg'))
                
                output_file = Path(tmpdir) / 'clip_results.json'
                results = mock_detector_full.analyze_directory(
                    input_dir=tmpdir,
                    output_file=str(output_file),
                    create_visualizations=False
                )
                
                assert len(results) == 3
                assert output_file.exists()
                
                # Check JSON structure
                with open(output_file, 'r') as f:
                    data = json.load(f)
                
                assert 'model_info' in data
                assert 'summary' in data
                assert 'photos' in data
                assert data['model_info']['model_name'] == 'ViT-B/32'
    
    def test_analyze_random_photos(self, mock_detector_full):
        """Test random photos analysis"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            for i in range(5):
                img = Image.new('RGB', (100, 100), color='blue')
                img_path = Path(tmpdir) / f"test_image_{i}.jpg"
                img.save(str(img_path))
            
            with patch('pathlib.Path.glob') as mock_glob, \
                 patch('random.sample') as mock_sample:
                
                test_files = list(Path(tmpdir).glob('*.jpg'))
                mock_glob.return_value = test_files
                mock_sample.return_value = test_files[:3]
                
                results = mock_detector_full.analyze_random_photos(
                    input_dir=tmpdir,
                    num_photos=3,
                    show_images=False
                )
                
                assert len(results) == 3
    
    def test_analyze_single_photo_workflow(self, mock_detector_full):
        """Test single photo analysis workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Image.new('RGB', (100, 100), color='green')
            img_path = Path(tmpdir) / "test_image.jpg"
            img.save(str(img_path))
            
            result = mock_detector_full.analyze_single_photo(
                str(img_path),
                show_image=False,
                save_result=True
            )
            
            assert result is not None
            assert result['filename'] == 'test_image.jpg'
    
    def test_compare_with_text_queries(self, mock_detector_full):
        """Test custom text query comparison"""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Image.new('RGB', (100, 100), color='yellow')
            img_path = Path(tmpdir) / "test_image.jpg"
            img.save(str(img_path))
            
            # Mock the text comparison functionality
            mock_similarities = [0.8, 0.6, 0.4, 0.2]
            mock_queries = [
                "historical photograph",
                "family portrait", 
                "vintage clothing",
                "modern scene"
            ]
            
            with patch.object(mock_detector_full, 'preprocess') as mock_preprocess, \
                 patch.object(mock_detector_full.model, 'encode_image') as mock_encode_img, \
                 patch.object(mock_detector_full.model, 'encode_text') as mock_encode_txt, \
                 patch('clip_analysis.clip_detector.clip.tokenize') as mock_tokenize, \
                 patch('torch.no_grad'):
                
                # Setup mocks
                mock_preprocess.return_value = torch.randn(3, 224, 224)
                mock_encode_img.return_value = torch.randn(1, 512)
                mock_encode_txt.return_value = torch.randn(4, 512)
                mock_tokenize.return_value = torch.zeros(4, 77)
                
                # Mock similarity calculation
                with patch.object(torch.Tensor, 'softmax') as mock_softmax:
                    mock_softmax.return_value.cpu.return_value.numpy.return_value = [mock_similarities]
                    
                    results = mock_detector_full.compare_with_text_queries(str(img_path), mock_queries)
                    
                    assert isinstance(results, list)
                    assert len(results) == 4
                    assert results[0]['similarity'] >= results[1]['similarity']  # Should be sorted


class TestCLIPDetectorHelpers:
    """Test helper methods and utilities"""
    
    @pytest.fixture 
    def detector_for_helpers(self):
        """Create detector for testing helpers"""
        with patch('clip_analysis.clip_detector.clip.load'):
            return CLIPDetector()
    
    def test_calculate_summary_empty(self, detector_for_helpers):
        """Test summary calculation with empty results"""
        summary = detector_for_helpers._calculate_summary([])
        assert summary == {}
    
    def test_calculate_summary_with_data(self, detector_for_helpers):
        """Test summary calculation with sample data"""
        results = [
            {
                'total_detections': 3,
                'category_counts': {'family portrait': 1, 'European building': 1, 'person': 1}
            },
            {
                'total_detections': 2,
                'category_counts': {'family portrait': 1, 'period furniture': 1}
            },
            {
                'total_detections': 0,
                'category_counts': {}
            }
        ]
        
        summary = detector_for_helpers._calculate_summary(results)
        
        assert summary['photos_with_detections'] == 2
        assert summary['photos_without_detections'] == 1
        assert summary['total_detections'] == 5
        assert summary['average_detections_per_photo'] == round(5/3, 2)
        assert summary['most_common_categories']['family portrait'] == 2
    
    def test_text_features_encoding_helper(self, detector_for_helpers):
        """Test _encode_text_categories helper method"""
        with patch('clip_analysis.clip_detector.clip.tokenize') as mock_tokenize, \
             patch.object(detector_for_helpers.model, 'encode_text') as mock_encode, \
             patch('torch.no_grad'):
            
            mock_tokenize.return_value = torch.zeros(50, 77)
            mock_features = torch.randn(50, 512)
            mock_encode.return_value = mock_features
            
            result = detector_for_helpers._encode_text_categories()
            
            assert result is not None
            mock_tokenize.assert_called_once()
            mock_encode.assert_called_once()


class TestCLIPDetectorErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.fixture
    def detector_with_errors(self):
        """Create detector for error testing"""
        with patch('clip_analysis.clip_detector.clip.load'):
            return CLIPDetector()
    
    def test_analyze_image_semantics_exception(self, detector_with_errors):
        """Test exception handling in analyze_image_semantics"""
        # Mock preprocess to raise exception
        detector_with_errors.preprocess = Mock(side_effect=Exception("Preprocessing error"))
        
        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
            img = Image.new('RGB', (100, 100))
            img.save(tmp.name)
            
            detections = detector_with_errors.analyze_image_semantics(tmp.name)
            assert detections == []
    
    def test_visualization_error_handling(self, detector_with_errors):
        """Test visualization error handling"""
        detections = [
            {'category': 'family portrait', 'confidence': 0.8, 'historical_context': 'Test'}
        ]
        
        # Test with non-existent image
        success = detector_with_errors.create_semantic_visualization(
            '/path/that/does/not/exist.jpg',
            detections,
            show_image=False
        )
        
        assert success is False
    
    def test_confidence_threshold_filtering(self, detector_with_errors):
        """Test confidence threshold filtering"""
        # Mock analyze_image_semantics with various confidence scores
        mock_similarities = np.array([0.9, 0.7, 0.5, 0.3, 0.1] + [0.05] * 45)
        
        with patch.object(detector_with_errors, 'preprocess'), \
             patch.object(detector_with_errors.model, 'encode_image'), \
             patch('torch.no_grad'), \
             patch.object(torch.Tensor, 'softmax') as mock_softmax:
            
            mock_softmax.return_value.cpu.return_value.numpy.return_value = [mock_similarities]
            
            with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
                img = Image.new('RGB', (100, 100))
                img.save(tmp.name)
                
                # Test different thresholds
                detections_high = detector_with_errors.analyze_image_semantics(tmp.name, 0.8)
                detections_med = detector_with_errors.analyze_image_semantics(tmp.name, 0.4)
                detections_low = detector_with_errors.analyze_image_semantics(tmp.name, 0.2)
                
                assert len(detections_high) < len(detections_med) < len(detections_low)


if __name__ == '__main__':
    # Run tests  
    pytest.main([__file__, '-v'])