#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for historical photo analysis tests
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path
from PIL import Image
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(scope="session")
def sample_photos_dir():
    """Create a temporary directory with sample test photos for all tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Create various test images
        test_images = [
            # Bright formal photo (simulates portrait)
            ("portrait.jpg", np.full((200, 150, 3), [200, 180, 160], dtype=np.uint8)),
            
            # Dark document-like photo
            ("document.jpg", np.full((300, 200, 3), [50, 50, 50], dtype=np.uint8)),
            
            # Colorful group photo simulation
            ("group.jpg", np.random.randint(100, 255, (250, 300, 3), dtype=np.uint8)),
            
            # Landscape/building simulation
            ("building.jpg", np.full((400, 600, 3), [120, 100, 90], dtype=np.uint8)),
            
            # High contrast edge-heavy image
            ("detailed.jpg", create_checkerboard_pattern(200, 200)),
        ]
        
        # Save test images
        for filename, img_array in test_images:
            img_path = tmpdir_path / filename
            cv2.imwrite(str(img_path), img_array)
        
        yield str(tmpdir_path)

def create_checkerboard_pattern(height, width, square_size=20):
    """Create a checkerboard pattern for testing edge detection"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(0, height, square_size):
        for j in range(0, width, square_size):
            if ((i // square_size) + (j // square_size)) % 2 == 0:
                image[i:i+square_size, j:j+square_size] = 255
    
    return image

@pytest.fixture
def single_test_image():
    """Create a single test image for individual tests"""
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        cv2.imwrite(tmp.name, img)
        yield tmp.name
    os.unlink(tmp.name)

@pytest.fixture
def rgb_test_image():
    """Create a PIL RGB test image"""
    img = Image.new('RGB', (224, 224), color='red')
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img.save(tmp.name)
        yield tmp.name
    os.unlink(tmp.name)

@pytest.fixture
def mock_analysis_results():
    """Mock analysis results for testing workflows"""
    return [
        {
            'filename': 'test1.jpg',
            'total_objects': 2,
            'object_counts': {'person': 1, 'chair': 1},
            'brightness': 150,
            'contrast': 75
        },
        {
            'filename': 'test2.jpg', 
            'total_objects': 1,
            'object_counts': {'person': 1},
            'brightness': 120,
            'contrast': 60
        },
        {
            'filename': 'test3.jpg',
            'total_objects': 0,
            'object_counts': {},
            'brightness': 90,
            'contrast': 45
        }
    ]

@pytest.fixture
def mock_clustering_data():
    """Mock clustering data for testing"""
    image_paths = [f'test_image_{i}.jpg' for i in range(10)]
    
    # Create feature vectors with 2 distinct clusters
    cluster1_features = np.random.normal(0, 1, (5, 128))
    cluster2_features = np.random.normal(3, 1, (5, 128)) 
    features = np.vstack([cluster1_features, cluster2_features])
    
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    return {
        'image_paths': image_paths,
        'features': features, 
        'labels': labels
    }

@pytest.fixture
def mock_yolo_detections():
    """Mock YOLO detection results"""
    return [
        {
            'class_id': 0,
            'class_name': 'person',
            'confidence': 0.85,
            'bbox': {'x1': 10, 'y1': 10, 'x2': 50, 'y2': 80, 'width': 40, 'height': 70}
        },
        {
            'class_id': 56,
            'class_name': 'chair', 
            'confidence': 0.72,
            'bbox': {'x1': 60, 'y1': 40, 'x2': 100, 'y2': 90, 'width': 40, 'height': 50}
        }
    ]

@pytest.fixture
def mock_clip_detections():
    """Mock CLIP semantic analysis results"""
    return [
        {
            'category': 'family portrait',
            'confidence': 0.82,
            'category_id': 0,
            'historical_context': 'Family relationships, social customs, formal photography practices'
        },
        {
            'category': 'a person in formal clothing',
            'confidence': 0.67, 
            'category_id': 1,
            'historical_context': 'Formal wear indicates social status, special occasions'
        },
        {
            'category': 'European building',
            'confidence': 0.45,
            'category_id': 2,
            'historical_context': 'Architecture styles help date and locate photographs'
        }
    ]

# Test data constants
SAMPLE_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
EXPECTED_YOLO_CLASSES = ['person', 'horse', 'car', 'train', 'bicycle', 'chair', 'book']
EXPECTED_CLIP_CATEGORIES = [
    'family portrait', 
    'a person in formal clothing',
    'steam locomotive',
    'European building'
]

# Skip conditions
def skip_if_no_cuda():
    """Skip test if CUDA is not available"""
    try:
        import torch
        return not torch.cuda.is_available()
    except ImportError:
        return True

def skip_if_no_clip():
    """Skip test if CLIP is not available"""
    try:
        import clip
        return False
    except ImportError:
        return True

def skip_if_no_yolo():
    """Skip test if YOLO (ultralytics) is not available"""
    try:
        from ultralytics import YOLO
        return False
    except ImportError:
        return True

# Pytest markers for conditional testing
pytest.mark.cuda = pytest.mark.skipif(skip_if_no_cuda(), reason="CUDA not available")
pytest.mark.clip = pytest.mark.skipif(skip_if_no_clip(), reason="CLIP not available") 
pytest.mark.yolo = pytest.mark.skipif(skip_if_no_yolo(), reason="YOLO not available")

# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "cuda: tests requiring CUDA")
    config.addinivalue_line("markers", "clip: tests requiring CLIP")
    config.addinivalue_line("markers", "yolo: tests requiring YOLO")
    config.addinivalue_line("markers", "slow: tests that take a long time")
    config.addinivalue_line("markers", "integration: integration tests")

# Helper functions for tests
def create_test_image_with_objects(width=400, height=300):
    """Create a test image that simulates objects for detection testing"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some rectangular "objects"
    # Person-like shape
    cv2.rectangle(img, (50, 50), (100, 200), (255, 200, 150), -1)  # Skin-like color
    
    # Chair-like shape  
    cv2.rectangle(img, (200, 150), (280, 250), (139, 69, 19), -1)  # Brown color
    
    # Add some texture
    noise = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    return img

def create_test_features(n_samples=10, n_features=128, n_clusters=2):
    """Create test feature vectors with known cluster structure"""
    features = []
    labels = []
    
    samples_per_cluster = n_samples // n_clusters
    
    for cluster_id in range(n_clusters):
        # Create cluster center
        center = np.random.randn(n_features) * 3
        
        for _ in range(samples_per_cluster):
            # Add some noise around center
            sample = center + np.random.randn(n_features) * 0.5
            features.append(sample)
            labels.append(cluster_id)
    
    return np.array(features), np.array(labels)