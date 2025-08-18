#!/usr/bin/env python3
"""
Unit tests for photo clustering functionality
"""

import pytest
import numpy as np
import cv2
import json
import os
import sys
import pickle
from pathlib import Path
import tempfile
from sklearn.cluster import KMeans

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from photo_clustering.process_photos import PhotoProcessor
from photo_clustering.kmeans_clustering import PhotoClusterer

class TestPhotoProcessor:
    """Test cases for PhotoProcessor class"""
    
    @pytest.fixture
    def processor(self):
        """Create PhotoProcessor instance for testing"""
        return PhotoProcessor()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image"""
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            yield tmp.name
        os.unlink(tmp.name)
    
    def test_processor_initialization(self, processor):
        """Test that PhotoProcessor initializes correctly"""
        assert processor is not None
        assert hasattr(processor, 'extract_features')
        assert hasattr(processor, 'process_directory')
    
    def test_extract_features_success(self, processor, sample_image):
        """Test feature extraction from valid image"""
        features = processor.extract_features(sample_image)
        
        assert features is not None
        assert isinstance(features, np.ndarray)
        assert len(features.shape) == 1  # Should be flattened
        assert features.shape[0] > 0  # Should have some features
    
    def test_extract_features_invalid_file(self, processor):
        """Test feature extraction with invalid file"""
        features = processor.extract_features('/path/that/does/not/exist.jpg')
        assert features is None
    
    def test_process_directory_empty(self, processor):
        """Test processing empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            image_paths, features = processor.process_directory(tmpdir)
            
            assert isinstance(image_paths, list)
            assert isinstance(features, list)
            assert len(image_paths) == 0
            assert len(features) == 0
    
    def test_process_directory_with_images(self, processor):
        """Test processing directory with sample images"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple test images
            for i in range(3):
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img_path = Path(tmpdir) / f"test_image_{i}.jpg"
                cv2.imwrite(str(img_path), img)
            
            image_paths, features = processor.process_directory(tmpdir)
            
            assert len(image_paths) == 3
            assert len(features) == 3
            assert all(isinstance(f, np.ndarray) for f in features)
            assert all(f.shape[0] > 0 for f in features)
    
    def test_save_features(self, processor):
        """Test saving features to pickle file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test data
            image_paths = ['img1.jpg', 'img2.jpg']
            features = [np.array([1, 2, 3]), np.array([4, 5, 6])]
            
            save_path = Path(tmpdir) / 'test_features.pkl'
            processor.save_features(image_paths, features, str(save_path))
            
            assert save_path.exists()
            
            # Load and verify
            with open(save_path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            assert 'image_paths' in loaded_data
            assert 'features' in loaded_data
            assert loaded_data['image_paths'] == image_paths
            np.testing.assert_array_equal(loaded_data['features'][0], features[0])


class TestPhotoClusterer:
    """Test cases for PhotoClusterer class"""
    
    @pytest.fixture
    def clusterer(self):
        """Create PhotoClusterer instance for testing"""
        return PhotoClusterer()
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature vectors for testing"""
        # Create 10 sample feature vectors with 2 clear clusters
        cluster1 = np.random.normal(0, 1, (5, 100))
        cluster2 = np.random.normal(5, 1, (5, 100))
        features = np.vstack([cluster1, cluster2])
        image_paths = [f'image_{i}.jpg' for i in range(10)]
        return image_paths, features
    
    def test_clusterer_initialization(self, clusterer):
        """Test that PhotoClusterer initializes correctly"""
        assert clusterer is not None
        assert hasattr(clusterer, 'perform_clustering')
        assert hasattr(clusterer, 'create_cluster_visualization')
    
    def test_perform_clustering_success(self, clusterer, sample_features):
        """Test clustering with valid data"""
        image_paths, features = sample_features
        
        results = clusterer.perform_clustering(image_paths, features, n_clusters=2)
        
        assert results is not None
        assert isinstance(results, dict)
        assert 'labels' in results
        assert 'centers' in results
        assert 'silhouette_score' in results
        assert 'inertia' in results
        
        assert len(results['labels']) == len(image_paths)
        assert results['centers'].shape == (2, features.shape[1])
        assert -1 <= results['silhouette_score'] <= 1
        assert results['inertia'] >= 0
    
    def test_perform_clustering_single_cluster(self, clusterer):
        """Test clustering with single cluster"""
        image_paths = ['image1.jpg']
        features = np.array([[1, 2, 3]])
        
        results = clusterer.perform_clustering(image_paths, features, n_clusters=1)
        
        assert results['labels'][0] == 0
        assert results['centers'].shape == (1, 3)
    
    def test_determine_optimal_clusters(self, clusterer, sample_features):
        """Test optimal cluster number determination"""
        _, features = sample_features
        
        optimal_k = clusterer.determine_optimal_clusters(features, max_k=5)
        
        assert isinstance(optimal_k, int)
        assert 1 <= optimal_k <= 5
    
    def test_organize_clusters(self, clusterer, sample_features):
        """Test cluster organization"""
        image_paths, features = sample_features
        labels = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        
        clusters = clusterer.organize_clusters(image_paths, labels)
        
        assert isinstance(clusters, dict)
        assert len(clusters) == 2  # Two clusters (0 and 1)
        assert 0 in clusters
        assert 1 in clusters
        assert len(clusters[0]) == 5  # 5 images in cluster 0
        assert len(clusters[1]) == 5  # 5 images in cluster 1
    
    def test_create_cluster_visualization(self, clusterer, sample_features):
        """Test cluster visualization creation"""
        image_paths, features = sample_features
        labels = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy images for visualization
            for path in image_paths:
                img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
                cv2.imwrite(str(Path(tmpdir) / path), img)
            
            # Update paths to point to temp directory
            full_paths = [str(Path(tmpdir) / path) for path in image_paths]
            
            viz_path = Path(tmpdir) / 'cluster_viz.png'
            success = clusterer.create_cluster_visualization(full_paths, labels, str(viz_path))
            
            assert success
            assert viz_path.exists()
    
    def test_save_results(self, clusterer, sample_features):
        """Test saving clustering results"""
        image_paths, features = sample_features
        results = {
            'labels': np.array([0, 1, 0, 1, 1]),
            'centers': np.array([[1, 2], [3, 4]]),
            'silhouette_score': 0.5,
            'inertia': 10.0
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / 'results.json'
            clusterer.save_results(results, image_paths, str(output_file))
            
            assert output_file.exists()
            
            with open(output_file, 'r') as f:
                saved_data = json.load(f)
            
            assert 'clustering_info' in saved_data
            assert 'photos' in saved_data
            assert saved_data['clustering_info']['silhouette_score'] == 0.5


class TestPhotoClusteringIntegration:
    """Integration tests for complete clustering workflow"""
    
    def test_end_to_end_clustering(self):
        """Test complete clustering workflow from images to organized clusters"""
        processor = PhotoProcessor()
        clusterer = PhotoClusterer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images with two distinct patterns
            for i in range(6):
                if i < 3:
                    # Dark images
                    img = np.full((100, 100, 3), 50, dtype=np.uint8)
                else:
                    # Bright images  
                    img = np.full((100, 100, 3), 200, dtype=np.uint8)
                
                img_path = Path(tmpdir) / f"test_image_{i}.jpg"
                cv2.imwrite(str(img_path), img)
            
            # Process images
            image_paths, features = processor.process_directory(tmpdir)
            
            assert len(image_paths) == 6
            assert len(features) == 6
            
            # Perform clustering
            results = clusterer.perform_clustering(image_paths, features, n_clusters=2)
            
            assert len(results['labels']) == 6
            assert len(set(results['labels'])) <= 2  # At most 2 clusters
            
            # Organize clusters
            clusters = clusterer.organize_clusters(image_paths, results['labels'])
            assert isinstance(clusters, dict)
            assert len(clusters) <= 2
    
    def test_cluster_directory_creation(self):
        """Test creating organized cluster directories"""
        clusterer = PhotoClusterer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test setup
            image_paths = [f'image_{i}.jpg' for i in range(4)]
            labels = np.array([0, 0, 1, 1])
            
            # Create actual test images
            for path in image_paths:
                img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
                cv2.imwrite(str(Path(tmpdir) / path), img)
            
            clusters = clusterer.organize_clusters(image_paths, labels)
            
            # Create cluster directories (simulate the organize_into_directories function)
            clusters_dir = Path(tmpdir) / 'clusters'
            clusters_dir.mkdir()
            
            for cluster_id, cluster_images in clusters.items():
                cluster_dir = clusters_dir / f'cluster_{cluster_id}'
                cluster_dir.mkdir()
                
                for img_path in cluster_images:
                    # Copy images to cluster directory
                    src = Path(tmpdir) / Path(img_path).name
                    dst = cluster_dir / Path(img_path).name
                    if src.exists():
                        import shutil
                        shutil.copy2(src, dst)
            
            # Verify cluster directories were created
            assert (clusters_dir / 'cluster_0').exists()
            assert (clusters_dir / 'cluster_1').exists()
            
            # Verify images were copied
            cluster_0_files = list((clusters_dir / 'cluster_0').glob('*.jpg'))
            cluster_1_files = list((clusters_dir / 'cluster_1').glob('*.jpg'))
            
            assert len(cluster_0_files) == 2
            assert len(cluster_1_files) == 2


class TestFeatureExtraction:
    """Test feature extraction methods"""
    
    def test_color_histogram_features(self):
        """Test color histogram feature extraction"""
        processor = PhotoProcessor()
        
        # Create image with known colors
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50, :, 0] = 255  # Top half red
        img[50:, :, 2] = 255  # Bottom half blue
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            features = processor.extract_features(tmp.name)
            
            assert features is not None
            assert len(features) > 0
            
        os.unlink(tmp.name)
    
    def test_texture_features(self):
        """Test texture feature extraction consistency"""
        processor = PhotoProcessor()
        
        # Create two identical images
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp1:
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp2:
                cv2.imwrite(tmp1.name, img)
                cv2.imwrite(tmp2.name, img)
                
                features1 = processor.extract_features(tmp1.name)
                features2 = processor.extract_features(tmp2.name)
                
                # Features should be very similar for identical images
                similarity = np.corrcoef(features1, features2)[0, 1]
                assert similarity > 0.9  # Should be highly correlated
        
        os.unlink(tmp1.name)
        os.unlink(tmp2.name)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])