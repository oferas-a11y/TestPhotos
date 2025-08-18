#!/usr/bin/env python3
"""
Integration tests for complete photo analysis workflows
Tests the interaction between different components
"""

import pytest
import numpy as np
import json
import tempfile
import cv2
from pathlib import Path
from PIL import Image
import sys
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestCompleteWorkflow:
    """Test complete end-to-end photo analysis workflow"""
    
    def test_multi_tool_analysis_pipeline(self, sample_photos_dir):
        """Test running all analysis tools on the same photo set"""
        photos_dir = Path(sample_photos_dir)
        
        # Import all analyzers (with mocking where needed)
        from opencv_analysis.photo_analyzer import PhotoAnalyzer
        
        with patch('yolo_detection.yolo_detector.YOLO'), \
             patch('clip_analysis.clip_detector.clip.load'):
            
            from yolo_detection.yolo_detector import YOLODetector
            from clip_analysis.clip_detector import CLIPDetector
        
        # Initialize analyzers
        opencv_analyzer = PhotoAnalyzer()
        yolo_detector = YOLODetector()
        clip_detector = CLIPDetector()
        
        # Mock YOLO and CLIP for consistent results
        yolo_detector.analyze_photo = Mock(return_value={
            'filename': 'portrait.jpg',
            'total_objects': 2,
            'object_counts': {'person': 1, 'chair': 1},
            'priority_score': 15
        })
        
        clip_detector.analyze_photo = Mock(return_value={
            'filename': 'portrait.jpg', 
            'total_detections': 3,
            'category_counts': {'family portrait': 1, 'period furniture': 1, 'formal clothing': 1},
            'priority_score': 18.5
        })
        
        # Test photo
        test_photo = photos_dir / 'portrait.jpg'
        
        # Run all analyses
        opencv_result = opencv_analyzer.analyze_image(str(test_photo))
        yolo_result = yolo_detector.analyze_photo(str(test_photo))
        clip_result = clip_detector.analyze_photo(str(test_photo))
        
        # Verify all analyses completed
        assert opencv_result is not None
        assert yolo_result is not None
        assert clip_result is not None
        
        # Verify each has expected structure
        assert 'brightness' in opencv_result
        assert 'total_objects' in yolo_result
        assert 'total_detections' in clip_result
        
        # Test combined results structure
        combined_result = {
            'filename': opencv_result['filename'],
            'opencv_analysis': opencv_result,
            'yolo_detection': yolo_result,
            'clip_semantics': clip_result,
            'analysis_summary': {
                'technical_quality': opencv_result['brightness'],
                'objects_detected': yolo_result['total_objects'], 
                'semantic_categories': clip_result['total_detections']
            }
        }
        
        # Verify combined result can be serialized
        json_str = json.dumps(combined_result, indent=2)
        assert isinstance(json_str, str)
    
    def test_clustering_with_analysis_integration(self, sample_photos_dir):
        """Test photo clustering integrated with analysis results"""
        from photo_clustering.process_photos import PhotoProcessor
        from photo_clustering.kmeans_clustering import PhotoClusterer
        from opencv_analysis.photo_analyzer import PhotoAnalyzer
        
        photos_dir = Path(sample_photos_dir)
        processor = PhotoProcessor()
        clusterer = PhotoClusterer()
        analyzer = PhotoAnalyzer()
        
        # Extract features for clustering
        image_paths, features = processor.process_directory(str(photos_dir))
        
        assert len(image_paths) > 0
        assert len(features) == len(image_paths)
        
        # Perform clustering
        results = clusterer.perform_clustering(image_paths, features, n_clusters=2)
        
        assert 'labels' in results
        assert len(results['labels']) == len(image_paths)
        
        # Analyze photos in each cluster
        clusters = clusterer.organize_clusters(image_paths, results['labels'])
        
        cluster_analyses = {}
        for cluster_id, cluster_images in clusters.items():
            cluster_analyses[cluster_id] = []
            
            for img_path in cluster_images[:2]:  # Analyze first 2 in each cluster
                full_path = str(photos_dir / Path(img_path).name)
                analysis = analyzer.analyze_image(full_path)
                if analysis:
                    cluster_analyses[cluster_id].append(analysis)
        
        # Verify cluster analysis results
        for cluster_id, analyses in cluster_analyses.items():
            assert isinstance(analyses, list)
            for analysis in analyses:
                assert 'brightness' in analysis
                assert 'contrast' in analysis
    
    def test_results_consistency_across_tools(self, single_test_image):
        """Test that different tools can analyze the same image consistently"""
        from opencv_analysis.photo_analyzer import PhotoAnalyzer
        
        with patch('yolo_detection.yolo_detector.YOLO'), \
             patch('clip_analysis.clip_detector.clip.load'):
            
            from yolo_detection.yolo_detector import YOLODetector
            from clip_analysis.clip_detector import CLIPDetector
        
        # Initialize tools
        opencv_analyzer = PhotoAnalyzer()
        yolo_detector = YOLODetector()
        clip_detector = CLIPDetector()
        
        # Mock for consistent testing
        yolo_detector.detect_objects = Mock(return_value=[])
        clip_detector.analyze_image_semantics = Mock(return_value=[])
        
        # Analyze same image with all tools
        opencv_result = opencv_analyzer.analyze_image(single_test_image)
        yolo_result = yolo_detector.analyze_photo(single_test_image)
        clip_result = clip_detector.analyze_photo(single_test_image)
        
        # All should have processed the same filename
        filename = Path(single_test_image).name
        assert opencv_result['filename'] == filename
        assert yolo_result['filename'] == filename
        assert clip_result['filename'] == filename
        
        # All should complete without errors
        assert opencv_result is not None
        assert yolo_result is not None
        assert clip_result is not None


class TestDataFlowIntegration:
    """Test data flow between different components"""
    
    def test_feature_extraction_to_clustering_pipeline(self, sample_photos_dir):
        """Test complete pipeline from feature extraction to cluster organization"""
        from photo_clustering.process_photos import PhotoProcessor
        from photo_clustering.kmeans_clustering import PhotoClusterer
        
        processor = PhotoProcessor()
        clusterer = PhotoClusterer()
        
        # Step 1: Extract features
        image_paths, features = processor.process_directory(sample_photos_dir)
        
        assert len(image_paths) > 0
        assert len(features) > 0
        assert len(features) == len(image_paths)
        
        # Step 2: Perform clustering
        clustering_results = clusterer.perform_clustering(image_paths, features, n_clusters=2)
        
        assert 'labels' in clustering_results
        assert 'centers' in clustering_results
        
        # Step 3: Organize into clusters
        organized_clusters = clusterer.organize_clusters(image_paths, clustering_results['labels'])
        
        assert isinstance(organized_clusters, dict)
        assert len(organized_clusters) <= 2  # At most 2 clusters
        
        # Verify all images are assigned to clusters
        total_assigned = sum(len(cluster_images) for cluster_images in organized_clusters.values())
        assert total_assigned == len(image_paths)
    
    def test_analysis_results_aggregation(self, mock_analysis_results):
        """Test aggregating results from multiple analysis tools"""
        # Simulate results from different tools
        opencv_results = [
            {'filename': 'test1.jpg', 'brightness': 150, 'contrast': 75, 'edge_density': 0.3},
            {'filename': 'test2.jpg', 'brightness': 120, 'contrast': 60, 'edge_density': 0.4},
            {'filename': 'test3.jpg', 'brightness': 90, 'contrast': 45, 'edge_density': 0.2}
        ]
        
        yolo_results = [
            {'filename': 'test1.jpg', 'total_objects': 2, 'object_counts': {'person': 1, 'chair': 1}},
            {'filename': 'test2.jpg', 'total_objects': 1, 'object_counts': {'person': 1}}, 
            {'filename': 'test3.jpg', 'total_objects': 0, 'object_counts': {}}
        ]
        
        clip_results = [
            {'filename': 'test1.jpg', 'total_detections': 3, 'priority_score': 15.2},
            {'filename': 'test2.jpg', 'total_detections': 2, 'priority_score': 12.1},
            {'filename': 'test3.jpg', 'total_detections': 1, 'priority_score': 8.5}
        ]
        
        # Aggregate results by filename
        aggregated = {}
        
        for opencv_result in opencv_results:
            filename = opencv_result['filename']
            aggregated[filename] = {'opencv': opencv_result}
        
        for yolo_result in yolo_results:
            filename = yolo_result['filename']
            if filename in aggregated:
                aggregated[filename]['yolo'] = yolo_result
        
        for clip_result in clip_results:
            filename = clip_result['filename']
            if filename in aggregated:
                aggregated[filename]['clip'] = clip_result
        
        # Verify aggregation
        assert len(aggregated) == 3
        
        for filename, results in aggregated.items():
            assert 'opencv' in results
            assert 'yolo' in results
            assert 'clip' in results
            
            # Verify filename consistency
            assert results['opencv']['filename'] == filename
            assert results['yolo']['filename'] == filename
            assert results['clip']['filename'] == filename
    
    def test_json_serialization_of_complete_results(self, mock_yolo_detections, mock_clip_detections):
        """Test that complete analysis results can be serialized to JSON"""
        complete_result = {
            'photo_info': {
                'filename': 'test_photo.jpg',
                'analysis_timestamp': '2024-01-01T12:00:00Z',
                'file_size': 1024000
            },
            'opencv_analysis': {
                'brightness': 145.5,
                'contrast': 78.2,
                'dominant_colors': [[255, 200, 150], [100, 150, 200]],
                'edge_density': 0.35,
                'texture_score': 42.1
            },
            'yolo_detection': {
                'total_objects': len(mock_yolo_detections),
                'detections': mock_yolo_detections,
                'priority_score': 25.5
            },
            'clip_semantics': {
                'total_detections': len(mock_clip_detections),
                'detections': mock_clip_detections,
                'priority_score': 18.7
            },
            'summary': {
                'analysis_complete': True,
                'total_analysis_tools': 3,
                'objects_found': len(mock_yolo_detections),
                'semantic_categories': len(mock_clip_detections)
            }
        }
        
        # Should serialize without errors
        json_str = json.dumps(complete_result, indent=2)
        assert isinstance(json_str, str)
        
        # Should deserialize back to same structure
        loaded_result = json.loads(json_str)
        assert loaded_result['photo_info']['filename'] == 'test_photo.jpg'
        assert loaded_result['summary']['analysis_complete'] is True


class TestErrorHandlingIntegration:
    """Test error handling across integrated workflows"""
    
    def test_partial_analysis_failure_handling(self, single_test_image):
        """Test handling when some analysis tools fail"""
        from opencv_analysis.photo_analyzer import PhotoAnalyzer
        
        with patch('yolo_detection.yolo_detector.YOLO'), \
             patch('clip_analysis.clip_detector.clip.load'):
            
            from yolo_detection.yolo_detector import YOLODetector
            from clip_analysis.clip_detector import CLIPDetector
        
        opencv_analyzer = PhotoAnalyzer()
        yolo_detector = YOLODetector()
        clip_detector = CLIPDetector()
        
        # Mock one tool to fail
        yolo_detector.analyze_photo = Mock(return_value=None)  # Simulate failure
        clip_detector.analyze_photo = Mock(return_value={
            'filename': Path(single_test_image).name,
            'total_detections': 2,
            'priority_score': 10.0
        })
        
        # Collect results, handling failures
        results = {}
        
        try:
            results['opencv'] = opencv_analyzer.analyze_image(single_test_image)
        except Exception:
            results['opencv'] = None
        
        try:
            results['yolo'] = yolo_detector.analyze_photo(single_test_image)
        except Exception:
            results['yolo'] = None
        
        try:
            results['clip'] = clip_detector.analyze_photo(single_test_image)
        except Exception:
            results['clip'] = None
        
        # Verify partial success handling
        assert results['opencv'] is not None  # Should succeed
        assert results['yolo'] is None        # Mocked to fail
        assert results['clip'] is not None    # Should succeed
        
        # Create summary with available results
        successful_analyses = [k for k, v in results.items() if v is not None]
        assert len(successful_analyses) == 2
        assert 'opencv' in successful_analyses
        assert 'clip' in successful_analyses
    
    def test_empty_directory_handling(self):
        """Test handling of empty directories across all tools"""
        from opencv_analysis.photo_analyzer import PhotoAnalyzer
        from photo_clustering.process_photos import PhotoProcessor
        
        with patch('yolo_detection.yolo_detector.YOLO'), \
             patch('clip_analysis.clip_detector.clip.load'):
            
            from yolo_detection.yolo_detector import YOLODetector
            from clip_analysis.clip_detector import CLIPDetector
        
        with tempfile.TemporaryDirectory() as empty_dir:
            opencv_analyzer = PhotoAnalyzer()
            processor = PhotoProcessor()
            yolo_detector = YOLODetector()
            clip_detector = CLIPDetector()
            
            # Mock directory analysis methods
            yolo_detector.analyze_directory = Mock(return_value=[])
            clip_detector.analyze_directory = Mock(return_value=[])
            
            # Test all tools with empty directory
            opencv_results = opencv_analyzer.analyze_directory(empty_dir)
            clustering_paths, clustering_features = processor.process_directory(empty_dir)
            yolo_results = yolo_detector.analyze_directory(empty_dir)
            clip_results = clip_detector.analyze_directory(empty_dir)
            
            # All should handle empty directory gracefully
            assert opencv_results == []
            assert clustering_paths == []
            assert clustering_features == []
            assert yolo_results == []
            assert clip_results == []
    
    def test_invalid_image_handling_across_tools(self):
        """Test how all tools handle invalid/corrupted images"""
        from opencv_analysis.photo_analyzer import PhotoAnalyzer
        from photo_clustering.process_photos import PhotoProcessor
        
        with patch('yolo_detection.yolo_detector.YOLO'), \
             patch('clip_analysis.clip_detector.clip.load'):
            
            from yolo_detection.yolo_detector import YOLODetector
            from clip_analysis.clip_detector import CLIPDetector
        
        # Create invalid image file (empty)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b'invalid image data')
            invalid_image_path = tmp.name
        
        try:
            opencv_analyzer = PhotoAnalyzer()
            processor = PhotoProcessor()
            yolo_detector = YOLODetector()
            clip_detector = CLIPDetector()
            
            # Mock methods to simulate error handling
            yolo_detector.detect_objects = Mock(return_value=[])
            clip_detector.analyze_image_semantics = Mock(return_value=[])
            
            # Test all tools with invalid image
            opencv_result = opencv_analyzer.analyze_image(invalid_image_path)
            clustering_features = processor.extract_features(invalid_image_path)
            yolo_result = yolo_detector.analyze_photo(invalid_image_path)
            clip_result = clip_detector.analyze_photo(invalid_image_path)
            
            # Tools should handle errors gracefully (return None or empty results)
            # OpenCV analyzer might return None for invalid images
            assert opencv_result is None or isinstance(opencv_result, dict)
            assert clustering_features is None or isinstance(clustering_features, np.ndarray)
            # YOLO and CLIP should return valid structures even with no detections
            assert yolo_result is None or isinstance(yolo_result, dict)
            assert clip_result is None or isinstance(clip_result, dict)
            
        finally:
            os.unlink(invalid_image_path)


class TestPerformanceIntegration:
    """Test performance characteristics of integrated workflows"""
    
    @pytest.mark.slow
    def test_batch_processing_performance(self, sample_photos_dir):
        """Test performance of batch processing multiple photos"""
        from opencv_analysis.photo_analyzer import PhotoAnalyzer
        import time
        
        opencv_analyzer = PhotoAnalyzer()
        
        # Time the batch processing
        start_time = time.time()
        results = opencv_analyzer.analyze_directory(sample_photos_dir)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Basic performance checks
        assert len(results) > 0
        assert processing_time < 30  # Should complete within 30 seconds for test images
        
        # Calculate average time per image
        avg_time_per_image = processing_time / len(results)
        assert avg_time_per_image < 10  # Should be under 10 seconds per image
    
    def test_memory_usage_with_large_dataset(self, sample_photos_dir):
        """Test memory usage patterns with multiple images"""
        from photo_clustering.process_photos import PhotoProcessor
        import psutil
        import os
        
        processor = PhotoProcessor()
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process images
        image_paths, features = processor.process_directory(sample_photos_dir)
        
        # Get final memory usage  
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for test images)
        assert memory_increase < 500 * 1024 * 1024  # 500MB
        
        # Features should be generated
        assert len(features) > 0
        assert len(features) == len(image_paths)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])