#!/usr/bin/env python3
"""
K-means Clustering Script for Historical Photos
Creates clusters and visualizes results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import shutil
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class PhotoClusterer:
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.pca = PCA(n_components=50)  # Reduce dimensions for better clustering
        
    def load_features(self, features_file):
        """Load features from pickle file"""
        try:
            with open(features_file, 'rb') as f:
                data = pickle.load(f)
            return data['features'], data['image_paths']
        except Exception as e:
            print(f"Error loading features: {e}")
            return None, None
    
    def perform_clustering(self, features):
        """Perform K-means clustering"""
        print(f"Performing PCA dimensionality reduction...")
        # Apply PCA for better clustering
        features_pca = self.pca.fit_transform(features)
        print(f"Features reduced from {features.shape[1]} to {features_pca.shape[1]} dimensions")
        
        print(f"Performing K-means clustering with {self.n_clusters} clusters...")
        cluster_labels = self.kmeans.fit_predict(features_pca)
        
        return cluster_labels, features_pca
    
    def create_cluster_directories(self, image_paths, cluster_labels, output_dir="clusters"):
        """Create directories for each cluster and copy images"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create cluster directories
        for i in range(self.n_clusters):
            cluster_dir = output_dir / f"cluster_{i}"
            cluster_dir.mkdir(exist_ok=True)
        
        # Copy images to cluster directories
        for image_path, label in zip(image_paths, cluster_labels):
            src_path = Path(image_path)
            dst_path = output_dir / f"cluster_{label}" / src_path.name
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Error copying {src_path}: {e}")
        
        print(f"Cluster directories created in {output_dir}")
        return output_dir
    
    def visualize_clusters_2d(self, features_pca, cluster_labels, image_paths, save_path="cluster_visualization.png"):
        """Create 2D visualization of clusters"""
        print("Creating 2D visualization...")
        
        # Further reduce to 2D for visualization
        if features_pca.shape[1] > 2:
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features_pca)-1))
            features_2d = tsne.fit_transform(features_pca)
        else:
            features_2d = features_pca
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_clusters))
        
        for i in range(self.n_clusters):
            mask = cluster_labels == i
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[colors[i]], label=f'Cluster {i} ({np.sum(mask)} photos)', 
                       alpha=0.7, s=50)
        
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(f'K-means Clustering Results ({self.n_clusters} clusters)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"2D visualization saved to {save_path}")
        plt.show()
    
    def create_cluster_summary(self, image_paths, cluster_labels, output_dir):
        """Create a visual summary of each cluster"""
        output_dir = Path(output_dir)
        
        for cluster_id in range(self.n_clusters):
            cluster_images = [path for path, label in zip(image_paths, cluster_labels) if label == cluster_id]
            
            if not cluster_images:
                continue
            
            # Create a grid of sample images from this cluster
            n_samples = min(9, len(cluster_images))  # Show up to 9 images per cluster
            sample_images = cluster_images[:n_samples]
            
            fig, axes = plt.subplots(3, 3, figsize=(12, 12))
            fig.suptitle(f'Cluster {cluster_id} - {len(cluster_images)} photos', fontsize=16)
            
            for i, ax in enumerate(axes.flat):
                if i < len(sample_images):
                    try:
                        img = Image.open(sample_images[i])
                        ax.imshow(img)
                        ax.set_title(Path(sample_images[i]).name, fontsize=8)
                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error loading\n{Path(sample_images[i]).name}', 
                               ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
            
            plt.tight_layout()
            summary_path = output_dir / f"cluster_{cluster_id}_summary.png"
            plt.savefig(summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Cluster {cluster_id} summary saved to {summary_path}")
    
    def analyze_clusters(self, image_paths, cluster_labels):
        """Print analysis of cluster distribution"""
        print("\n" + "="*50)
        print("CLUSTER ANALYSIS")
        print("="*50)
        
        for i in range(self.n_clusters):
            cluster_images = [path for path, label in zip(image_paths, cluster_labels) if label == i]
            print(f"Cluster {i}: {len(cluster_images)} photos")
            if cluster_images:
                # Show first few image names
                sample_names = [Path(img).name for img in cluster_images[:3]]
                print(f"  Sample images: {', '.join(sample_names)}")
                if len(cluster_images) > 3:
                    print(f"  ... and {len(cluster_images) - 3} more")
            print()

def main():
    # Default settings
    features_file = 'photo_features.pkl'
    n_clusters = 10
    output_dir = 'clusters'
    visualize = True
    
    # Create clusterer
    clusterer = PhotoClusterer(n_clusters=n_clusters)
    
    # Load features
    print("Loading features...")
    features, image_paths = clusterer.load_features(features_file)
    
    if features is None:
        print("Failed to load features. Make sure to run process_photos.py first.")
        return
    
    print(f"Loaded {len(image_paths)} images with {features.shape[1]} features each")
    
    # Perform clustering
    cluster_labels, features_pca = clusterer.perform_clustering(features)
    
    # Analyze clusters
    clusterer.analyze_clusters(image_paths, cluster_labels)
    
    # Create cluster directories
    output_dir = clusterer.create_cluster_directories(image_paths, cluster_labels, output_dir)
    
    if visualize:
        # Create 2D visualization
        clusterer.visualize_clusters_2d(features_pca, cluster_labels, image_paths, 
                                       f"{output_dir}/cluster_visualization.png")
        
        # Create cluster summaries
        clusterer.create_cluster_summary(image_paths, cluster_labels, output_dir)
    
    print(f"\nClustering complete! Results saved in '{output_dir}' directory")

if __name__ == "__main__":
    main()