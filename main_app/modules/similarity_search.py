#!/usr/bin/env python3
"""
CLIP-based similarity search for finding visually and textually similar photos.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
import json

class PhotoSimilaritySearch:
    def __init__(self):
        self.visual_embeddings = {}
        self.text_embeddings = {}
        self.photo_metadata = {}
        self.embeddings_loaded = False
        
    def load_embeddings(self) -> bool:
        """Load pre-computed CLIP embeddings from disk"""
        embeddings_dir = Path("main_app_outputs/embeddings")
        
        try:
            # Load visual embeddings
            visual_path = embeddings_dir / "visual_embeddings.pkl"
            if visual_path.exists():
                with open(visual_path, 'rb') as f:
                    self.visual_embeddings = pickle.load(f)
                print(f"✅ Loaded {len(self.visual_embeddings)} visual embeddings")
            
            # Load text embeddings  
            text_path = embeddings_dir / "text_embeddings.pkl"
            if text_path.exists():
                with open(text_path, 'rb') as f:
                    self.text_embeddings = pickle.load(f)
                print(f"✅ Loaded {len(self.text_embeddings)} text embeddings")
            
            # Load metadata
            metadata_path = embeddings_dir / "photo_metadata.pkl"
            if metadata_path.exists():
                with open(metadata_path, 'rb') as f:
                    self.photo_metadata = pickle.load(f)
                print(f"✅ Loaded metadata for {len(self.photo_metadata)} photos")
            
            self.embeddings_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            return False
    
    def find_similar_photos_visual(self, photo_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find visually similar photos using CLIP visual embeddings"""
        if not self.embeddings_loaded:
            if not self.load_embeddings():
                return []
        
        if photo_path not in self.visual_embeddings:
            print(f"❌ Photo not found in visual embeddings: {photo_path}")
            return []
        
        query_embedding = self.visual_embeddings[photo_path]
        similarities = []
        
        for other_path, other_embedding in self.visual_embeddings.items():
            if other_path != photo_path:  # Skip self
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    [query_embedding], 
                    [other_embedding]
                )[0][0]
                similarities.append((other_path, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def find_similar_photos_text(self, photo_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find textually similar photos using CLIP text embeddings"""
        if not self.embeddings_loaded:
            if not self.load_embeddings():
                return []
        
        if photo_path not in self.text_embeddings:
            print(f"❌ Photo not found in text embeddings: {photo_path}")
            return []
        
        query_embedding = self.text_embeddings[photo_path]
        similarities = []
        
        for other_path, other_embedding in self.text_embeddings.items():
            if other_path != photo_path:  # Skip self
                # Calculate cosine similarity
                similarity = cosine_similarity(
                    [query_embedding], 
                    [other_embedding]
                )[0][0]
                similarities.append((other_path, similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def find_similar_photos_combined(self, photo_path: str, top_k: int = 5, visual_weight: float = 0.5) -> List[Tuple[str, float]]:
        """Find similar photos using combined visual and text embeddings"""
        if not self.embeddings_loaded:
            if not self.load_embeddings():
                return []
        
        if photo_path not in self.visual_embeddings or photo_path not in self.text_embeddings:
            print(f"❌ Photo not found in embeddings: {photo_path}")
            return []
        
        query_visual = self.visual_embeddings[photo_path]
        query_text = self.text_embeddings[photo_path]
        similarities = []
        
        for other_path in self.visual_embeddings:
            if other_path != photo_path:  # Skip self
                # Calculate visual similarity
                visual_sim = cosine_similarity(
                    [query_visual], 
                    [self.visual_embeddings[other_path]]
                )[0][0]
                
                # Calculate text similarity
                text_sim = cosine_similarity(
                    [query_text], 
                    [self.text_embeddings[other_path]]
                )[0][0]
                
                # Combine similarities with weighting
                combined_sim = (visual_weight * visual_sim) + ((1 - visual_weight) * text_sim)
                similarities.append((other_path, combined_sim))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_photo_similarities(self, photo_path: str, top_k: int = 3) -> Dict[str, List[Dict[str, any]]]:
        """Get both visual and text similarities for a photo"""
        result = {
            'visual_similar': [],
            'text_similar': [],
            'combined_similar': []
        }
        
        # Get visual similarities
        visual_similar = self.find_similar_photos_visual(photo_path, top_k)
        for similar_path, score in visual_similar:
            metadata = self.photo_metadata.get(similar_path, {})
            result['visual_similar'].append({
                'path': similar_path,
                'filename': metadata.get('filename', Path(similar_path).name),
                'similarity_score': float(score),
                'text': metadata.get('text', '')
            })
        
        # Get text similarities
        text_similar = self.find_similar_photos_text(photo_path, top_k)
        for similar_path, score in text_similar:
            metadata = self.photo_metadata.get(similar_path, {})
            result['text_similar'].append({
                'path': similar_path,
                'filename': metadata.get('filename', Path(similar_path).name),
                'similarity_score': float(score),
                'text': metadata.get('text', '')
            })
        
        # Get combined similarities
        combined_similar = self.find_similar_photos_combined(photo_path, top_k)
        for similar_path, score in combined_similar:
            metadata = self.photo_metadata.get(similar_path, {})
            result['combined_similar'].append({
                'path': similar_path,
                'filename': metadata.get('filename', Path(similar_path).name),
                'similarity_score': float(score),
                'text': metadata.get('text', '')
            })
        
        return result
    
    def get_embeddings_stats(self) -> Dict[str, any]:
        """Get statistics about loaded embeddings"""
        if not self.embeddings_loaded:
            self.load_embeddings()
        
        return {
            'total_photos': len(self.visual_embeddings),
            'has_visual_embeddings': len(self.visual_embeddings) > 0,
            'has_text_embeddings': len(self.text_embeddings) > 0,
            'embedding_dimension': 512,
            'embeddings_ready': self.embeddings_loaded
        }


# Global instance for use by the dashboard
similarity_search = PhotoSimilaritySearch()


def get_photo_similarities(photo_path: str, top_k: int = 3) -> Dict[str, List[Dict[str, any]]]:
    """
    Convenience function to get photo similarities.
    Returns visual, text, and combined similarities for dashboard integration.
    """
    return similarity_search.get_photo_similarities(photo_path, top_k)


def get_visual_similarities(photo_path: str, top_k: int = 5) -> List[Dict[str, any]]:
    """Get only visual similarities for a photo"""
    visual_similar = similarity_search.find_similar_photos_visual(photo_path, top_k)
    result = []
    for similar_path, score in visual_similar:
        metadata = similarity_search.photo_metadata.get(similar_path, {})
        result.append({
            'path': similar_path,
            'filename': metadata.get('filename', Path(similar_path).name),
            'similarity_score': float(score),
            'text': metadata.get('text', '')
        })
    return result


def get_combined_similarities(photo_path: str, top_k: int = 5) -> List[Dict[str, any]]:
    """Get combined visual+text similarities for a photo"""
    combined_similar = similarity_search.find_similar_photos_combined(photo_path, top_k)
    result = []
    for similar_path, score in combined_similar:
        metadata = similarity_search.photo_metadata.get(similar_path, {})
        result.append({
            'path': similar_path,
            'filename': metadata.get('filename', Path(similar_path).name),
            'similarity_score': float(score),
            'text': metadata.get('text', '')
        })
    return result


def get_embeddings_status() -> Dict[str, any]:
    """Get status of embeddings for dashboard"""
    return similarity_search.get_embeddings_stats()


if __name__ == "__main__":
    # Test the similarity search
    search = PhotoSimilaritySearch()
    stats = search.get_embeddings_stats()
    print(f"📊 Embeddings stats: {json.dumps(stats, indent=2)}")
    
    if stats['embeddings_ready'] and stats['total_photos'] > 0:
        # Test with first photo
        first_photo = list(search.visual_embeddings.keys())[0]
        print(f"🔍 Testing similarity search with: {Path(first_photo).name}")
        
        similarities = search.get_photo_similarities(first_photo, top_k=3)
        
        print("\n📸 Visual similarities:")
        for sim in similarities['visual_similar']:
            print(f"  {sim['filename']} (score: {sim['similarity_score']:.3f})")
        
        print("\n📝 Text similarities:")
        for sim in similarities['text_similar']:
            print(f"  {sim['filename']} (score: {sim['similarity_score']:.3f})")
    else:
        print("❌ No embeddings found. Run generate_clip_embeddings.py first.")