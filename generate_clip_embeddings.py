#!/usr/bin/env python3
"""
Generate CLIP embeddings for all photos in the collection.
Creates both visual and text embeddings for similarity search.
"""

import os
import json
import csv
import pickle
import torch
import clip
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

class CLIPEmbeddingGenerator:
    def __init__(self):
        print("🚀 Initializing CLIP model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"📱 Using device: {self.device}")
        
        # Load CLIP model
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print("✅ CLIP model loaded successfully")
        
        # Storage for embeddings
        self.visual_embeddings = {}
        self.text_embeddings = {}
        self.photo_metadata = {}
        
    def load_photo_data(self) -> List[Dict[str, Any]]:
        """Load photo data from data_text.csv"""
        csv_path = Path("main_app_outputs/results/data_text.csv")
        if not csv_path.exists():
            print("❌ data_text.csv not found. Run photo processing first")
            return []
        
        photos = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                photos.append({
                    'path': row['original_path'],
                    'text': row['comprehensive_text']
                })
        
        print(f"📚 Loaded {len(photos)} photos from CSV")
        return photos
    
    def generate_visual_embedding(self, image_path: str) -> np.ndarray:
        """Generate CLIP visual embedding for an image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                # Normalize embeddings
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"❌ Error processing image {image_path}: {e}")
            return np.zeros(512)  # Return zero embedding on error
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate CLIP text embedding for text description"""
        try:
            # Tokenize text
            text_input = clip.tokenize([text], truncate=True).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                text_features = self.model.encode_text(text_input)
                # Normalize embeddings
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
            return text_features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"❌ Error processing text: {e}")
            return np.zeros(512)  # Return zero embedding on error
    
    def process_all_photos(self):
        """Process all photos and generate embeddings"""
        photos = self.load_photo_data()
        
        if not photos:
            return
        
        print(f"🔄 Processing {len(photos)} photos for CLIP embeddings...")
        
        successful = 0
        failed = 0
        
        for i, photo in enumerate(photos, 1):
            photo_path = photo['path']
            photo_text = photo['text']
            
            print(f"📸 [{i}/{len(photos)}] Processing: {Path(photo_path).name}")
            
            # Check if image file exists
            if not os.path.exists(photo_path):
                print(f"  ⚠️  Image file not found: {photo_path}")
                failed += 1
                continue
            
            try:
                # Generate visual embedding
                visual_emb = self.generate_visual_embedding(photo_path)
                
                # Generate text embedding
                text_emb = self.generate_text_embedding(photo_text)
                
                # Store embeddings with photo path as key
                photo_key = photo_path
                self.visual_embeddings[photo_key] = visual_emb
                self.text_embeddings[photo_key] = text_emb
                self.photo_metadata[photo_key] = {
                    'text': photo_text,
                    'filename': Path(photo_path).name
                }
                
                successful += 1
                
                # Progress update every 50 photos
                if i % 50 == 0:
                    print(f"  ✅ Processed {i}/{len(photos)} photos ({successful} successful, {failed} failed)")
                    
            except Exception as e:
                print(f"  ❌ Error processing {photo_path}: {e}")
                failed += 1
        
        print(f"""
🎉 Embedding generation complete!
✅ Successfully processed: {successful} photos
❌ Failed: {failed} photos
📊 Total visual embeddings: {len(self.visual_embeddings)}
📊 Total text embeddings: {len(self.text_embeddings)}
        """)
    
    def save_embeddings(self):
        """Save embeddings to disk"""
        embeddings_dir = Path("main_app_outputs/embeddings")
        embeddings_dir.mkdir(exist_ok=True)
        
        print("💾 Saving embeddings to disk...")
        
        # Save visual embeddings
        visual_path = embeddings_dir / "visual_embeddings.pkl"
        with open(visual_path, 'wb') as f:
            pickle.dump(self.visual_embeddings, f)
        print(f"✅ Saved visual embeddings: {visual_path}")
        
        # Save text embeddings
        text_path = embeddings_dir / "text_embeddings.pkl"
        with open(text_path, 'wb') as f:
            pickle.dump(self.text_embeddings, f)
        print(f"✅ Saved text embeddings: {text_path}")
        
        # Save metadata
        metadata_path = embeddings_dir / "photo_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.photo_metadata, f)
        print(f"✅ Saved metadata: {metadata_path}")
        
        # Save embedding info as JSON
        info = {
            'total_photos': len(self.visual_embeddings),
            'embedding_dimension': 512,
            'model_used': 'ViT-B/32',
            'device_used': self.device,
            'photos': list(self.visual_embeddings.keys())
        }
        
        info_path = embeddings_dir / "embeddings_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved embedding info: {info_path}")


def main():
    print("🎯 CLIP Embedding Generation for Photo Similarity Search")
    print("=" * 60)
    
    generator = CLIPEmbeddingGenerator()
    
    # Process all photos
    generator.process_all_photos()
    
    # Save embeddings
    generator.save_embeddings()
    
    print("""
🎉 SUCCESS! CLIP embeddings generated successfully!
══════════════════════════════════════════════════

📁 Embeddings saved to: main_app_outputs/embeddings/
🖼️  visual_embeddings.pkl - Visual similarity search
📝 text_embeddings.pkl - Text similarity search  
📊 photo_metadata.pkl - Photo information
ℹ️  embeddings_info.json - Summary information

🔍 Ready for similarity search integration!
    """)


if __name__ == "__main__":
    main()