#!/usr/bin/env python3
"""
Export all ChromaDB data for migration to Pinecone
"""
import sys
import json
from pathlib import Path
import numpy as np
from typing import Dict, List, Any

# Add project paths
project_root = str(Path(__file__).parent)
main_app_path = str(Path(__file__).parent / "main_app")
sys.path.insert(0, project_root)
sys.path.insert(1, main_app_path)

from modules.chroma_handler import create_chroma_handler

def export_chromadb_data():
    """Export all ChromaDB vectors and metadata to JSON file."""
    print("🔄 Exporting ChromaDB data...")
    
    # Initialize ChromaDB handler
    handler = create_chroma_handler()
    if not handler:
        print("❌ Failed to initialize ChromaDB handler")
        return False
    
    # Get collection stats
    stats = handler.get_collection_stats()
    total_photos = stats.get('total_photos', 0)
    print(f"📊 Found {total_photos} photos in ChromaDB")
    
    if total_photos == 0:
        print("⚠️  No photos to export")
        return False
    
    # Export all data in batches
    export_data = {
        'metadata': {
            'total_photos': total_photos,
            'collection_name': 'historical_photos',
            'vector_dimension': None,
            'export_timestamp': str(pd.Timestamp.now()),
        },
        'photos': []
    }
    
    batch_size = 100
    exported_count = 0
    
    try:
        # Get all data from ChromaDB
        results = handler.photos_collection.get(
            include=['embeddings', 'metadatas', 'documents']
        )
        
        if not results or not results.get('ids') or len(results.get('ids', [])) == 0:
            print("❌ No data found in ChromaDB")
            return False
        
        # Check vector dimension from first embedding
        embeddings_list = results.get('embeddings', [])
        if embeddings_list and len(embeddings_list) > 0:
            first_embedding = embeddings_list[0]
            if first_embedding is not None:
                vector_dim = len(first_embedding)
                export_data['metadata']['vector_dimension'] = vector_dim
                print(f"📐 Vector dimension: {vector_dim}")
            else:
                print("⚠️  First embedding is None")
        
        # Process each photo record
        ids = results.get('ids', [])
        embeddings = results.get('embeddings', [])
        metadatas = results.get('metadatas', [])
        documents = results.get('documents', [])
        
        for i, photo_id in enumerate(ids):
            # Safely get embedding, metadata, and document
            embedding = embeddings[i] if i < len(embeddings) and embeddings[i] is not None else None
            metadata = metadatas[i] if i < len(metadatas) and metadatas[i] is not None else {}
            document = documents[i] if i < len(documents) and documents[i] is not None else None
            
            photo_record = {
                'id': photo_id,
                'embedding': embedding,
                'metadata': metadata,
                'document': document
            }
            
            export_data['photos'].append(photo_record)
            exported_count += 1
            
            if exported_count % 100 == 0:
                print(f"   Exported {exported_count}/{total_photos} photos...")
        
        # Save to JSON file
        export_file = Path("main_app_outputs/chromadb_export.json")
        export_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(export_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        
        print(f"✅ Export completed!")
        print(f"📁 Exported {exported_count} photos to: {export_file}")
        print(f"📊 File size: {export_file.stat().st_size / 1024 / 1024:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Export error: {e}")
        return False

if __name__ == "__main__":
    import pandas as pd
    success = export_chromadb_data()
    if not success:
        sys.exit(1)