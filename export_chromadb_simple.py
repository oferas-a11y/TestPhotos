#!/usr/bin/env python3
"""
Simple ChromaDB data export using peek method to avoid array issues
"""
import sys
import json
from pathlib import Path

# Add project paths
project_root = str(Path(__file__).parent)
main_app_path = str(Path(__file__).parent / "main_app")
sys.path.insert(0, project_root)
sys.path.insert(1, main_app_path)

from modules.chroma_handler import create_chroma_handler

def export_chromadb_simple():
    """Export ChromaDB data using a simple approach."""
    print("🔄 Exporting ChromaDB data (simple method)...")
    
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
    
    try:
        # Use peek to get some sample data first
        sample = handler.photos_collection.peek(limit=5)
        print(f"📝 Sample data structure:")
        if sample.get('ids'):
            print(f"   IDs: {len(sample['ids'])} items")
            if sample.get('embeddings') and len(sample['embeddings']) > 0:
                print(f"   Vector dimension: {len(sample['embeddings'][0])}")
            if sample.get('metadatas'):
                print(f"   Metadata keys: {list(sample['metadatas'][0].keys())}")
        
        # Now try to get all data using query instead
        print("🔍 Querying all data...")
        
        # Get all IDs first
        all_data = handler.photos_collection.get()
        
        if all_data and all_data.get('ids'):
            ids = all_data['ids']
            print(f"✅ Retrieved {len(ids)} photo IDs")
            
            # Create export file with basic info
            export_file = Path("main_app_outputs/chromadb_export_simple.json")
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            export_info = {
                'total_photos': len(ids),
                'sample_ids': ids[:10],
                'collection_name': 'historical_photos',
                'export_method': 'simple_ids_only'
            }
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(export_info, f, indent=2)
            
            print(f"📁 Exported basic info to: {export_file}")
            return True
        else:
            print("❌ No IDs found")
            return False
        
    except Exception as e:
        print(f"❌ Export error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = export_chromadb_simple()
    if not success:
        sys.exit(1)