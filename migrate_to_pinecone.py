#!/usr/bin/env python3
"""
Migrate all photo analysis data from ChromaDB to Pinecone.
This preserves all vectors, metadata, and enables cloud access.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any
import time

# Add project paths
project_root = str(Path(__file__).parent)
main_app_path = str(Path(__file__).parent / "main_app")
sys.path.insert(0, project_root)
sys.path.insert(1, main_app_path)

# Import handlers
try:
    from modules.chroma_handler import create_chroma_handler
    from modules.pinecone_handler import create_pinecone_handler
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure both ChromaDB and Pinecone handlers are available")
    sys.exit(1)


def migrate_chromadb_to_pinecone():
    """Migrate all data from ChromaDB to Pinecone."""
    print("🔄 Starting ChromaDB → Pinecone migration...")
    print("=" * 60)
    
    # Check for Pinecone API key
    if not os.getenv('PINECONE_API_KEY'):
        print("❌ PINECONE_API_KEY environment variable not set!")
        print("Please add your Pinecone API key to the .env file:")
        print("   PINECONE_API_KEY=your_api_key_here")
        return False
    
    # Initialize ChromaDB handler
    print("📖 Connecting to ChromaDB...")
    chroma_handler = create_chroma_handler()
    if not chroma_handler:
        print("❌ Failed to connect to ChromaDB")
        return False
    
    # Get ChromaDB stats
    chroma_stats = chroma_handler.get_collection_stats()
    total_photos = chroma_stats.get('total_photos', 0)
    print(f"📊 Found {total_photos} photos in ChromaDB")
    
    if total_photos == 0:
        print("⚠️  No photos to migrate")
        return True
    
    # Initialize Pinecone handler
    print("☁️  Connecting to Pinecone...")
    pinecone_handler = create_pinecone_handler()
    if not pinecone_handler:
        print("❌ Failed to connect to Pinecone")
        return False
    
    # Get Pinecone stats
    pinecone_stats = pinecone_handler.get_collection_stats()
    existing_photos = pinecone_stats.get('total_photos', 0)
    print(f"📊 Pinecone currently has {existing_photos} photos")
    
    if existing_photos > 0:
        print("⚠️  Pinecone index already contains data")
        choice = input("Continue and add/update photos? (y/N): ").strip().lower()
        if choice not in ['y', 'yes']:
            print("👋 Migration cancelled")
            return False
    
    # Start migration
    print(f"\\n🚀 Starting migration of {total_photos} photos...")
    print("   This may take several minutes...")
    
    try:
        # Get all data from ChromaDB in batches to avoid memory issues
        batch_size = 100
        migrated_count = 0
        failed_count = 0
        
        # Process in batches
        for batch_start in range(0, total_photos, batch_size):
            batch_end = min(batch_start + batch_size, total_photos)
            print(f"\\n📦 Processing batch {batch_start//batch_size + 1}: photos {batch_start+1}-{batch_end}")
            
            # Get batch from ChromaDB - using limit and offset doesn't work well with ChromaDB
            # Instead, get all data and slice it
            if batch_start == 0:  # First batch - get all data
                print("   Fetching all data from ChromaDB...")
                all_results = chroma_handler.photos_collection.get(
                    include=['embeddings', 'metadatas', 'documents']
                )
                
                if not all_results or not all_results.get('ids'):
                    print("❌ No data retrieved from ChromaDB")
                    return False
                
                all_ids = all_results['ids']
                all_embeddings = all_results.get('embeddings', [])
                all_metadatas = all_results.get('metadatas', [])
                all_documents = all_results.get('documents', [])
                
                print(f"   Retrieved {len(all_ids)} records from ChromaDB")
            
            # Process current batch
            batch_records = []
            for i in range(batch_start, batch_end):
                if i >= len(all_ids):
                    break
                
                # Safely extract data
                photo_id = all_ids[i]
                embedding = None
                metadata = {}
                document = None
                
                # Handle embeddings (numpy arrays)
                if i < len(all_embeddings) and all_embeddings[i] is not None:
                    emb = all_embeddings[i]
                    if hasattr(emb, 'tolist'):
                        embedding = emb.tolist()
                    else:
                        embedding = emb
                
                # Handle metadata
                if i < len(all_metadatas) and all_metadatas[i] is not None:
                    metadata = all_metadatas[i]
                
                # Handle document
                if i < len(all_documents) and all_documents[i] is not None:
                    document = all_documents[i]
                
                if embedding is not None:
                    batch_records.append({
                        'id': photo_id,
                        'embedding': embedding,
                        'metadata': metadata,
                        'document': document
                    })
                else:
                    print(f"   ⚠️  Skipping {photo_id}: No embedding")
                    failed_count += 1
            
            # Store batch in Pinecone
            if batch_records:
                try:
                    stored_ids = pinecone_handler.store_batch_analysis(batch_records)
                    migrated_count += len(stored_ids)
                    print(f"   ✅ Migrated {len(stored_ids)} photos")
                except Exception as e:
                    print(f"   ❌ Batch migration failed: {e}")
                    failed_count += len(batch_records)
            
            # Progress update
            progress = (migrated_count + failed_count) / total_photos * 100
            print(f"   📈 Progress: {progress:.1f}% ({migrated_count} migrated, {failed_count} failed)")
        
        # Final statistics
        print(f"\\n✅ Migration completed!")
        print("=" * 60)
        print(f"📊 Final Statistics:")
        print(f"   📥 ChromaDB photos: {total_photos}")
        print(f"   📤 Successfully migrated: {migrated_count}")
        print(f"   ❌ Failed migrations: {failed_count}")
        
        # Verify in Pinecone
        final_stats = pinecone_handler.get_collection_stats()
        final_count = final_stats.get('total_photos', 0)
        print(f"   ☁️  Pinecone photos: {final_count}")
        
        if migrated_count > 0:
            print(f"\\n🎉 Success! Your photos are now available in Pinecone cloud database.")
            print(f"🔗 You can now access your data from any terminal with the same API key.")
            print(f"📱 Ready for web deployment on Render or other cloud platforms!")
            
            # Show sample search
            print(f"\\n🔍 Testing search functionality...")
            try:
                # Get a sample vector for testing
                sample_results = chroma_handler.photos_collection.get(limit=1, include=['embeddings'])
                if sample_results and sample_results.get('embeddings'):
                    sample_vector = sample_results['embeddings'][0]
                    if hasattr(sample_vector, 'tolist'):
                        sample_vector = sample_vector.tolist()
                    
                    search_results = pinecone_handler.search_photos(sample_vector, top_k=3)
                    print(f"   ✅ Search test successful: Found {len(search_results)} similar photos")
                else:
                    print("   ⚠️  Could not test search: No sample vector available")
            except Exception as e:
                print(f"   ⚠️  Search test failed: {e}")
        
        return migrated_count > 0
        
    except Exception as e:
        print(f"\\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main migration function."""
    print("🌟 ChromaDB to Pinecone Migration Tool")
    print("Migrate your photo analysis data to the cloud!")
    print()
    
    # Check if .env exists
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found. Creating from template...")
        template_file = Path(".env.template")
        if template_file.exists():
            with open(template_file, 'r') as f:
                template_content = f.read()
            
            with open(env_file, 'w') as f:
                f.write(template_content)
                f.write("\\n# Pinecone Configuration\\n")
                f.write("PINECONE_API_KEY=your_pinecone_api_key_here\\n")
            
            print(f"✅ Created .env file from template")
            print(f"🔑 Please add your Pinecone API key to .env file:")
            print(f"   PINECONE_API_KEY=your_api_key_here")
            print(f"\\n📖 Get your free Pinecone API key at: https://www.pinecone.io/")
            return
    
    success = migrate_chromadb_to_pinecone()
    if success:
        print(f"\\n🎯 Next Steps:")
        print(f"   1. Test your search functionality: python run.py dashboard")
        print(f"   2. Your data is now ready for web deployment!")
        print(f"   3. You can access the same data from multiple terminals/servers")
    else:
        print(f"\\n💡 Troubleshooting:")
        print(f"   1. Make sure PINECONE_API_KEY is set in .env file")
        print(f"   2. Check your internet connection")
        print(f"   3. Verify ChromaDB data exists: python run.py status")


if __name__ == "__main__":
    main()