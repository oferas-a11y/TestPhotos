"""
Pinecone vector database handler for photo analysis data.
Replaces ChromaDB with cloud-hosted Pinecone for persistence across deployments.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import time

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("⚠️  Pinecone not available. Install with: pip install pinecone-client")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PineconePhotoHandler:
    """Handles photo analysis data storage and retrieval using Pinecone."""
    
    def __init__(self, api_key: Optional[str] = None, index_name: str = "historical-photos"):
        """
        Initialize Pinecone handler.
        
        Args:
            api_key: Pinecone API key (if None, reads from PINECONE_API_KEY env var)
            index_name: Name of the Pinecone index to use
        """
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone client not available. Install with: pip install pinecone-client")
        
        # Get API key
        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        if not self.api_key:
            raise ValueError("Pinecone API key not found. Set PINECONE_API_KEY environment variable or pass api_key parameter")
        
        self.index_name = index_name
        self.dimension = None  # Will be set based on actual data
        
        # Initialize Pinecone client
        try:
            self.client = Pinecone(api_key=self.api_key)
            print(f"✅ Pinecone client initialized")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Pinecone client: {e}")
        
        # Initialize or get index (dimension will be set during first data storage)
        self.index = None
        
        print(f"✅ Pinecone handler ready: {self.index_name}")
    
    def _initialize_index(self, dimension: int):
        """Initialize or get existing Pinecone index with the correct dimension."""
        try:
            # Check if index already exists
            existing_indexes = [idx.name for idx in self.client.list_indexes()]
            
            if self.index_name in existing_indexes:
                print(f"📍 Using existing Pinecone index: {self.index_name}")
                index = self.client.Index(self.index_name)
                
                # Check if dimensions match
                index_stats = index.describe_index_stats()
                existing_dimension = index_stats.dimension
                if existing_dimension != dimension:
                    print(f"⚠️  Dimension mismatch! Index has {existing_dimension}D, data has {dimension}D")
                    print(f"🗑️  Deleting existing index to recreate with correct dimension...")
                    self.client.delete_index(self.index_name)
                    time.sleep(5)  # Wait for deletion
                else:
                    return index
            
            # Create new index with correct dimension
            print(f"🔨 Creating new Pinecone index: {self.index_name} ({dimension}D)")
            self.client.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            
            # Wait for index to be ready
            print("⏳ Waiting for index to be ready...")
            while not self.client.describe_index(self.index_name).status['ready']:
                time.sleep(1)
            
            print(f"✅ Index created successfully: {self.index_name} ({dimension}D)")
            return self.client.Index(self.index_name)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Pinecone index: {e}")
    
    def store_photo_analysis(self, record: Dict[str, Any]) -> str:
        """
        Store a single photo analysis record in Pinecone.
        
        Args:
            record: Photo analysis record containing vectors and metadata
            
        Returns:
            str: ID of the stored record
        """
        # Extract required fields
        photo_id = record.get('id') or record.get('photo_id') or f"photo_{int(time.time())}"
        embedding = record.get('embedding') or record.get('vector')
        metadata = record.get('metadata', {})
        
        if embedding is None:
            raise ValueError("No embedding/vector found in record")
        
        # Ensure embedding is a list
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        
        # Initialize index with correct dimension if not done yet
        if self.index is None:
            self.dimension = len(embedding)
            print(f"📐 Detected vector dimension: {self.dimension}")
            self.index = self._initialize_index(self.dimension)
        elif len(embedding) != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {len(embedding)}")
        
        # Prepare metadata for Pinecone (string values only)
        pinecone_metadata = {}
        for key, value in metadata.items():
            if value is not None:
                if isinstance(value, (str, int, float, bool)):
                    pinecone_metadata[key] = str(value)
                else:
                    pinecone_metadata[key] = str(value)
        
        # Add document text if available
        if 'document' in record and record['document']:
            pinecone_metadata['document'] = str(record['document'])
        
        try:
            # Upsert to Pinecone
            self.index.upsert(vectors=[{
                'id': photo_id,
                'values': embedding,
                'metadata': pinecone_metadata
            }])
            
            return photo_id
            
        except Exception as e:
            logger.error(f"Failed to store photo {photo_id}: {e}")
            raise
    
    def store_batch_analysis(self, records: List[Dict[str, Any]], batch_size: int = 100) -> List[str]:
        """
        Store multiple photo analysis records in batches.
        
        Args:
            records: List of photo analysis records
            batch_size: Number of records to process in each batch
            
        Returns:
            List[str]: List of stored record IDs
        """
        stored_ids = []
        
        # Initialize index with correct dimension from first record
        if self.index is None and records:
            first_embedding = records[0].get('embedding') or records[0].get('vector')
            if first_embedding is not None:
                if hasattr(first_embedding, 'tolist'):
                    first_embedding = first_embedding.tolist()
                self.dimension = len(first_embedding)
                print(f"📐 Detected vector dimension: {self.dimension}")
                self.index = self._initialize_index(self.dimension)
        
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            batch_vectors = []
            
            for record in batch:
                photo_id = record.get('id') or record.get('photo_id') or f"photo_{int(time.time())}_{i}"
                embedding = record.get('embedding') or record.get('vector')
                metadata = record.get('metadata', {})
                
                if embedding is None:
                    logger.warning(f"Skipping record {photo_id}: No embedding found")
                    continue
                
                # Ensure embedding is a list
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                
                # Prepare metadata
                pinecone_metadata = {}
                for key, value in metadata.items():
                    if value is not None:
                        pinecone_metadata[key] = str(value)
                
                if 'document' in record and record['document']:
                    pinecone_metadata['document'] = str(record['document'])
                
                batch_vectors.append({
                    'id': photo_id,
                    'values': embedding,
                    'metadata': pinecone_metadata
                })
                stored_ids.append(photo_id)
            
            if batch_vectors:
                try:
                    self.index.upsert(vectors=batch_vectors)
                    print(f"✅ Stored batch {i//batch_size + 1}: {len(batch_vectors)} photos")
                except Exception as e:
                    logger.error(f"Failed to store batch {i//batch_size + 1}: {e}")
                    raise
        
        return stored_ids
    
    def search_photos(self, query_vector: List[float], top_k: int = 10, 
                     filter_dict: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar photos using vector similarity.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of matching photos with metadata
        """
        if self.index is None:
            print("❌ Pinecone index not initialized. Please store some data first.")
            return []
        
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True
            )
            
            photos = []
            for match in results.matches:
                photo_data = {
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                }
                photos.append(photo_data)
            
            return photos
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index."""
        if self.index is None:
            return {'total_photos': 0, 'index_name': self.index_name, 'error': 'Index not initialized'}
        
        try:
            index_stats = self.index.describe_index_stats()
            
            return {
                'total_photos': index_stats.total_vector_count,
                'index_name': self.index_name,
                'dimension': index_stats.dimension,
                'index_fullness': index_stats.index_fullness,
                'namespaces': index_stats.namespaces
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'total_photos': 0, 'index_name': self.index_name, 'error': str(e)}
    
    def delete_photo(self, photo_id: str) -> bool:
        """Delete a photo record by ID."""
        if self.index is None:
            return False
        try:
            self.index.delete(ids=[photo_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete photo {photo_id}: {e}")
            return False
    
    def clear_all_data(self) -> bool:
        """Clear all data from the index."""
        if self.index is None:
            return False
        try:
            self.index.delete(delete_all=True)
            print("🗑️  All data cleared from Pinecone index")
            return True
        except Exception as e:
            logger.error(f"Failed to clear data: {e}")
            return False


def create_pinecone_handler(api_key: Optional[str] = None, 
                          index_name: str = "historical-photos") -> Optional[PineconePhotoHandler]:
    """
    Factory function to create a PineconePhotoHandler instance.
    
    Args:
        api_key: Pinecone API key (if None, reads from env)
        index_name: Name of the Pinecone index
        
    Returns:
        PineconePhotoHandler instance or None if creation fails
    """
    try:
        return PineconePhotoHandler(api_key=api_key, index_name=index_name)
    except Exception as e:
        logger.error(f"Failed to create Pinecone handler: {e}")
        return None


if __name__ == "__main__":
    # Test the handler
    handler = create_pinecone_handler()
    if handler:
        stats = handler.get_collection_stats()
        print(f"Pinecone stats: {stats}")