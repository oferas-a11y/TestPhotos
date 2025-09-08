"""
ChromaDB handler for storing and querying historical photo analysis data.

This module provides a clean interface for storing photo analysis results
in ChromaDB and performing both semantic and category-based searches.
"""

import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: ChromaDB not available. Install with: pip install chromadb")

# Load environment variables from .env file
def _load_env():
    """Load environment variables from .env file in project root"""
    env_file = Path(__file__).parent.parent.parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

_load_env()


class ChromaPhotoHandler:
    """
    Handles ChromaDB operations for historical photo analysis data.
    
    This class provides methods to store photo analysis results and perform
    both semantic searches and category-based filtering using ChromaDB's
    vector database capabilities.
    """
    
    def __init__(self, persist_directory: str = "main_app_outputs/chromadb"):
        """
        Initialize ChromaDB handler.
        
        Args:
            persist_directory: Directory to store ChromaDB data
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Please install with: pip install chromadb")
        
        # Check for required ChromaDB access key
        chroma_access_key = os.environ.get("CHROMADB_ACCESS_KEY")
        if not chroma_access_key:
            raise PermissionError(
                "❌ ChromaDB Access Denied: CHROMADB_ACCESS_KEY not found in environment variables.\n"
                "   This database requires authorized access. Please contact the administrator\n"
                "   to obtain the required .env file with proper credentials."
            )
        
        # Validate access key format (should be at least 16 chars for security)
        if len(chroma_access_key) < 16:
            raise PermissionError(
                "❌ ChromaDB Access Denied: Invalid access key format.\n"
                "   Please contact the administrator for a valid access key."
            )
        
        print(f"✅ ChromaDB Access Granted: Using key {chroma_access_key[:8]}...{chroma_access_key[-4:]}")
        
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get the main collection
        self.photos_collection = self.client.get_or_create_collection(
            name="historical_photos",
            metadata={
                "description": "Historical photo analysis data with AI-generated insights",
                "version": "1.0"
            }
        )
        
        print(f"✅ ChromaDB initialized: {self.persist_dir}")
    
    def store_photo_analysis(self, record: Dict[str, Any]) -> str:
        """
        Store a single photo analysis record in ChromaDB.
        
        Args:
            record: Complete photo analysis record from the pipeline
            
        Returns:
            Generated photo ID
        """
        # Generate unique ID based on original path if available
        original_path = record.get("original_filename", "")
        if original_path:
            # Create deterministic ID from path to avoid duplicates
            photo_id = f"photo_{hash(original_path) % (10**10):010d}"
        else:
            photo_id = str(uuid.uuid4())
        
        # Check if photo already exists
        try:
            existing = self.photos_collection.get(ids=[photo_id])
            if existing['ids']:
                print(f"📋 Photo already exists in ChromaDB: {Path(original_path).name}")
                return photo_id
        except Exception:
            pass
        
        # Build comprehensive text for embedding
        comprehensive_text = self._build_comprehensive_text(record)
        
        # Extract metadata for filtering
        metadata = self._extract_metadata(record)
        
        # Store in ChromaDB
        try:
            self.photos_collection.add(
                documents=[comprehensive_text],
                metadatas=[metadata],
                ids=[photo_id]
            )
            print(f"💾 Stored in ChromaDB: {Path(original_path).name if original_path else photo_id}")
        except Exception as e:
            print(f"❌ Failed to store in ChromaDB: {e}")
            raise
        
        return photo_id
    
    def batch_store_photos(self, records: List[Dict[str, Any]]) -> List[str]:
        """
        Store multiple photo records in batch for better performance.
        
        Args:
            records: List of photo analysis records
            
        Returns:
            List of generated photo IDs
        """
        if not records:
            return []
        
        photo_ids = []
        documents = []
        metadatas = []
        
        # Check for existing photos and prepare new ones
        existing_count = 0
        for record in records:
            original_path = record.get("original_filename", "")
            if original_path:
                photo_id = f"photo_{hash(original_path) % (10**10):010d}"
            else:
                photo_id = str(uuid.uuid4())
            
            # Check if already exists
            try:
                existing = self.photos_collection.get(ids=[photo_id])
                if existing['ids']:
                    existing_count += 1
                    continue
            except Exception:
                pass
            
            # Prepare for batch insert
            comprehensive_text = self._build_comprehensive_text(record)
            metadata = self._extract_metadata(record)
            
            photo_ids.append(photo_id)
            documents.append(comprehensive_text)
            metadatas.append(metadata)
        
        # Batch insert new photos
        if photo_ids:
            try:
                self.photos_collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=photo_ids
                )
                print(f"💾 Batch stored {len(photo_ids)} new photos in ChromaDB")
            except Exception as e:
                print(f"❌ Batch store failed: {e}")
                raise
        
        if existing_count > 0:
            print(f"📋 Skipped {existing_count} photos already in ChromaDB")
        
        return photo_ids
    
    def search_photos(self, query: str, n_results: int = 10, where: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Search photos using semantic similarity.
        
        Args:
            query: Search query text
            n_results: Maximum number of results to return
            where: Optional metadata filters
            
        Returns:
            ChromaDB query results
        """
        try:
            results = self.photos_collection.query(
                query_texts=[query],
                n_results=min(n_results, 100),  # Reasonable limit
                where=where,
                include=['documents', 'metadatas', 'distances']
            )
            return results
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return {'documents': [[]], 'metadatas': [[]], 'ids': [[]], 'distances': [[]]}
    
    def search_by_category(self, category: str, value: Any = True, n_results: int = 50) -> Dict[str, Any]:
        """
        Search photos by specific category using metadata filtering.
        
        Args:
            category: Metadata field name to filter by
            value: Value to match
            n_results: Maximum number of results
            
        Returns:
            ChromaDB query results
        """
        where_clause = {category: value}
        
        try:
            # Use a generic query since we're primarily filtering by metadata
            results = self.photos_collection.query(
                query_texts=["historical photograph"],
                n_results=min(n_results, 200),  # Higher limit for category searches
                where=where_clause,
                include=['documents', 'metadatas', 'distances']
            )
            return results
        except Exception as e:
            print(f"❌ Category search failed: {e}")
            return {'documents': [[]], 'metadatas': [[]], 'ids': [[]], 'distances': [[]]}
    
    def get_photo_by_path(self, original_path: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve photo data by original file path.
        
        Args:
            original_path: Original file path of the photo
            
        Returns:
            Photo data or None if not found
        """
        try:
            results = self.photos_collection.query(
                query_texts=[""],
                n_results=1,
                where={"original_path": original_path},
                include=['documents', 'metadatas', 'distances']
            )
            
            if results['documents'] and results['documents'][0]:
                return {
                    'id': results['ids'][0][0],
                    'document': results['documents'][0][0],
                    'metadata': results['metadatas'][0][0],
                    'distance': results['distances'][0][0] if results['distances'] and results['distances'][0] else None
                }
        except Exception as e:
            print(f"❌ Failed to get photo by path: {e}")
        
        return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the photo collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.photos_collection.count()
            return {
                'total_photos': count,
                'collection_name': 'historical_photos',
                'persist_directory': str(self.persist_dir),
                'chromadb_available': CHROMADB_AVAILABLE
            }
        except Exception as e:
            return {
                'total_photos': 0,
                'collection_name': 'historical_photos',
                'persist_directory': str(self.persist_dir),
                'error': str(e),
                'chromadb_available': CHROMADB_AVAILABLE
            }
    
    def delete_collection(self) -> bool:
        """
        Delete the entire photo collection. Use with caution!
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete_collection("historical_photos")
            print("🗑️  ChromaDB collection deleted")
            return True
        except Exception as e:
            print(f"❌ Failed to delete collection: {e}")
            return False
    
    def _build_comprehensive_text(self, record: Dict[str, Any]) -> str:
        """
        Build comprehensive text description for embedding.
        
        This method combines all relevant textual information from the photo
        analysis to create a rich description for semantic search.
        
        Args:
            record: Complete photo analysis record
            
        Returns:
            Comprehensive text description
        """
        text_parts = []
        
        try:
            # 1. LLM CAPTION (highest priority)
            llm_obj = record.get("llm")
            llm_parsed = {}
            
            if isinstance(llm_obj, dict):
                llm_json_str = llm_obj.get("json")
                if isinstance(llm_json_str, str) and llm_json_str.strip():
                    try:
                        llm_parsed = json.loads(llm_json_str)
                        caption = llm_parsed.get('caption', '')
                        if caption:
                            text_parts.append(caption)
                    except json.JSONDecodeError:
                        pass
            
            # 2. Fallback to filename-based description
            if not text_parts:
                orig_name = record.get("original_filename", "")
                if orig_name:
                    filename = Path(orig_name).stem
                    basic_desc = filename.replace('_', ' ').replace('-', ' ')
                    if basic_desc:
                        text_parts.append(f"Historical photograph {basic_desc}")
            
            # 3. TEXT ANALYSIS (Hebrew/German)
            if llm_parsed:
                text_analysis = llm_parsed.get('text_analysis', {})
                
                # Hebrew text
                hebrew_text = text_analysis.get('hebrew_text', {})
                if hebrew_text.get('present', False):
                    hebrew_found = hebrew_text.get('text_found', '')
                    if hebrew_found:
                        text_parts.append(f"Hebrew text: {hebrew_found}")
                    hebrew_translation = hebrew_text.get('translation', '')
                    if hebrew_translation:
                        text_parts.append(f"Hebrew translation: {hebrew_translation}")
                
                # German text  
                german_text = text_analysis.get('german_text', {})
                if german_text.get('present', False):
                    german_found = german_text.get('text_found', '')
                    if german_found:
                        text_parts.append(f"German text: {german_found}")
                    german_translation = german_text.get('translation', '')
                    if german_translation:
                        text_parts.append(f"German translation: {german_translation}")
            
            # 4. CLIP ANALYSIS
            clip_data = record.get('clip', {})
            indoor_outdoor = clip_data.get('indoor_outdoor', '')
            if indoor_outdoor:
                text_parts.append(f"{indoor_outdoor} setting")
            
            # Background information
            background_detections = clip_data.get('background_detections', [])
            for bg in background_detections[:2]:  # Top 2 background categories
                if isinstance(bg, dict):
                    category = bg.get('category', '')
                    if category:
                        text_parts.append(f"Background: {category}")
            
            # People gender analysis
            people_gender = clip_data.get('people_gender', [])
            men_count = sum(1 for p in people_gender if isinstance(p, dict) and 
                           float(p.get('man', 0)) >= float(p.get('woman', 0)))
            women_count = len(people_gender) - men_count
            if men_count > 0:
                text_parts.append(f"{men_count} men")
            if women_count > 0:
                text_parts.append(f"{women_count} women")
            
            # 5. YOLO OBJECT DETECTION
            yolo_data = record.get('yolo', {})
            object_counts = yolo_data.get('object_counts', {})
            for obj_type, count in object_counts.items():
                if count > 0:
                    text_parts.append(f"{count} {obj_type}{'s' if count > 1 else ''} detected")
            
            # 6. LLM DETAILED ANALYSIS
            if llm_parsed:
                # People under 18
                people_under_18 = llm_parsed.get('people_under_18', 0)
                if people_under_18 > 0:
                    text_parts.append(f"{people_under_18} people under 18")
                
                # Symbol analysis
                if llm_parsed.get('has_jewish_symbols', False):
                    text_parts.append("Jewish symbols present")
                    jewish_symbols = llm_parsed.get('jewish_symbols_details', [])
                    for symbol in jewish_symbols[:3]:  # Top 3 symbols
                        if isinstance(symbol, dict):
                            symbol_type = symbol.get('symbol_type', '')
                            if symbol_type:
                                text_parts.append(symbol_type)
                
                if llm_parsed.get('has_nazi_symbols', False):
                    text_parts.append("Nazi symbols present")
                    nazi_symbols = llm_parsed.get('nazi_symbols_details', [])
                    for symbol in nazi_symbols[:3]:  # Top 3 symbols
                        if isinstance(symbol, dict):
                            symbol_type = symbol.get('symbol_type', '')
                            if symbol_type:
                                text_parts.append(symbol_type)
                
                # Objects and artifacts
                objects = llm_parsed.get('main_objects_artifacts_animals', [])
                for obj in objects[:5]:  # Top 5 objects
                    if isinstance(obj, dict):
                        item = obj.get('item', '')
                        description = obj.get('description', '')
                        if item:
                            text_parts.append(item)
                        if description:
                            text_parts.append(description)
                
                # Violence assessment
                violence = llm_parsed.get('violence_assessment', {})
                if violence.get('signs_of_violence', False):
                    explanation = violence.get('explanation', '')
                    if explanation and explanation != "No signs of violence detected":
                        text_parts.append(explanation)
            
            # 7. OCR TEXT
            ocr_data = record.get('ocr', {})
            if ocr_data.get('has_text', False):
                ocr_lines = ocr_data.get('lines', [])
                for line in ocr_lines[:2]:  # First 2 OCR lines
                    if line and line.strip():
                        text_parts.append(f"OCR text: {line.strip()}")
            
            # 8. COLLECTION CONTEXT from folder path
            orig_name = record.get("original_filename", "")
            if orig_name:
                try:
                    path_parts = Path(orig_name).parts
                    if len(path_parts) > 2 and path_parts[0] == "photo_collections":
                        if len(path_parts) > 2:
                            collection_name = path_parts[2]
                            collection_context = f"Collection: {collection_name.replace(' – ', ' - ').replace('_', ' ')}"
                            text_parts.append(collection_context)
                except Exception:
                    pass
            
            # Combine all text parts
            comprehensive_text = ' '.join([part for part in text_parts if part and part.strip()])
            
            # Fallback if no content found
            if not comprehensive_text:
                comprehensive_text = f"Historical photograph from {Path(orig_name).name}" if orig_name else "Historical photograph"
            
            return comprehensive_text
            
        except Exception as e:
            print(f"❌ Error building comprehensive text: {e}")
            return "Historical photograph"
    
    def _extract_metadata(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata for ChromaDB filtering and organization.
        
        Args:
            record: Complete photo analysis record
            
        Returns:
            Dictionary of metadata fields
        """
        metadata = {
            'original_path': record.get('original_filename', ''),
            'colorized_path': record.get('colorized_path', ''),
            'processed_path': record.get('processed_path', ''),
            'source_path': record.get('source_path', '')
        }
        
        try:
            # Extract LLM analysis metadata
            llm_obj = record.get("llm")
            if isinstance(llm_obj, dict):
                llm_json_str = llm_obj.get("json")
                if isinstance(llm_json_str, str) and llm_json_str.strip():
                    try:
                        llm_parsed = json.loads(llm_json_str)
                        
                        # Boolean fields for filtering
                        metadata.update({
                            'has_jewish_symbols': bool(llm_parsed.get('has_jewish_symbols', False)),
                            'has_nazi_symbols': bool(llm_parsed.get('has_nazi_symbols', False)),
                            'signs_of_violence': bool(llm_parsed.get('violence_assessment', {}).get('signs_of_violence', False)),
                            'people_under_18': int(llm_parsed.get('people_under_18', 0)),
                        })
                        
                        # Text presence indicators
                        text_analysis = llm_parsed.get('text_analysis', {})
                        metadata.update({
                            'has_hebrew_text': bool(text_analysis.get('hebrew_text', {}).get('present', False)),
                            'has_german_text': bool(text_analysis.get('german_text', {}).get('present', False)),
                        })
                        
                    except json.JSONDecodeError:
                        pass
            
            # CLIP metadata
            clip_data = record.get('clip', {})
            metadata['indoor_outdoor'] = clip_data.get('indoor_outdoor', 'unknown')
            
            # People counts from CLIP gender analysis
            people_gender = clip_data.get('people_gender', [])
            women_count = sum(1 for g in people_gender if isinstance(g, dict) and 
                             float(g.get('woman', 0.0)) >= float(g.get('man', 0.0)))
            men_count = len(people_gender) - women_count
            metadata.update({
                'total_people': len(people_gender),
                'men_count': men_count,
                'women_count': women_count
            })
            
            # Object detection metadata
            yolo_data = record.get('yolo', {})
            object_counts = yolo_data.get('object_counts', {})
            metadata['total_objects'] = sum(object_counts.values())
            
            # OCR metadata
            ocr_data = record.get('ocr', {})
            metadata['has_ocr_text'] = bool(ocr_data.get('has_text', False))
            
            # Collection information from path
            orig_path = record.get('original_filename', '')
            if orig_path:
                try:
                    path_parts = Path(orig_path).parts
                    if len(path_parts) > 2:
                        metadata['collection_folder'] = path_parts[2] if path_parts[0] == "photo_collections" else path_parts[0]
                        metadata['filename'] = Path(orig_path).name
                        metadata['file_stem'] = Path(orig_path).stem
                except Exception:
                    metadata['filename'] = Path(orig_path).name
                    metadata['file_stem'] = Path(orig_path).stem
            
        except Exception as e:
            print(f"❌ Error extracting metadata: {e}")
        
        # Ensure all values are JSON-serializable
        for key, value in metadata.items():
            if isinstance(value, (bool, int, float, str)) or value is None:
                continue
            else:
                metadata[key] = str(value)
        
        return metadata


# Convenience function for checking ChromaDB availability
def is_chromadb_available() -> bool:
    """Check if ChromaDB is available for use."""
    return CHROMADB_AVAILABLE


# Factory function for safe ChromaDB handler creation
def create_chroma_handler(persist_directory: str = "main_app_outputs/chromadb") -> Optional[ChromaPhotoHandler]:
    """
    Safely create a ChromaDB handler.
    
    Args:
        persist_directory: Directory to store ChromaDB data
        
    Returns:
        ChromaPhotoHandler instance or None if ChromaDB is not available or access is denied
    """
    if not is_chromadb_available():
        print("⚠️  ChromaDB is not installed. Install with: pip install chromadb")
        return None
    
    try:
        return ChromaPhotoHandler(persist_directory)
    except PermissionError as e:
        print(f"\n{e}")
        return None
    except Exception as e:
        print(f"❌ Failed to create ChromaDB handler: {e}")
        return None