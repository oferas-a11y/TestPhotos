#!/usr/bin/env python3
"""
Gemma model for reranking semantic search results.
Uses google-genai to improve search result relevance.
"""

import os
import json
import random
from typing import List, Dict, Any
from pathlib import Path
from google import genai
from google.genai import types

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file in project root"""
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()


class GemmaReranker:
    """Reranks search results using Gemma model for improved relevance."""
    
    def __init__(self):
        self.model = "gemini-2.5-flash-lite"
        self._check_api_key()
    
    def _check_api_key(self) -> None:
        """Check if GEMINI_API_KEY is available"""
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("❌ [GEMINI DEBUG] GEMINI_API_KEY not found in environment variables")
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        print(f"✅ [GEMINI DEBUG] API key found: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else '****'}")
        print(f"🔍 [GEMINI DEBUG] Using model: {self.model}")
    
    def rerank_results(self, query: str, results: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Rerank search results using Gemma model.
        
        Args:
            query: Original search query
            results: List of search results from MiniLM
            top_k: Number of top results to return
            
        Returns:
            List of reranked results
        """
        if not results:
            return []
        
        print(f"🔍 DEBUG: Starting rerank with {len(results)} results for query: '{query}'")
        
        # Show original order
        print("📋 DEBUG: Original MiniLM order:")
        for i, result in enumerate(results[:5], 1):  # Show first 5
            filename = result.get('original_path', '').split('/')[-1] if result.get('original_path') else f'Photo {i}'
            print(f"  {i}. {filename}")
        
        # Remove any similarity scores to prevent bias
        clean_results = []
        for result in results:
            clean_result = result.copy()
            # Remove any keys that might indicate ranking/scoring
            keys_to_remove = ['similarity', 'score', 'rank', 'distance', 'confidence']
            removed_keys = []
            for key in keys_to_remove:
                if key in clean_result:
                    removed_keys.append(key)
                    clean_result.pop(key, None)
            clean_results.append(clean_result)
        
        print(f"🧹 DEBUG: Removed scoring keys: {removed_keys}")
        
        # Prepare context for Gemma with input descriptions
        context = self._prepare_context(query, clean_results)
        
        try:
            print("\n--- GEMINI RERANKING DEBUG START ---")
            print(f"🔍 [GEMINI DEBUG] Query: '{query}'")
            print(f"🔍 [GEMINI DEBUG] Number of results to rerank: {len(clean_results)}")
            print(f"🔍 [GEMINI DEBUG] Top K requested: {top_k}")
            print(f"🔍 [GEMINI DEBUG] Context length: {len(context)} characters")
            print("🤖 [GEMINI DEBUG] Sending request to Gemini API...")
            
            # Get ranking from Gemma
            ranking_response = self._query_gemma(context)
            
            print(f"✅ [GEMINI DEBUG] Response received!")
            print(f"📝 [GEMINI DEBUG] Response length: {len(ranking_response)} chars")
            print(f"📝 [GEMINI DEBUG] Full Gemini response:")
            print("=" * 60)
            print(ranking_response)
            print("=" * 60)
            
            # Parse ranking and reorder results (note: we use original results for output)
            reranked_results = self._parse_and_reorder(ranking_response, clean_results, top_k)
            
            print("🏆 [GEMINI DEBUG] Final reranked order:")
            for i, result in enumerate(reranked_results[:5], 1):  # Show first 5
                filename = result.get('original_path', '').split('/')[-1] if result.get('original_path') else f'Photo {i}'
                gemma_rank = result.get('gemma_rank', 'N/A')
                print(f"  {i}. {filename} (Gemma rank: {gemma_rank})")
            
            print("--- GEMINI RERANKING DEBUG END ---\n")
            return reranked_results
            
        except Exception as e:
            print(f"❌ [GEMINI DEBUG] Gemma reranking failed: {e}")
            print(f"❌ [GEMINI DEBUG] Exception type: {type(e).__name__}")
            import traceback
            print(f"❌ [GEMINI DEBUG] Full traceback:")
            traceback.print_exc()
            print("📋 [GEMINI DEBUG] Falling back to original MiniLM ranking")
            print("--- GEMINI RERANKING DEBUG END ---\n")
            return results[:top_k]
    
    def _prepare_context(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Prepare enhanced context for Gemma model with chain of thought reasoning"""
        context_parts = [
            "You are a distinguished historian specializing in 20th century European history, particularly the Holocaust and WWII period. Your expertise includes analyzing historical photographs, documents, and testimonies.",
            "",
            "TASK: Rerank historical photograph search results to best match the user's research query.",
            "",
            "⚠️  IMPORTANT: The items below are presented without similarity scores.",
            "Analyze each item independently and create your own ranking based on TEXTUAL relevance.",
            "DO NOT assume any existing order.",
            "",
            f"USER RESEARCH QUERY: \"{query}\"",
            "",
            "STEP 1 - EXTRACT MAIN SUBJECT & FOCUS (TEXTUAL):",
            "Identify the PRIMARY concept the user seeks in text:",
            f"From the query \"{query}\", what is the MAIN SUBJECT or CONCEPT?",
            "Consider entities, actions, settings, symbols, and relationships explicitly described.",
            "",
            "STEP 2 - TEXTUAL CONTENT ANALYSIS:",
            "For each item, examine if the DESCRIPTION, CAPTION, and ITEMS clearly match the main subject:",
            "- Is the main concept explicitly present in the description and/or caption?",
            "- How specific and direct is the match?",
            "- Does the context (time/place/actors) align with the query?",
            "- Rate textual presence: STRONG (clear), MODERATE (somewhat), WEAK (barely), NONE (absent)",
            "",
            f"ITEMS TO ANALYZE ({len(results)} total):",
            ""
        ]
        
        # Add each result with comprehensive details
        for i, result in enumerate(results, 1):
            desc = result.get('description', '') or result.get('comprehensive_text', '')
            filename = result.get('original_path', '').split('/')[-1] if result.get('original_path') else f'Photo {i}'
            
            context_parts.append(f"ITEM [{i}]: {filename}")
            context_parts.append(f"Description: {desc[:800]}")  # Use the embedded text description
            caption = result.get('caption', '') or ''
            if caption:
                context_parts.append(f"Caption: {caption[:400]}")
            items = result.get('items', '') or ''
            if items:
                context_parts.append(f"Items: {items[:400]}")
            context_parts.append("---")
        
        context_parts.extend([
            "",
            "STEP 3 - FINAL RANKING (TEXTUAL):",
            "Rank items by how well their DESCRIPTION, CAPTION, and ITEMS match the query.",
            "Prefer clear, specific textual matches over vague references.",
            "",
            "OUTPUT FORMAT:",
            "Provide ONLY a JSON array with indices in order of textual relevance:",
            f"[most_relevant_index, second_most_relevant, ..., least_relevant]",
            f"Must include all {len(results)} indices from 1 to {len(results)}.",
            "",
            "INDEPENDENT RANKING:"
        ])
        
        return "\n".join(context_parts)
    
    def _query_gemma(self, context: str) -> str:
        """Query Gemma model with context"""
        print(f"🔧 [GEMINI DEBUG] Creating Gemini client...")
        api_key = os.environ.get("GEMINI_API_KEY")
        
        try:
            client = genai.Client(api_key=api_key)
            print(f"✅ [GEMINI DEBUG] Client created successfully")
        except Exception as e:
            print(f"❌ [GEMINI DEBUG] Failed to create client: {e}")
            raise
        
        print(f"🔧 [GEMINI DEBUG] Preparing request content...")
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=context),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            temperature=0.1,  # Very low temperature for precise historical analysis
            candidate_count=1,  # Single best response
        )
        
        print(f"🔧 [GEMINI DEBUG] Config: temperature=0.1, candidate_count=1")
        print(f"📤 [GEMINI DEBUG] Sending request to model: {self.model}")
        
        try:
            response_text = ""
            chunk_count = 0
            for chunk in client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=generate_content_config,
            ):
                chunk_count += 1
                response_text += chunk.text
                if chunk_count <= 3:  # Log first few chunks
                    print(f"📦 [GEMINI DEBUG] Chunk {chunk_count}: {len(chunk.text)} chars")
            
            print(f"✅ [GEMINI DEBUG] Received {chunk_count} chunks, total length: {len(response_text)} chars")
            return response_text.strip()
            
        except Exception as e:
            print(f"❌ [GEMINI DEBUG] API call failed: {e}")
            print(f"❌ [GEMINI DEBUG] Error type: {type(e).__name__}")
            raise
    
    def _parse_and_reorder(self, ranking_response: str, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Parse Gemma ranking response and reorder results"""
        try:
            print(f"🔍 DEBUG: Parsing ranking response...")
            print(f"📝 DEBUG: Full Gemini response:\n{ranking_response}")
            
            # Extract JSON from response
            ranking_json = self._extract_json(ranking_response)
            
            print(f"📊 DEBUG: Extracted ranking JSON: {ranking_json}")
            
            if not ranking_json:
                raise ValueError("No valid JSON ranking found")
            
            print(f"🔢 DEBUG: Expected {len(results)} indices, got {len(ranking_json)}")
            
            # Validate ranking
            if not isinstance(ranking_json, list):
                raise ValueError(f"Invalid ranking format: expected list, got {type(ranking_json)}")
            
            if len(ranking_json) != len(results):
                print(f"⚠️  DEBUG: Length mismatch - adjusting to available indices")
                # Use what we have, but ensure we don't exceed available results
                ranking_json = ranking_json[:len(results)]
            
            print(f"🎯 DEBUG: Using ranking: {ranking_json}")
            
            # Reorder results based on ranking, with validation and de-duplication
            seen = set()
            reranked = []
            for rank_idx in ranking_json:
                if not isinstance(rank_idx, int):
                    continue
                if 1 <= rank_idx <= len(results) and rank_idx not in seen:
                    seen.add(rank_idx)
                    result_copy = results[rank_idx - 1].copy()  # Convert to 0-based index
                    result_copy['gemma_rank'] = len(reranked) + 1
                    reranked.append(result_copy)
                if len(reranked) >= top_k:
                    break

            # If fewer than requested, append remaining items in original order
            if len(reranked) < top_k:
                for i in range(1, len(results) + 1):
                    if i not in seen:
                        result_copy = results[i - 1].copy()
                        result_copy['gemma_rank'] = len(reranked) + 1
                        reranked.append(result_copy)
                        if len(reranked) >= top_k:
                            break

            print(f"📦 DEBUG: Reranked {len(reranked)} results")
            return reranked
            
        except Exception as e:
            print(f"⚠️  Error parsing Gemma ranking: {e}")
            print(f"📝 DEBUG: Raw response that failed: {ranking_response[:500]}...")
            # Fallback to original order
            return results[:top_k]
    
    def _extract_json(self, text: str) -> List[int]:
        """Extract JSON list from text response"""
        import re
        
        print(f"🔍 DEBUG: Extracting JSON from text length {len(text)}")
        
        # Look for JSON arrays specifically, prioritizing longer arrays
        json_arrays = []
        
        # Find all potential JSON arrays in the text
        for i in range(len(text)):
            if text[i] == '[':
                # Find matching closing bracket
                bracket_count = 0
                for j in range(i, len(text)):
                    if text[j] == '[':
                        bracket_count += 1
                    elif text[j] == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            try:
                                json_str = text[i:j+1]
                                parsed = json.loads(json_str)
                                if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
                                    json_arrays.append((len(parsed), parsed))
                                    print(f"🔍 DEBUG: Found valid JSON array with {len(parsed)} elements: {json_str}")
                                    break
                            except json.JSONDecodeError:
                                continue
                            break
        
        # Return the longest valid array (most likely to be the ranking)
        if json_arrays:
            json_arrays.sort(reverse=True)  # Sort by length, longest first
            result = json_arrays[0][1]
            print(f"✅ DEBUG: Using longest JSON array with {len(result)} elements: {result}")
            return result
        
        # If no JSON found, try to extract numbers
        print("🔍 DEBUG: No valid JSON found, extracting numbers...")
        numbers = re.findall(r'\b\d+\b', text)
        if numbers:
            result = [int(n) for n in numbers]
            print(f"🔢 DEBUG: Extracted numbers: {result}")
            return result
        
        print("❌ DEBUG: No JSON or numbers found")
        return []


def test_reranker():
    """Test the Gemma reranker"""
    reranker = GemmaReranker()
    
    # Sample test results
    test_results = [
        {
            'original_path': 'photo1.jpg',
            'description': 'A group of Jewish children in the Warsaw Ghetto, wearing yellow stars.',
        },
        {
            'original_path': 'photo2.jpg', 
            'description': 'Nazi soldiers marching through a German town in 1940.',
        },
        {
            'original_path': 'photo3.jpg',
            'description': 'Jewish families being deported from a train station.',
        }
    ]
    
    query = "Jewish families in Warsaw Ghetto"
    
    print("🧪 Testing Gemma reranker...")
    print(f"Query: {query}")
    print(f"Original results: {len(test_results)}")
    
    reranked = reranker.rerank_results(query, test_results, top_k=3)
    
    print("\n📊 Reranked results:")
    for i, result in enumerate(reranked, 1):
        print(f"{i}. {result.get('original_path')} (Gemma rank: {result.get('gemma_rank', 'N/A')})")
        print(f"   {result.get('description', '')[:100]}...")


if __name__ == "__main__":
    test_reranker()
