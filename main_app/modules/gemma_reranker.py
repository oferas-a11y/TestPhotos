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
        
        # Randomize results order to prevent bias from MiniLM ranking
        shuffled_results = results.copy()
        original_order = [r.get('original_path', '') for r in shuffled_results]
        random.shuffle(shuffled_results)
        shuffled_order = [r.get('original_path', '') for r in shuffled_results]
        
        print("🎲 DEBUG: After shuffling:")
        for i, result in enumerate(shuffled_results[:5], 1):  # Show first 5
            filename = result.get('original_path', '').split('/')[-1] if result.get('original_path') else f'Photo {i}'
            print(f"  {i}. {filename}")
        
        # Verify randomization worked
        if original_order == shuffled_order:
            print("⚠️  DEBUG: WARNING - Shuffle didn't change order!")
        else:
            print("✅ DEBUG: Shuffle successful - order changed")
        
        # Remove any similarity scores to prevent bias
        clean_results = []
        for result in shuffled_results:
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
        
        # Prepare context for Gemma with randomized input
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
            "⚠️  IMPORTANT: The photographs below are presented in RANDOM ORDER with NO similarity scores.",
            "You must analyze each photograph independently and create your own ranking based on visual relevance.",
            "DO NOT assume any existing order - the photos are shuffled and you must rank them from scratch.",
            "",
            f"USER RESEARCH QUERY: \"{query}\"",
            "",
            "STEP 1 - EXTRACT MAIN SUBJECT & FOCUS:",
            "Identify the PRIMARY subject the user wants to see:",
            f"From the query \"{query}\", what is the MAIN SUBJECT or CONCEPT?",
            "- If query mentions 'love', 'families', 'children' → Look for visible emotional connections, family bonds",
            "- If query mentions 'violence', 'persecution', 'suffering' → Look for visible signs of hardship, oppression",
            "- If query mentions 'daily life', 'work', 'activities' → Look for visible everyday activities, occupations",
            "- If query mentions 'symbols', 'signs', 'text' → Look for visible symbols, writing, markings",
            "- If query mentions specific places → Look for visible location markers, architecture, settings",
            "- If query mentions specific people/groups → Look for visible identification of those people",
            "",
            "CORE VISUAL FOCUS: What should be VISIBLY PRESENT in the most relevant photos?",
            "",
            "STEP 2 - VISUAL CONTENT ANALYSIS:",
            "For each photograph, examine if you can SEE the main subject:",
            "- Does this photo VISUALLY show the main concept from Step 1?",
            "- Are the key visual elements actually visible and clear?",
            "- How prominently is the main subject featured?",
            "- Does the image provide visual evidence of what the user is looking for?",
            "- Rate visual presence: STRONG (clearly visible), MODERATE (somewhat visible), WEAK (barely visible), NONE (not visible)",
            "",
            f"HISTORICAL PHOTOGRAPHS TO ANALYZE ({len(results)} total):",
            ""
        ]
        
        # Add each result with comprehensive details
        for i, result in enumerate(results, 1):
            desc = result.get('description', '') or result.get('comprehensive_text', '')
            filename = result.get('original_path', '').split('/')[-1] if result.get('original_path') else f'Photo {i}'
            
            context_parts.append(f"PHOTOGRAPH [{i}]: {filename}")
            context_parts.append(f"Content Analysis: {desc[:800]}")  # Increased limit for better analysis
            context_parts.append("---")
        
        context_parts.extend([
            "",
            "STEP 3 - RANKING BY VISUAL RELEVANCE:",
            "Prioritize photographs that SHOW the main subject most clearly:",
            "1. VISUAL MATCH: Does this photo clearly show what the user is looking for?",
            "   - PRIORITY 1: Photos with STRONG visual presence of the main subject",
            "   - PRIORITY 2: Photos with MODERATE visual presence", 
            "   - PRIORITY 3: Photos with WEAK visual presence",
            "   - LOWEST: Photos with NO visual presence (only mentioned in text)",
            "",
            "2. VISUAL QUALITY: How clearly can you see the main subject?",
            "3. CONTEXT RELEVANCE: Does the setting/situation match the query?",
            "4. HISTORICAL AUTHENTICITY: Is this genuine documentation of the subject?",
            "",
            "STEP 4 - FINAL RANKING DECISION:",
            "⚠️  CRITICAL: The photos above are in RANDOM ORDER. You must create your OWN ranking.",
            "Analyze each photograph independently and rank by how well they VISUALLY demonstrate the main subject:",
            "",
            "RANKING PRIORITY:",
            "1st PRIORITY: Photos where you can CLEARLY SEE the main subject/concept",
            "2nd PRIORITY: Photos where the main subject is SOMEWHAT VISIBLE", 
            "3rd PRIORITY: Photos where the main subject is BARELY VISIBLE",
            "LAST PRIORITY: Photos where the main subject is only mentioned in text but NOT VISIBLE",
            "",
            "Remember: The user wants to SEE evidence of their query, not just read about it.",
            "Ignore the current order - create your own ranking based on visual relevance to the query.",
            "",
            "OUTPUT FORMAT:",
            "Provide ONLY a JSON array with photograph indices in order of historical relevance:",
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
            
            # Reorder results based on ranking
            reranked = []
            for rank_idx in ranking_json[:top_k]:
                print(f"🔄 DEBUG: Processing rank index {rank_idx}")
                if isinstance(rank_idx, int) and 1 <= rank_idx <= len(results):
                    result_copy = results[rank_idx - 1].copy()  # Convert to 0-based index
                    result_copy['gemma_rank'] = len(reranked) + 1
                    reranked.append(result_copy)
                    filename = result_copy.get('original_path', '').split('/')[-1] if result_copy.get('original_path') else 'Unknown'
                    print(f"  ✅ Added: {filename} (original pos {rank_idx}, new pos {len(reranked)})")
                else:
                    print(f"  ❌ Invalid index: {rank_idx} (not in range 1-{len(results)})")
            
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