import json
from db import get_milvus_client
from langchain_core.tools import tool
import os
from dotenv import load_dotenv, find_dotenv
from pymilvus import  AnnSearchRequest, RRFRanker

from typing import List,Dict
from google import genai
from google.genai.types import EmbedContentConfig

from helper.config import EMBEDDING_MODEL, HIGH_PRIORITY_MODEL

loaded = load_dotenv(find_dotenv(filename=".env", usecwd=True), override=True)

MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "test_kangyur_tengyur")



GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
genai_client = genai.Client(api_key=GEMINI_API_KEY)
doc_cfg = EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768)

# Helper: Get Embedding
def get_embedding(text: str) -> List[float]:
    """Generate embedding for the given text using Gemini."""
    resp = genai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=doc_cfg
    )
    return [i.values for i in resp.embeddings][0]


# Helper: Expand Query
def generate_expanded_queries(query: str) -> Dict[str, str]:
    """
    Generates 4 expanded queries: 
    2 for BM25 (Tibetan/English keywords), 2 for Semantic (Tibetan/English questions).
    """
    prompt = f"""
    You are an expert search query optimizer for a Tibetan Buddhist database.
    The user query is: "{query}"
    
    Generate 4 varied search queries to maximize retrieval recall:
    1. "tibetan_bm25": Key Tibetan terms/phrases (for keyword match).
    2. "english_bm25": Key English terms/phrases (for keyword match).
    3. "tibetan_semantic": A natural language question/statement in Tibetan capturing the meaning.
    4. "english_semantic": A natural language question/statement in English capturing the meaning.
    
    Return ONLY a JSON object with these 4 keys.
    """
    try:
        response = genai_client.models.generate_content(
            model=HIGH_PRIORITY_MODEL,
            contents=prompt,
            config={'response_mime_type': 'application/json'}
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error expanding query: {e}")
        # Fallback: just use original query for all if expansion fails
        return {
            "tibetan_bm25": query,
            "english_bm25": query,
            "tibetan_semantic": query,
            "english_semantic": query
        }



# --- Custom Hybrid Search Tool with Expansion ---
@tool
def hybrid_search_tool(query: str):
    """
    Searches the Tibetan knowledge base. 
    Automatically expands the query into Tibetan and English variations 
    (keywords and semantic) to improve coverage.
    Returns relevant text segments with metadata.
    """
    limit = 15 # Higher limit since we dedup later
    
    # 1. Generate Expanded Queries
    expanded = generate_expanded_queries(query)
    print(f"DEBUG: Expanded queries: {expanded}")
    
    queries_to_run = list(expanded.values())
    # Ensure original query is included if not covered? 
    # The expanded ones should cover it.
    
    all_results = []
    
    # 2. Run searches for each variation
    for q_text in queries_to_run:
        if not q_text or not q_text.strip():
            continue
            
        try:
            # Generate embedding
            q_emb = get_embedding(q_text)
            
            # BM25 Req
            req_bm25 = AnnSearchRequest(
                data=[q_text],
                anns_field="sparce_vector",
                param={},
                limit=limit
            )
            
            # Dense Req
            req_dense = AnnSearchRequest(
                data=[q_emb],
                anns_field="dense_vector",
                param={"drop_ratio_search": 0.2},
                limit=limit
            )
            
            # Hybrid Search
            res = get_milvus_client().hybrid_search(
                collection_name=MILVUS_COLLECTION_NAME,
                reqs=[req_bm25, req_dense],
                ranker=RRFRanker(),
                limit=limit,
                output_fields=["text", "title", "id"]
            )
            
            # Collect hits
            for hits in res:
                for hit in hits:
                    all_results.append(hit)
                    
        except Exception as e:
            print(f"Error searching for '{q_text}': {e}")
            continue

    # 3. Deduplicate and Format
    unique_items = {}
    for hit in all_results:
        entity = hit.get("entity", {})
        # ID might be in 'id' or 'entity.id'
        item_id = str(hit.get("id") or entity.get("id") or "")
        
        if item_id and item_id not in unique_items:
            unique_items[item_id] = {
                "id": item_id,
                "title": entity.get("title", "Unknown"),
                "text": entity.get("text", ""),
                "score": hit.get("score", 0),
                "distance": hit.get("distance", 0)
            }
            
    # Sort by score/distance? RRF scores are relative.
    # Let's just return them list values.
    final_items = list(unique_items.values())
    
    # Limit total return size
    final_items = final_items[:20] 
    
    # Return results + queries
    output = {
        "results": final_items,
        "queries": expanded
    }
    
    return json.dumps(output, ensure_ascii=False)
