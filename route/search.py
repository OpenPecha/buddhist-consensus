from fastapi import APIRouter, FastAPI, HTTPException

from db import MILVUS_COLLECTION_NAME, get_milvus_client
from helper.chat_utils import build_children_filter_from_parents, build_filter_expression, combine_filters
from helper.search_functions import perform_bm25_search, perform_exact_search, perform_hybrid_search, perform_semantic_search
from type.search import SearchRequest, SearchResponse

app = FastAPI()
search_router = APIRouter()

@search_router.get("/info", tags=["search"])
async def info():
    """Root endpoint with API information."""
    return {
        "message": "OpenPecha Search API",
        "version": "1.0.0",
        "endpoints": {
            "search": "/search"
        },
        "search_types": {
            "hybrid": "Combined BM25 + semantic search (default)",
            "bm25": "Keyword-based search",
            "semantic": "Meaning-based search",
            "exact": "Exact phrase matching"
        },
        "docs": "/docs"
    }

@search_router.get("/debug", tags=["search"])
def debug_search():
    """Debug endpoint to test basic search functionality."""
    import sys
    print('DEBUG ENDPOINT HIT', flush=True)
    sys.stdout.flush()
    try:
        # Test basic BM25 search without any fancy options
        results = get_milvus_client().search(
            collection_name=MILVUS_COLLECTION_NAME,
            data=["how to worry less?"],
            anns_field="sparce_vector",
            limit=5,
            output_fields=["text"]
        )
        
        return {
            "status": "success",
            "raw_results": str(results),
            "results_type": str(type(results)),
            "results_length": len(results),
            "first_result": str(results[0]) if results else None
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": str(type(e))
        }


@search_router.post("", tags=["search"])
def unified_search_post(req: SearchRequest):
    """
    Unified search endpoint (POST) accepting JSON body.
    Mirrors the GET /search behavior using the SearchRequest schema.
    """
    print(req)
    try:
        search_type_lower = req.search_type.lower()
        
        # Validate search type
        valid_types = ["hybrid", "bm25", "semantic", "exact"]
        if search_type_lower not in valid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid search_type. Must be one of: {', '.join(valid_types)}"
            )
        
        # Build base filter expression from structured filter
        base_filter_expr = build_filter_expression(req.filter)
        print(base_filter_expr)
        
        # If not hierarchical, run single-stage as before
        if not req.hierarchical:
            if search_type_lower == "hybrid":
                return perform_hybrid_search(req.query, req.limit, base_filter_expr, req.return_text)
            elif search_type_lower == "bm25":
                return perform_bm25_search(req.query, req.limit, base_filter_expr, req.return_text)
            elif search_type_lower == "semantic":
                return perform_semantic_search(req.query, req.limit, base_filter_expr, req.return_text)
            elif search_type_lower == "exact":
                return perform_exact_search(req.query, req.limit, base_filter_expr, req.return_text)
        
        # Hierarchical: parents -> children; return only children
        parent_limit = req.parent_limit if req.parent_limit is not None else req.limit
        parent_stage_filter = combine_filters(base_filter_expr, 'parent_id == ""')
        
        # Stage 1: parents (no text needed)
        if search_type_lower == "hybrid":
            parent_resp = perform_hybrid_search(req.query, parent_limit, parent_stage_filter, return_text=False)
        elif search_type_lower == "bm25":
            parent_resp = perform_bm25_search(req.query, parent_limit, parent_stage_filter, return_text=False)
        elif search_type_lower == "semantic":
            parent_resp = perform_semantic_search(req.query, parent_limit, parent_stage_filter, return_text=False)
        else:  # exact
            parent_resp = perform_exact_search(req.query, parent_limit, parent_stage_filter, return_text=False)
        
        parent_ids = [r.id for r in parent_resp.results]
        if not parent_ids:
            return SearchResponse(query=req.query, search_type=search_type_lower, results=[], count=0)
        
        children_filter_expr = combine_filters(base_filter_expr, build_children_filter_from_parents(parent_ids))
        
        # Stage 2: children (respect return_text)
        if search_type_lower == "hybrid":
            return perform_hybrid_search(req.query, req.limit, children_filter_expr, req.return_text)
        elif search_type_lower == "bm25":
            return perform_bm25_search(req.query, req.limit, children_filter_expr, req.return_text)
        elif search_type_lower == "semantic":
            return perform_semantic_search(req.query, req.limit, children_filter_expr, req.return_text)
        else:  # exact
            return perform_exact_search(req.query, req.limit, children_filter_expr, req.return_text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")