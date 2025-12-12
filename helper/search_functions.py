
# Internal search functions
from typing import Optional

from db import MILVUS_COLLECTION_NAME, milvus_client
from helper.tool import get_embedding
from type.search import SearchResponse
from pymilvus import  AnnSearchRequest, RRFRanker
from helper.chat_utils import format_results


async def perform_hybrid_search(query: str, limit: int, filter_expr: Optional[str], return_text: bool = True) -> SearchResponse:
    """Perform hybrid search combining BM25 and semantic search."""
    # Get embedding for semantic search
    embedding = get_embedding(query)
    
    # BM25 search parameters
    search_param_1 = {
        "data": [query],
        "anns_field": "sparce_vector",
        "param": {},
        "limit": limit
    }
    if filter_expr:
        search_param_1["expr"] = filter_expr
    request_1 = AnnSearchRequest(**search_param_1)
    
    # Semantic search parameters
    search_param_2 = {
        "data": [embedding],
        "anns_field": "dense_vector",
        "param": {"drop_ratio_search": 0.2},
        "limit": limit
    }
    if filter_expr:
        search_param_2["expr"] = filter_expr
    request_2 = AnnSearchRequest(**search_param_2)
    
    # Determine output fields based on return_text parameter
    output_fields = ['text','language'] if return_text else []
    
    # Perform hybrid search
    ranker = RRFRanker()
    results = milvus_client.hybrid_search(
        collection_name=MILVUS_COLLECTION_NAME,
        reqs=[request_1, request_2],
        ranker=ranker,
        limit=limit,
        output_fields=output_fields
    )
    
    return format_results(results, query, "hybrid")


async def perform_bm25_search(query: str, limit: int, filter_expr: Optional[str], return_text: bool = True) -> SearchResponse:
    """Perform BM25 (sparse vector) search."""
    # Determine output fields based on return_text parameter
    output_fields = ['text','language'] if return_text else []
    
    # Prepare search parameters
    search_params = {
        "collection_name": MILVUS_COLLECTION_NAME,
        "data": [query],
        "anns_field": "sparce_vector",
        "limit": limit,
        "output_fields": output_fields
    }
    
    if filter_expr:
        search_params["filter"] = filter_expr
    
    # Perform BM25 search
    results = milvus_client.search(**search_params)
    
    return format_results(results, query, "bm25")


async def perform_semantic_search(query: str, limit: int, filter_expr: Optional[str], return_text: bool = True) -> SearchResponse:
    """Perform semantic (dense vector) search."""
    # Get embedding
    embedding = get_embedding(query)
    
    # Determine output fields based on return_text parameter
    output_fields = ['text','language'] if return_text else []
    
    # Prepare search parameters
    search_params = {
        "collection_name": MILVUS_COLLECTION_NAME,
        "data": [embedding],
        "anns_field": "dense_vector",
        "limit": limit,
        "output_fields": output_fields
    }
    
    if filter_expr:
        search_params["filter"] = filter_expr
    
    # Perform semantic search
    results = milvus_client.search(**search_params)
    
    return format_results(results, query, "semantic")


async def perform_exact_search(query: str, limit: int, filter_expr: Optional[str], return_text: bool = True) -> SearchResponse:
    """Perform exact phrase match search."""
    # Escape single quotes in query to prevent filter syntax errors
    escaped_query = query.replace("'", "\\'")
    
    # Build filter expression for exact phrase match
    phrase_filter = f"PHRASE_MATCH(text, '{escaped_query}')"
    
    # Combine filters if additional filter exists
    if filter_expr:
        final_filter = f"{phrase_filter} && {filter_expr}"
    else:
        final_filter = phrase_filter
    
    # Determine output fields based on return_text parameter
    output_fields = ['text'] if return_text else []
    
    # Prepare search parameters
    search_params = {
        "collection_name": MILVUS_COLLECTION_NAME,
        "data": [query],
        "anns_field": "sparce_vector",
        "limit": limit,
        "output_fields": output_fields,
        "filter": final_filter
    }
    
    # Perform exact match search
    results = milvus_client.search(**search_params)
    
    return format_results(results, query, "exact")