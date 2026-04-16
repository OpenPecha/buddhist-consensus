
# Internal search functions
from typing import Optional

from db import MILVUS_COLLECTION_NAME, get_milvus_client
from helper.tool import get_embedding
from type.search import SearchResponse
from pymilvus import AnnSearchRequest, RRFRanker
from helper.chat_utils import _escape_milvus_string, format_results


def perform_hybrid_search(query: str, limit: int, filter_expr: Optional[str], return_text: bool = True) -> SearchResponse:
    """Perform hybrid search combining BM25 and semantic search."""
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
    output_fields = ['text', 'language'] if return_text else []
    
    # Perform hybrid search
    ranker = RRFRanker()
    results = get_milvus_client().hybrid_search(
        collection_name=MILVUS_COLLECTION_NAME,
        reqs=[request_1, request_2],
        ranker=ranker,
        limit=limit,
        output_fields=output_fields
    )
    
    return format_results(results, query, "hybrid")


def perform_bm25_search(query: str, limit: int, filter_expr: Optional[str], return_text: bool = True) -> SearchResponse:
    """Perform BM25 (sparse vector) search."""
    output_fields = ['text', 'language'] if return_text else []
    
    search_params = {
        "collection_name": MILVUS_COLLECTION_NAME,
        "data": [query],
        "anns_field": "sparce_vector",
        "limit": limit,
        "output_fields": output_fields
    }
    
    if filter_expr:
        search_params["filter"] = filter_expr
    
    results = get_milvus_client().search(**search_params)
    
    return format_results(results, query, "bm25")


def perform_semantic_search(query: str, limit: int, filter_expr: Optional[str], return_text: bool = True) -> SearchResponse:
    """Perform semantic (dense vector) search."""
    embedding = get_embedding(query)
    
    output_fields = ['text', 'language'] if return_text else []
    
    search_params = {
        "collection_name": MILVUS_COLLECTION_NAME,
        "data": [embedding],
        "anns_field": "dense_vector",
        "limit": limit,
        "output_fields": output_fields
    }
    
    if filter_expr:
        search_params["filter"] = filter_expr
    
    results = get_milvus_client().search(**search_params)
    
    return format_results(results, query, "semantic")


def _escape_like_substring(value: str) -> str:
    """Escape backslash, %, and _ so the substring is matched literally in Milvus LIKE."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def perform_exact_search(query: str, limit: int, filter_expr: Optional[str], return_text: bool = True) -> SearchResponse:
    """Match rows whose text contains the query as a contiguous substring (SQL LIKE %...%)."""
    pattern = f"%{_escape_like_substring(query)}%"
    contains_filter = f'text LIKE "{_escape_milvus_string(pattern)}"'

    if filter_expr:
        final_filter = f"{contains_filter} && {filter_expr}"
    else:
        final_filter = contains_filter

    output_fields = ['text'] if return_text else []
    search_params = {
        "collection_name": MILVUS_COLLECTION_NAME,
        "data": [query],
        "anns_field": "sparce_vector",
        "limit": limit,
        "output_fields": output_fields,
        "filter": final_filter
    }
    
    results = get_milvus_client().search(**search_params)
    
    return format_results(results, query, "exact")