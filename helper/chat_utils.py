
# Helper function to build filter expression
from typing import Any, List, Optional
from type.search import SearchFilter, SearchResponse, SearchResult


def build_filter_expression(filter_obj: Optional[SearchFilter]) -> Optional[str]:
    """Build Milvus filter expression from filter object."""
    if not filter_obj:
        return None
    
    conditions = []
    if filter_obj.title:
        # Support single string or list of strings for title filter
        if isinstance(filter_obj.title, list):
            titles = [ _escape_milvus_string(str(title)) for title in filter_obj.title if str(title).strip() != "" ]
            if titles:
                conditions.append(f'title in ["{"\", \"".join(titles)}"]')
        else:
            conditions.append(f'title == "{_escape_milvus_string(str(filter_obj.title))}"')
    if filter_obj.language:
        # Support single string or list of strings for language filter
        if isinstance(filter_obj.language, list):
            langs = [ _escape_milvus_string(str(lang)) for lang in filter_obj.language if str(lang).strip() != "" ]
            if langs:
                conditions.append(f'language in ["{"\", \"".join(langs)}"]')
        else:
            conditions.append(f'language == "{_escape_milvus_string(str(filter_obj.language))}"')
    
    return " && ".join(conditions) if conditions else None




# Helpers for hierarchical filtering
def _escape_milvus_string(value: str) -> str:
    """Escape characters for use inside Milvus string literals."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def combine_filters(*filters: Optional[str]) -> Optional[str]:
    """Combine multiple filter expressions using logical AND."""
    parts = [f for f in filters if f]
    return " && ".join(parts) if parts else None


def build_children_filter_from_parents(parent_ids: List[Any]) -> Optional[str]:
    """Build a filter expression for children of the given parent ids."""
    if not parent_ids:
        return None
    quoted = '", "'.join(_escape_milvus_string(str(pid)) for pid in parent_ids)
    return f'parent_id in ["{quoted}"]'


# Helper function to format results
def format_results(results: List, query: str, search_type: str) -> SearchResponse:
    """Format raw Milvus results into SearchResponse."""
    formatted_results = []
    
    for result_list in results:
        for hit in result_list:
            formatted_results.append(SearchResult(
                id=hit.get('id'),
                distance=hit.get('distance', 0.0),
                entity=hit.get('entity', {})
            ))
    
    return SearchResponse(
        query=query,
        search_type=search_type,
        results=formatted_results,
        count=len(formatted_results)
    )

