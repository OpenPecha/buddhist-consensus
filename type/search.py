from pydantic import BaseModel, Field
from typing import Optional, Union, List, Dict, Any


class SearchFilter(BaseModel):
    title: Optional[Union[str, List[str]]] = Field(None, description="Filter results by title or list of titles")
    language: Optional[Union[str, List[str]]] = Field(None, description="Filter results by language or list of languages")


class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query text", min_length=1, examples=["དེ་ལ་མི་དགར་ཅི་ཞིག་ཡོད། །"])
    search_type: str = Field("hybrid", description="Type of search: 'hybrid', 'bm25', 'semantic', or 'exact'", examples=["hybrid"])
    limit: int = Field(10, description="Maximum number of results to return", ge=1, le=100, examples=[10])
    return_text: bool = Field(True, description="If True, return full text in results. If False, return only ID and distance", examples=[True])
    hierarchical: bool = Field(False, description="If true, perform parent->children two-stage search and return only children", examples=[False])
    parent_limit: Optional[int] = Field(None, description="Max parents to retrieve when hierarchical=true; defaults to 'limit'", ge=1, le=200, examples=[20])
    filter: Optional[SearchFilter] = Field(None, description="Optional filters to apply")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "དེ་ལ་མི་དགར་ཅི་ཞིག་ཡོད། །",
                    "search_type": "hybrid",
                    "limit": 10,
                    "return_text": True,
                    "filter": None
                },
                {
                    "query": "how to worry less?",
                    "search_type": "semantic",
                    "limit": 5,
                    "return_text": True,
                    "filter": None
                },
                {
                    "query": "དེ་ལ་མི་དགར་ཅི་ཞིག་ཡོད། །",
                    "search_type": "exact",
                    "limit": 10,
                    "return_text": True,
                    "filter": None
                },
                {
                    "query": "ཕམ་པར་གྱུར་བའི་ཆོས་དུན་པ",
                    "search_type": "bm25",
                    "limit": 10,
                    "return_text": False,
                    "filter": None
                },
                {
                    "query": "དེ་ལ་མི་དགར་ཅི་ཞིག་ཡོད། །",
                    "search_type": "hybrid",
                    "limit": 10,
                    "parent_limit": 20,
                    "hierarchical": True,
                    "return_text": True,
                    "filter": {"title": "Some Title"}
                }
            ]
        }
    }


class SearchResult(BaseModel):
    id: Any
    distance: float
    entity: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    search_type: str
    results: List[SearchResult]
    count: int
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "དེ་ལ་མི་དགར་ཅི་ཞིག་ཡོད། །",
                    "search_type": "hybrid",
                    "results": [
                        {
                            "id": "449691587532670411",
                            "distance": 0.95,
                            "entity": {
                                "text": "དེ་ལ་མི་དགར་ཅི་ཞིག་ཡོད། །གང་ཕྱིར་འདི་དག་རང་བཞིན་མེད།"
                            }
                        },
                        {
                            "id": "449691587532670412",
                            "distance": 0.87,
                            "entity": {
                                "text": "སངས་རྒྱས་ཀྱི་བསྟན་པ་ལ་གུས་པར་བྱོས།"
                            }
                        }
                    ],
                    "count": 2
                },
                {
                    "query": "how to worry less?",
                    "search_type": "semantic",
                    "results": [
                        {
                            "id": "449691587532670413",
                            "distance": 0.82,
                            "entity": {
                                "text": "སེམས་ཅན་ཐམས་ཅད་བདེ་བར་གྱུར་ཅིག"
                            }
                        }
                    ],
                    "count": 1
                }
            ]
        }
    }