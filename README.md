# Webuddhist AI

A FastAPI-based AI application for searching and chatting with Buddhist texts using RAG (Retrieval-Augmented Generation) technology.

## Overview

Webuddhist AI provides an intelligent search and chat interface for Buddhist texts, supporting multiple search types including hybrid, semantic, BM25, and exact matching. The API uses LangGraph for agentic workflows and integrates with Milvus vector database for efficient text retrieval.

## API Routes

### Root & Health Endpoints

#### `GET /`

Returns the HTML chat interface.

**Response:** HTML content (text/html)

---

#### `GET /health`

Health check endpoint that verifies environment variables and service status.

**Response:**

```json
{
  "status": "healthy"
}
```

**Error Response (500):**

```json
{
  "detail": "Missing environment variables for Milvus or Gemini."
}
```

---

### Chat Endpoints

#### `POST /api/chat/stream`

Streaming chat endpoint using Server-Sent Events (SSE). This endpoint processes chat messages through a LangGraph workflow and streams responses in real-time.

**Request Body:**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is the meaning of compassion?"
    }
  ]
}
```

**Response:** Server-Sent Events stream with the following event types:

- `search_results`: Search results from hybrid search tool
- `token`: Streaming text tokens from the AI model
- `done`: Indicates completion
- `error`: Error information if something goes wrong

**Example Event:**

```
data: {"type": "token", "data": "Compassion is..."}

data: {"type": "search_results", "data": [...], "queries": {...}}

data: {"type": "done", "data": {}}
```

---

### Search Endpoints

All search endpoints are prefixed with `/search`.

#### `GET /search/info`

Returns API information and available search types.

**Response:**

```json
{
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
```

---

#### `GET /search/debug`

Debug endpoint to test basic search functionality.

**Response:**

```json
{
  "status": "success",
  "raw_results": "...",
  "results_type": "...",
  "results_length": 5,
  "first_result": "..."
}
```

---

#### `POST /search`

Unified search endpoint supporting multiple search types with filtering and hierarchical search capabilities.

**Request Body:**

```json
{
  "query": "དེ་ལ་མི་དགར་ཅི་ཞིག་ཡོད། །",
  "search_type": "hybrid",
  "limit": 10,
  "return_text": true,
  "hierarchical": false,
  "parent_limit": null,
  "filter": {
    "title": "Some Title",
    "language": "bo"
  }
}
```

**Request Parameters:**

| Parameter        | Type    | Required | Default  | Description                                                     |
| ---------------- | ------- | -------- | -------- | --------------------------------------------------------------- |
| `query`        | string  | Yes      | -        | The search query text (min length: 1)                           |
| `search_type`  | string  | No       | "hybrid" | Type of search:`hybrid`, `bm25`, `semantic`, or `exact` |
| `limit`        | integer | No       | 10       | Maximum number of results (1-100)                               |
| `return_text`  | boolean | No       | true     | If true, return full text in results                            |
| `hierarchical` | boolean | No       | false    | If true, perform parent->children two-stage search              |
| `parent_limit` | integer | No       | null     | Max parents to retrieve when hierarchical=true (1-200)          |
| `filter`       | object  | No       | null     | Optional filters (title, language)                              |

**Filter Object:**

```json
{
  "title": "Title Name" | ["Title1", "Title2"],
  "language": "bo" | ["bo", "en"]
}
```

**Response:**

```json
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
    }
  ],
  "count": 1
}
```

**Search Types:**

1. **hybrid** (default): Combines BM25 keyword search with semantic vector search for best results
2. **bm25**: Keyword-based search using BM25 algorithm
3. **semantic**: Meaning-based search using vector embeddings
4. **exact**: Exact phrase matching

**Hierarchical Search:**

When `hierarchical: true`, the search performs a two-stage process:

1. First searches for parent documents matching the query
2. Then searches for children of those parents
3. Returns only the children results

This is useful for structured documents with parent-child relationships.

---

## Environment Variables

The following environment variables are required:

- `GEMINI_API_KEY` or `GOOGLE_API_KEY`: Google Gemini API key for LLM
- `MILVUS_URI`: Milvus vector database URI
- `MILVUS_TOKEN`: Milvus authentication token
- `MILVUS_COLLECTION_NAME`: Name of the Milvus collection (default: "test_kangyur_tengyur")
- `PORT`: Server port (default: 8000)
- `ENV`: Environment mode (set to "development" for auto-reload)

---

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

The API will be available at `http://localhost:8000` (or the port specified in `PORT`).

Interactive API documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Technology Stack

- **FastAPI**: Web framework
- **LangGraph**: Agentic workflow orchestration
- **LangChain**: LLM integration
- **Google Gemini**: Language model
- **Milvus**: Vector database for semantic search
- **Server-Sent Events (SSE)**: Real-time streaming

---

## License

See LICENSE file for details.
