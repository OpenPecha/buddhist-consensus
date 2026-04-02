from pymilvus import MilvusClient

from dotenv import load_dotenv, find_dotenv
import os

loaded = load_dotenv(find_dotenv(filename=".env", usecwd=True), override=True)

MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "test_kangyur_tengyur")

_milvus_client = None

def get_milvus_client() -> MilvusClient:
    """Lazy initialization of Milvus client to avoid blocking at import time."""
    global _milvus_client
    if _milvus_client is None:
        uri = MILVUS_URI.rstrip('/') if MILVUS_URI else ""
        # Ensure URI ends with :443 for Zilliz Cloud HTTPS connections
        if uri.startswith("https://") and ":443" not in uri:
            uri = uri + ":443"
        print(f"Connecting to Milvus at {uri}...", flush=True)
        _milvus_client = MilvusClient(
            uri=uri,
            token=MILVUS_TOKEN,
            timeout=30,
        )
        print("Milvus client connected.", flush=True)
    return _milvus_client
