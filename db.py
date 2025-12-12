from pymilvus import MilvusClient

from dotenv import load_dotenv, find_dotenv
import os

loaded = load_dotenv(find_dotenv(filename=".env", usecwd=True), override=True)

MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "test_kangyur_tengyur")

milvus_client = MilvusClient(
    uri=MILVUS_URI,
    token=MILVUS_TOKEN,
    collection_name=MILVUS_COLLECTION_NAME
)
