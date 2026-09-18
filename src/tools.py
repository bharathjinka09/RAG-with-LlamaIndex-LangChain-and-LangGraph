from langchain.tools import tool
from src.ingestion import get_llamaindex_query_engine

_query_engine = None

def get_engine():
    global _query_engine
    if _query_engine is None:
        _query_engine = get_llamaindex_query_engine()
    return _query_engine

@tool
def titan_docs_retriever(query: str) -> str:
    """Useful to search internal documentation regarding Project Titan, latency SLAs, and system specifications."""
    engine = get_engine()
    response = engine.query(query)
    return str(response)

RETRIEVAL_TOOLS = [titan_docs_retriever]
