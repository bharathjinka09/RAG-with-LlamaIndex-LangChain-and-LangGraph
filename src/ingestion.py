from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)

STORAGE_DIR = "./storage"

def build_or_load_index(data_dir: str = "./data") -> VectorStoreIndex:
    storage_path = Path(STORAGE_DIR)
    if storage_path.exists():
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        return load_index_from_storage(storage_context)
    
    docs = SimpleDirectoryReader(input_dir=data_dir).load_data()
    index = VectorStoreIndex.from_documents(docs)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    return index

def get_llamaindex_query_engine():
    index = build_or_load_index()
    return index.as_query_engine(similarity_top_k=3)
