import os
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

PERSIST_DIR = os.getenv("PERSIST_DIRECTORY", "./persist/chroma_db_prod")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "multi_rag")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))

LOG = logging.getLogger(__name__)
LOG.setLevel(os.getenv("LOG_LEVEL", "INFO"))


def create_vector_store_from_documents(documents: List[Document], persist_directory: Optional[str] = None):
    if not documents:
        LOG.error("No documents provided to create vector store.")
        return None
    persist = persist_directory or PERSIST_DIR
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
    try:
# NEW (Correct) -> changed 'embedding_function' to 'embedding'
        vectordb = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=persist, collection_name=COLLECTION_NAME)
        if hasattr(vectordb, "persist"):
            vectordb.persist()
        LOG.info("Created Chroma at %s", persist)
        return vectordb
    except Exception as e:
        LOG.exception("Failed to create Chroma: %s", e)
        return None


def get_existing_retriever(persist_directory: Optional[str] = None):
    persist = persist_directory or PERSIST_DIR
    if not os.path.isdir(persist):
        LOG.warning("Persist directory missing: %s", persist)
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)
        vectordb = Chroma(persist_directory=persist, embedding_function=embeddings, collection_name=COLLECTION_NAME)
        # best-effort
        try:
            cnt = vectordb._collection.count()
            LOG.info("Loaded collection %s with %d vectors", COLLECTION_NAME, cnt)
        except Exception:
            LOG.debug("Could not read internal collection count")
        return vectordb.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    except Exception as e:
        LOG.exception("Failed to load Chroma: %s", e)
        return None
