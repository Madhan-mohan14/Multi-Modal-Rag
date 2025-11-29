from typing import List, Iterable
import hashlib
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter,RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
DEDUP = os.getenv("DEDUP", "true").lower() in ("1", "true", "yes")#it helps user if he add twice it ignore and repetitive text it help to dedup catches it so it help to cost less 


def _hash_text(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:12]


def chunk_documents(docs: Iterable[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP, dedupe: bool = DEDUP) -> List[Document]:
    """
    Smart Chunking:
    1. Splits by Markdown Headers first (to keep logical sections together).
    2. Then splits by characters if the section is still too big.
    """

    # 1. Define Headers to split on
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
   # 2. Define Recursive Splitter for large sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    
    all_chunks = []
    seen_hashes = set()

    for doc in docs:
        content = doc.page_content
        original_meta = doc.metadata.copy()
        
        # A. Split by Markdown Structure
        md_docs = markdown_splitter.split_text(content)
        
        # B. Further split large sections
        final_chunks = text_splitter.split_documents(md_docs)
        
        for i, chunk in enumerate(final_chunks):
            # Merge original metadata (filename/page) with new header metadata
            combined_meta = {**original_meta, **chunk.metadata}
            
            # Create unique hash for deduplication
            chunk_hash = _hash_text(chunk.page_content)
            
            if chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)
                combined_meta.update({
                    "chunk_index": i, 
                    "chunk_hash": chunk_hash
                })
                all_chunks.append(Document(page_content=chunk.page_content, metadata=combined_meta))
                
    return all_chunks