# file_handler.py
import os
from typing import List
from langchain_core.documents import Document
from llama_parser_handler import parse_bytes_to_documents

# Supported Types
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt"}

def handle_uploaded_file_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Main entry point for file processing.
    Routes everything through LlamaParse for consistent handling.
    """
    ext = os.path.splitext(filename)[1].lower()
    
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    # If it's a simple text file, just read it directly to save API credits
    if ext == ".txt":
        text = file_bytes.decode("utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": filename, "page": 1})]

    # For PDFs and Images, use LlamaParse
    return parse_bytes_to_documents(file_bytes, filename)