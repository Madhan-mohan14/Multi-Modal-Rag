# llama_parser_handler.py
import os
import tempfile
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from llama_parse import LlamaParse
from langchain_core.documents import Document as LangChainDocument
from multimodal_utils import safe_filename, normalize_markdown

load_dotenv()

LLAMA_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
if not LLAMA_API_KEY:
    raise ValueError("LLAMA_CLOUD_API_KEY is missing!")

def parse_bytes_to_documents(file_bytes: bytes, filename: str) -> List[LangChainDocument]:
    """
    Uses LlamaParse to convert PDF/Images into Markdown text.
    Enabled with Multimodal Vision for charts/graphs.
    """
    safe_name = safe_filename(filename)
    file_ext = os.path.splitext(filename)[1] or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        # Initialize Parser with VISION capabilities
        parser = LlamaParse(
            api_key=LLAMA_API_KEY,
            result_type="markdown",
            verbose=True,
            language="en",
            # CRITICAL UPDATE: 'parsing_instruction' is deprecated. Use 'user_prompt'.
            user_prompt="Extract all text. For tables, preserve the structure exactly. For charts or graphs, provide a detailed textual description of the trends and data points.",
            # This forces it to use a Vision model (like GPT-4o) to 'see' charts
            use_vendor_multimodal_model=True,
            vendor_multimodal_model_name="openai-gpt-4o-mini"
        )

        # Execute Parse
        llama_docs = parser.load_data(tmp_path)
        
        langchain_docs = []
        for i, doc in enumerate(llama_docs):
            content = normalize_markdown(doc.text)
            if not content:
                continue
            
            meta = {
                "source": safe_name,
                "page": i + 1,
                "original_filename": filename
            }
            langchain_docs.append(LangChainDocument(page_content=content, metadata=meta))

        return langchain_docs

    except Exception as e:
        print(f"Error parsing file {filename}: {e}")
        return []
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)