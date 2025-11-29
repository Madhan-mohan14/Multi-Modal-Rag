# setup_db.py
"""
Setup script to create (or refresh) the Chroma vector DB from files in data/uploads.
- Parses files with LlamaParse (Multi-Modal)
- Chunks documents
- Indexes into Chroma
"""

import os
import sys
from pathlib import Path
import logging
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
# Defines where your local PDF files are stored for indexing
UPLOADS = Path(os.getenv("SETUP_UPLOADS_DIR", ROOT / "data" / "uploads"))
PERSIST = os.getenv("PERSIST_DIRECTORY", "./persist/chroma_db_prod")

# Add root to path so local modules import cleanly
sys.path.insert(0, str(ROOT))

# --- THE FIX IS HERE ---
# We removed 'save_temp_file' from the import because it no longer exists
from file_handler import handle_uploaded_file_bytes
from data_loader import chunk_documents
from vector_store_handler import create_vector_store_from_documents

# Logging Setup
LOG = logging.getLogger("setup_db")
LOG.setLevel(os.getenv("LOG_LEVEL", "INFO"))
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
LOG.addHandler(ch)


def gather_files(upload_dir: Path):
    """Finds all valid files in the upload directory."""
    if not upload_dir.exists():
        LOG.error("Uploads directory does not exist: %s", upload_dir)
        return []
    
    # Filter for supported extensions only
    valid_exts = {".pdf", ".png", ".jpg", ".jpeg", ".txt"}
    files = [p for p in sorted(upload_dir.iterdir()) if p.is_file() and p.suffix.lower() in valid_exts]
    
    LOG.info("Found %d files in uploads dir: %s", len(files), upload_dir)
    return files


def parse_and_chunk(file_path: Path):
    """Reads a file and sends it through the LlamaParse pipeline."""
    LOG.info("Parsing file: %s", file_path.name)
    
    try:
        # Read file as bytes
        data = file_path.read_bytes()
        
        # Send to file_handler (which sends to LlamaParse)
        docs = handle_uploaded_file_bytes(data, file_path.name)
        
        if not docs:
            LOG.warning("No parsed docs for %s", file_path.name)
            return []
            
        # Chunk the parsed text
        chunks = chunk_documents(docs)
        LOG.info("Produced %d chunks from %s", len(chunks), file_path.name)
        return chunks
        
    except Exception as e:
        LOG.exception("Failed processing %s: %s", file_path.name, e)
        return []


def main():
    LOG.info("Starting setup_db...")
    
    # 1. Get Files
    files = gather_files(UPLOADS)
    if not files:
        LOG.error("No files found. Place files in %s and re-run.", UPLOADS)
        sys.exit(1)
        
    all_chunks = []
    
    # 2. Process Files
    for f in files:
        chks = parse_and_chunk(f)
        all_chunks.extend(chks)
        
    LOG.info("Total chunks generated: %d", len(all_chunks))
    
    if not all_chunks:
        LOG.error("No chunks to index. Exiting.")
        sys.exit(1)
        
    # 3. Create/Update Vector Store
    LOG.info("Creating Vector Store...")
    vs = create_vector_store_from_documents(all_chunks, persist_directory=PERSIST)
    
    if vs is None:
        LOG.error("Vector DB creation failed")
        sys.exit(2)
        
    LOG.info("✅ Vector DB successfully created at: %s", PERSIST)
    LOG.info("Setup_db completed.")


if __name__ == "__main__":
    main()