import streamlit as st
import os
from dotenv import load_dotenv

# Load Logic
from file_handler import handle_uploaded_file_bytes
from data_loader import chunk_documents
from vector_store_handler import create_vector_store_from_documents, get_existing_retriever
from chain_handler import run_rag_chain

load_dotenv()

# --- 1. Page Config ---
st.set_page_config(
    page_title="Multi-Modal RAG",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Gemini-Style CSS (White Theme) ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E5E7EB;
    }
    
    /* Chat Message Bubbles */
    .stChatMessage {
        background-color: transparent;
        border: none;
        padding: 10px 0;
    }
    
    /* User Message - Grey Bubble */
    [data-testid="stChatMessageUser"] {
        background-color: #F3F4F6;
        border-radius: 20px;
        padding: 15px 20px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Assistant Message - White Bubble with Border */
    [data-testid="stChatMessageAssistant"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 20px;
        padding: 15px 20px;
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Input Box (Floating) */
    .stChatInputContainer {
        padding-bottom: 20px;
    }
    .stChatInputContainer > div {
        background-color: #FFFFFF;
        border-radius: 30px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* Citation Expander Styling */
    .streamlit-expanderHeader {
        background-color: #F9FAFB;
        border-radius: 10px;
        font-size: 0.9rem;
        color: #4B5563;
    }
    
    /* Hero Header */
    .hero-container {
        text-align: center;
        padding: 40px 0;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #4F46E5, #9333EA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 30px;
    }
    
    /* Example Button Styling */
    .stButton button {
        border-radius: 20px;
        border: 1px solid #E5E7EB;
        background-color: white;
        color: #4B5563;
        transition: all 0.2s;
    }
    .stButton button:hover {
        border-color: #4F46E5;
        color: #4F46E5;
        background-color: #EEF2FF;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Session State ---
if "retriever" not in st.session_state:
    st.session_state.retriever = get_existing_retriever()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# --- 4. Sidebar (Add Files) ---
with st.sidebar:
    st.markdown("### 📚 Knowledge Base")
    
    with st.expander("➕ Add Documents", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, Images", 
            type=["pdf", "png", "jpg", "jpeg", "txt"], 
            accept_multiple_files=True
        )
        
        if st.button("🚀 Process & Index", type="primary", use_container_width=True):
            if not uploaded_files:
                st.toast("⚠️ Please select a file first.", icon="📂")
            else:
                with st.status("⚙️ Processing...", expanded=True) as status:
                    all_docs = []
                    new_files = []
                    for f in uploaded_files:
                        if f.name not in st.session_state.processed_files:
                            file_bytes = f.read()
                            try:
                                docs = handle_uploaded_file_bytes(file_bytes, f.name)
                                all_docs.extend(docs)
                                new_files.append(f.name)
                                st.session_state.processed_files.add(f.name)
                            except Exception as e:
                                st.error(f"Error: {e}")
                    
                    if all_docs:
                        st.write("🧩 Chunking & Embedding...")
                        chunks = chunk_documents(all_docs)
                        create_vector_store_from_documents(chunks)
                        st.session_state.retriever = get_existing_retriever()
                        status.update(label="✅ Indexing Complete!", state="complete", expanded=False)
                        st.toast(f"Added {len(new_files)} documents!", icon="🎉")
                    else:
                        status.update(label="⚠️ No new data", state="error")

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- 5. Main UI ---

# Logic to handle button clicks (Callback)
def click_question(q):
    st.session_state.messages.append({"role": "user", "content": q})

# Hero Section (if empty)
if not st.session_state.messages:
    st.markdown("""
        <div class="hero-container">
            <div class="hero-title">Multi-Modal RAG</div>
            <div class="hero-subtitle">Ask questions about your Documents, Charts, and Images</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Example Questions Grid
    st.markdown("##### 💡 Try asking:")
    col1, col2 = st.columns(2)
    
    q1 = "Summarize the trend of Real Estate Prices from the charts."
    q2 = "What is the projected Real GDP growth for 2025 vs 2024?"
    q3 = "summarize the content in the given document"
    q4 = "Describe the trend of Real Estate Prices from the charts.?"

    with col1:
        if st.button(f"📊 {q1}", use_container_width=True): click_question(q1)
        if st.button(f"⚠️ {q2}", use_container_width=True): click_question(q2)
    with col2:
        if st.button(f"📉 {q3}", use_container_width=True): click_question(q3)
        if st.button(f"🚀 {q4}", use_container_width=True): click_question(q4)

# Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # If it's an assistant message with sources, show them
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📚 Reference Sources"):
                for src in msg["sources"]:
                    st.markdown(f"- **{src['source']}** (Page {src['page']})")
                    st.caption(f"\"{src['preview']}...\"")

# Chat Input
# Check if last message was from user (triggered by button) to run RAG immediately
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_query = st.session_state.messages[-1]["content"]
    # Only run if we haven't answered yet (simple check: last msg is user)
    # But wait, Streamlit reruns script. We need to check if we already answered this specific interaction.
    # Actually, simpler: Put the RAG logic in a function or check length.
    pass 

# Handle Text Input
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun() # Rerun to show the user message immediately

# --- Generation Logic (Runs if last message is User) ---
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_query = st.session_state.messages[-1]["content"]
    
    with st.chat_message("assistant"):
        if not st.session_state.retriever:
            st.warning("⚠️ Please upload and index documents in the sidebar first.")
        else:
            with st.spinner("🧠 Thinking..."):
                try:
                    # Run RAG
                    result = run_rag_chain(user_query, [], st.session_state.retriever)
                    answer = result["answer"]
                    docs = result["source_documents"]
                    
                    st.markdown(answer)
                    
                    # Prepare Source Metadata
                    source_meta = []
                    with st.expander("📚 Reference Sources"):
                        seen_sources = set()
                        for d in docs:
                            source_id = f"{d.metadata.get('source')} - Pg {d.metadata.get('page')}"
                            if source_id not in seen_sources:
                                st.markdown(f"**📄 {d.metadata.get('source', 'Unknown')}** (Page {d.metadata.get('page', '?')})")
                                preview = d.page_content[:150].replace("\n", " ")
                                st.caption(f"_{preview}..._")
                                source_meta.append({
                                    "source": d.metadata.get('source'),
                                    "page": d.metadata.get('page'),
                                    "preview": preview
                                })
                                seen_sources.add(source_id)

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": source_meta
                    })
                    
                except Exception as e:
                    st.error(f"Error: {e}")