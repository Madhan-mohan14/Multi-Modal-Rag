import os
import logging
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
# Ensure langchain_community is installed
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

GROQ_REPHRASE = os.getenv("GROQ_REPHRASE_MODEL", "llama-3.1-8b-instant")
GROQ_ANSWER = os.getenv("GROQ_ANSWER_MODEL", "llama-3.3-70b-versatile")

def build_rephrase_chain():
    template = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the search query to be precise and standalone."),
        ("human", "{input}")
    ])
    llm = ChatGroq(model=GROQ_REPHRASE)
    return template | llm

def build_answer_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         """You are an expert analyst. Answer the question based strictly on the provided context. 
         
         CRITICAL CITATION RULE:
         - Every factual statement must be backed by a citation from the context.
         - Use the format [Page X] at the end of the sentence.
         - If the answer is not in the context, state that you do not know.
         
         Context:
         {context}"""),
        ("human", "{input}")
    ])
    llm = ChatGroq(model=GROQ_ANSWER)
    return prompt | llm

def get_reranker_retriever(base_retriever):
    """
    Wraps the vector store retriever with a Reranker (FlashRank).
    """
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2") 
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    return compression_retriever

def run_rag_chain(question: str, history, base_retriever) -> Dict[str, Any]:
    """
    Returns a dictionary with 'answer' and 'source_documents'.
    """
    reranker = get_reranker_retriever(base_retriever)

    # 1. Rephrase
    try:
        rephraser = build_rephrase_chain()
        rewritten_query = rephraser.invoke({"input": question}).content
    except Exception:
        rewritten_query = question
    
    # 2. Retrieve & Rerank
    docs = reranker.invoke(rewritten_query)
    
    if not docs:
        return {"answer": "I couldn't find relevant information.", "source_documents": []}

    # 3. Format Context
    context_parts = []
    for d in docs:
        source = d.metadata.get('source', 'Unknown File')
        page = d.metadata.get('page', 'Unknown Page')
        h1 = d.metadata.get('Header 1', '')
        h2 = d.metadata.get('Header 2', '')
        h1 = h1 if h1 else ""
        h2 = h2 if h2 else ""
        context_header = f"{h1} > {h2}".strip(" > ")
        
        context_parts.append(f"--- SOURCE: {source} | Page: {page} | Section: {context_header} ---\n{d.page_content}")
    
    context_text = "\n\n".join(context_parts)

    # 4. Generate Answer
    answer_chain = build_answer_chain() 
    response = answer_chain.invoke({"input": question, "context": context_text})
    
    # Return BOTH answer and docs for the UI
    return {
        "answer": response.content,
        "source_documents": docs
    }