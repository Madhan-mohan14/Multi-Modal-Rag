import os
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from vector_store_handler import get_existing_retriever
from chain_handler import run_rag_chain
from ragas.run_config import RunConfig

# --- CONFIGURATION ---
# 1. Setup Models (Using the verified working model)
eval_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
eval_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 2. Define Test Set
questions = [
    "What is the projected Real GDP growth for 2025?",
    "What measures did QCB introduce regarding foreign liabilities?",
    "What is the target for renewable energy capacity by 2030?",
]

ground_truths = [
    "Real GDP growth is projected to improve gradually to 2 percent in 2024-25.",
    "What was the objective of the recent QCB measures regarding foreign liabilities?",
    "The target is to expand renewable energy capacity to 4 GW by 2030.",
]

def safe_extract_text(response):
    """
    Safety function to ensure we NEVER get a Dictionary or Document object.
    Always returns a clean string.
    """
    if isinstance(response, str):
        return response
    elif isinstance(response, dict):
        # Common LangChain output keys
        return response.get('answer') or response.get('result') or response.get('output') or str(response)
    elif hasattr(response, 'content'):
        # LangChain AIMessage object
        return response.content
    else:
        return str(response)

def run_evaluation():
    print("🚀 Starting Evaluation Suite...")
    
    # 3. Initialize Retrieval
    retriever = get_existing_retriever()
    answers = []
    contexts = []
    
    # 4. Loop through questions
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Asking: {q}")
        
        try:
            # A. Get Answer
            raw_response = run_rag_chain(q, [], retriever)
            clean_answer = safe_extract_text(raw_response)
            answers.append(clean_answer)
            
            # B. Get Context (Re-retrieve to be sure)
            docs = retriever.invoke(q)
            # FORCE CONVERSION TO STRING LIST - no Document objects allowed
            clean_context = [str(doc.page_content) for doc in docs] 
            contexts.append(clean_context)
            
        except Exception as e:
            print(f"❌ Error processing question '{q}': {e}")
            answers.append("Error generating answer")
            contexts.append(["No context retrieved"])

    # 5. SAFETY SAVE (Save before Eval)
    # If the code crashes after this, you don't lose your API usage.
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    
    print("\n💾 Saving intermediate results to 'pre_eval_backup.csv'...")
    df_backup = pd.DataFrame(data)
    df_backup.to_csv("pre_eval_backup.csv", index=False)
    
    # 6. Type Check Verification
    print("\n🔍 Verifying Data Types...")
    print(f" - Answer type: {type(answers[0])} (Must be <class 'str'>)")
    print(f" - Context type: {type(contexts[0])} (Must be <class 'list'>)")
    print(f" - Context content: {type(contexts[0][0])} (Must be <class 'str'>)")
    
    if not isinstance(answers[0], str) or not isinstance(contexts[0][0], str):
        print("⛔ CRITICAL STOP: Data contains non-string objects. Check 'pre_eval_backup.csv'.")
        return

    # 7. Create Dataset & Evaluate
    print("\n⚖️  Running Ragas Judge...")
    dataset = Dataset.from_dict(data)


    try:
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=eval_llm,
            embeddings=eval_embeddings,
            run_config=RunConfig(max_workers=1, timeout=60), 
            raise_exceptions=False
        )
        print("\n📊 Evaluation Results:")
        print(results)
        
        final_df = results.to_pandas()
        final_df.to_csv("evaluation_report.csv", index=False)
        print("✅ Success! Saved to 'evaluation_report.csv'")
        
    except Exception as e:
        print(f"⚠️ Ragas Grading Error: {e}")
        print("Your generated answers are safe in 'pre_eval_backup.csv'.")

if __name__ == "__main__":
    run_evaluation()