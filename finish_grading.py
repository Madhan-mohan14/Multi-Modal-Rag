import pandas as pd
import ast
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness  # Removed answer_relevancy
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

eval_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
eval_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

def finish_evaluation():
    print("📂 Loading saved data from 'pre_eval_backup.csv'...")
    try:
        df = pd.read_csv("pre_eval_backup.csv")
    except FileNotFoundError:
        print("❌ Error: 'pre_eval_backup.csv' not found.")
        return

    # Repair Data
    df['contexts'] = df['contexts'].apply(ast.literal_eval)
    
    # Hardcode Ground Truths (Safety)
    df['ground_truth'] = [
        "Real GDP growth is projected to improve gradually to 2 percent in 2024-25.",
        "QCB introduced measures to reduce banks' net short-term foreign liabilities and non-resident deposits.",
        "The target is to expand renewable energy capacity to 4 GW by 2030."
    ]

    dataset = Dataset.from_pandas(df)

    print("⚖️  Running Ragas Judge (Faithfulness Only)...")
    
    # Single metric = Fast & Stable
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness], 
        llm=eval_llm,
        embeddings=eval_embeddings,
        raise_exceptions=False
    )
    
    print("\n📊 Evaluation Results:")
    print(results)
    
    results.to_pandas().to_csv("evaluation_report.csv", index=False)
    print("✅ Final Report Saved: 'evaluation_report.csv'")

if __name__ == "__main__":
    finish_evaluation()