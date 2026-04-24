import sys
import os

# Add parent directory to path so we can import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.graph import run_agent

if __name__ == "__main__":
    queries = [
        "Who was the on-call engineer?",
        "What caused the outage?"
    ]
    
    for query in queries:
        print("\n" + "="*50)
        print(f"Query: {query}")
        result = run_agent(query)
        print(result["final_answer"])
        print(f"Iterations: {result['iterations']}")