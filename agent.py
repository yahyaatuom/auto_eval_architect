"""Working multi-agent research system - Single file, zero import errors"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# ============ MOCK DATA ============
DOCUMENTS = [
    "March 15th outage: Database connection pool exhausted at 14:23 UTC. Root cause: missing connection release after query.",
    "On-call engineer: Sarah Chen (sarah.chen@company.com) responded within 4 minutes.",
    "Resolution: Increased max_connections from 100 to 200. Added connection leak detection. Completed at 14:47 UTC.",
    "Post-mortem action: Add connection pooling monitoring and alert at 80% capacity.",
    "Jira OPS-123: Assigned to Sarah Chen, priority P0, resolved in 24 minutes.",
    "Slack summary: 'Root cause was the app server redeploy without proper connection cleanup.'"
]

# ============ RETRIEVAL SYSTEM ============
class VectorRetriever:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embeddings = self.model.encode(documents)
    
    def search(self, query: str, top_k: int = 3) -> List[str]:
        query_emb = self.model.encode([query])
        similarities = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

# ============ AGENT NODES ============
class Researcher:
    def __init__(self, retriever: VectorRetriever):
        self.retriever = retriever
    
    def research(self, query: str, missing_info: List[str] = None) -> List[str]:
        if missing_info:
            enhanced_query = f"{query} Specifically need: {', '.join(missing_info)}"
        else:
            enhanced_query = query
        
        docs = self.retriever.search(enhanced_query)
        print(f"🔍 [Researcher] Found {len(docs)} documents")
        for i, doc in enumerate(docs, 1):
            print(f"   {i}. {doc[:80]}...")
        return docs

class Critic:
    def evaluate(self, query: str, documents: List[str]) -> Dict:
        if not documents:
            return {"confidence": 0.0, "missing": ["any information"], "reason": "No documents found"}
        
        doc_text = " ".join(documents).lower()
        query_lower = query.lower()
        
        missing = []
        
        # Check for missing information based on question type
        if "who" in query_lower and "sarah" not in doc_text:
            missing.append("person/name")
        if "what" in query_lower and "cause" not in doc_text and "root" not in doc_text:
            missing.append("root cause")
        if "when" in query_lower and "14:" not in doc_text and "march" not in doc_text:
            missing.append("timestamp")
        if "how" in query_lower and "resolution" not in doc_text:
            missing.append("resolution steps")
        
        if missing:
            confidence = 0.3
            reason = f"Missing: {', '.join(missing)}"
        else:
            confidence = 0.85
            reason = "Sufficient information found"
        
        print(f"📊 [Critic] Confidence: {confidence:.0%} - {reason}")
        return {"confidence": confidence, "missing": missing, "reason": reason}

class Reporter:
    def report(self, query: str, documents: List[str], confidence: float, iterations: int) -> str:
        answer = f"""
{'='*60}
ANSWER TO: {query}
{'='*60}

Based on {len(documents)} document(s) after {iterations} research iteration(s):

"""
        for i, doc in enumerate(documents, 1):
            answer += f"{i}. {doc}\n\n"
        
        answer += f"""
{'='*60}
CONFIDENCE: {confidence:.0%}
{'='*60}
"""
        
        if confidence < 0.7:
            answer += "\n⚠️ Consider asking a more specific question or rephrasing.\n"
        
        return answer

# ============ MAIN AGENT LOOP ============
def run_agent(query: str, max_iterations: int = 3):
    print(f"\n🚀 Starting research: {query}\n")
    
    # Initialize components
    retriever = VectorRetriever(DOCUMENTS)
    researcher = Researcher(retriever)
    critic = Critic()
    reporter = Reporter()
    
    # State
    documents = []
    missing_info = []
    confidence = 0.0
    
    # Main loop
    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration} ---")
        
        # Research
        docs = researcher.research(query, missing_info)
        documents.extend(docs)
        
        # Critic
        evaluation = critic.evaluate(query, documents)
        confidence = evaluation["confidence"]
        missing_info = evaluation["missing"]
        
        # Stop if confident enough
        if confidence > 0.7:
            print(f"\n✅ Confidence threshold reached. Stopping at iteration {iteration}.")
            break
    
    # Generate final answer
    answer = reporter.report(query, documents, confidence, iteration)
    return answer

# ============ RUN TESTS ============
if __name__ == "__main__":
    test_queries = [
        "Who was the on-call engineer during the outage?",
        "What caused the outage?",
        "How was the issue resolved?"
    ]
    
    for query in test_queries:
        result = run_agent(query)
        print(result)
        print("\n" + "🌸"*20 + "\n")
    
    # Complex query that needs multiple iterations
    print("\n" + "🔥"*30)
    print("TESTING COMPLEX QUERY (needs iteration)")
    print("🔥"*30)
    
    complex_query = "Who fixed the issue and what did they do?"
    result = run_agent(complex_query)
    print(result)