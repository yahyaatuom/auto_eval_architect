from models.state import AgentState

class CriticNode:
    def __call__(self, state: AgentState):
        query = state["query"]
        docs = state["retrieved_docs"]
        
        if not docs:
            return {
                "confidence_score": 0.0,
                "critique": "No documents found",
                "missing_information": ["any information"]
            }
        
        # Simple heuristic (no LLM needed for demo)
        doc_text = " ".join(docs).lower()
        
        # Check if answer contains names for "who" questions
        missing = []
        if "who" in query.lower():
            if not any(name in doc_text for name in ["sarah", "chen"]):
                missing.append("person name")
        
        if missing:
            confidence = 0.3
            critique = f"Missing: {', '.join(missing)}"
        else:
            confidence = 0.85
            critique = "Sufficient information"
        
        print(f"[Critic] Score: {confidence}")
        
        return {
            "confidence_score": confidence,
            "critique": critique,
            "missing_information": missing
        }