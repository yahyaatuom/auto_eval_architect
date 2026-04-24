from models.state import AgentState
from utils.retriever import SimpleRetriever

class ResearcherNode:
    def __init__(self):
        self.retriever = SimpleRetriever()
    
    def __call__(self, state: AgentState):
        query = state["query"]
        
        if state.get("missing_information"):
            enhanced_query = f"{query} Need: {', '.join(state['missing_information'])}"
        else:
            enhanced_query = query
        
        docs = self.retriever.search(enhanced_query)
        print(f"[Researcher] Found {len(docs)} docs")
        
        return {
            "retrieved_docs": docs,
            "iterations": state.get("iterations", 0) + 1
        }