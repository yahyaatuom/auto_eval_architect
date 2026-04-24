from models.state import AgentState

class ReporterNode:
    def __call__(self, state: AgentState):
        query = state["query"]
        docs = state["retrieved_docs"]
        confidence = state["confidence_score"]
        
        answer = f"Answer to: {query}\n\n"
        for i, doc in enumerate(docs, 1):
            answer += f"{i}. {doc}\n\n"
        answer += f"Confidence: {confidence:.0%}"
        
        return {"final_answer": answer}