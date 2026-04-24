from models.state import AgentState
from models.researcher import ResearcherNode
from models.critic import CriticNode
from models.reporter import ReporterNode

def run_agent(query: str, max_iterations: int = 3):
    state = AgentState(
        query=query,
        retrieved_docs=[],
        critique="",
        confidence_score=0.0,
        iterations=0,
        final_answer="",
        missing_information=[]
    )
    
    researcher = ResearcherNode()
    critic = CriticNode()
    reporter = ReporterNode()
    
    for _ in range(max_iterations):
        state.update(researcher(state))
        state.update(critic(state))
        
        # Stop if confidence is high
        if state["confidence_score"] > 0.7:
            break
    
    state.update(reporter(state))
    return state