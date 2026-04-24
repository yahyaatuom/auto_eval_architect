from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    query: str
    retrieved_docs: Annotated[List[str], operator.add]
    critique: str
    confidence_score: float
    iterations: int
    final_answer: str
    missing_information: List[str]