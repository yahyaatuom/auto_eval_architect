import numpy as np
from sentence_transformers import SentenceTransformer

# Mock documents for testing
MOCK_DOCS = [
    "March 15th outage: Database connection pool exhausted at 14:23 UTC. Root cause: missing connection release.",
    "On-call engineer: Sarah Chen responded within 4 minutes.",
    "Resolution: Increased max_connections from 100 to 200. Completed at 14:47 UTC.",
    "Post-mortem: Add connection pooling monitoring.",
    "Jira ticket OPS-123: Assigned to Sarah Chen, resolved in 24 minutes."
]

class SimpleRetriever:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.documents = MOCK_DOCS
        self.embeddings = self.model.encode(self.documents)
    
    def search(self, query: str, top_k: int = 3):
        query_embedding = self.model.encode([query])
        similarities = np.dot(self.embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]