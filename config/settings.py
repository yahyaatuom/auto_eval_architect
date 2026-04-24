from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Hugging Face
    HF_TOKEN: str
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    GENERATION_MODEL: str = "Qwen/Qwen2.5-1.5B-Instruct"  # Start small
    
    # Vector DB
    PG_CONNECTION: str = "postgresql://user:pass@localhost:5432/vectordb"
    
    # Redis Cache
    REDIS_URL: str = "redis://localhost:6379"
    
    # Evaluation
    EVAL_MODEL: str = "meta-llama/Llama-3.2-3B-Instruct"
    
    class Config:
        env_file = ".env"

settings = Settings()