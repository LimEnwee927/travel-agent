from sentence_transformers import SentenceTransformer

# Local CPU embedding model shared by long-term memory (agent.py) and RAG
# (rag.py). Groq's API hosts chat/audio models only - it has no embeddings
# endpoint - so embeddings run locally instead of through the LLM provider.
EMBEDDING_DIM = 384

_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    return _model.encode(text, normalize_embeddings=True).tolist()
