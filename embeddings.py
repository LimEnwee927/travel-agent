import hashlib
import re
import numpy as np

# Lightweight embedding shared by long-term memory (agent.py) and RAG
# (rag.py): a normalized feature-hashed bag-of-words vector, not a
# transformer model. Render's free tier (512MB) can't fit torch +
# sentence-transformers alongside the rest of the app - this keeps the same
# FAISS + cosine-similarity architecture, just a far cheaper vector
# representation with no model download and near-zero import cost.
EMBEDDING_DIM = 256

_TOKEN_RE = re.compile(r"[a-z0-9]+")

def get_embedding(text: str):
    vec = np.zeros(EMBEDDING_DIM, dtype="float32")
    tokens = _TOKEN_RE.findall(text.lower())

    for token in tokens:
        digest = int(hashlib.md5(token.encode()).hexdigest(), 16)
        index = digest % EMBEDDING_DIM
        sign = 1.0 if (digest // EMBEDDING_DIM) % 2 == 0 else -1.0
        vec[index] += sign

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()
