try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

import numpy as np
import json
import os
import wikipediaapi
from bs4 import BeautifulSoup
import requests
from openai import OpenAI
from config import GROQ_API_KEY

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

DOCS_FILE = "rag_documents.json"
INDEX_FILE = "rag_index.faiss"

documents = []
index = None

def get_embedding(text):
    response = client.embeddings.create(
        model="nomic-embed-text-v1_5",
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

def save_rag():
    if not FAISS_AVAILABLE:
        return
    with open(DOCS_FILE, "w") as f:
        json.dump(documents, f, indent=2)
    faiss.write_index(index, INDEX_FILE)

def load_rag():
    global documents, index
    if not FAISS_AVAILABLE:
        print("⚠️ FAISS not available, RAG disabled.")
        return
    if os.path.exists(DOCS_FILE) and os.path.exists(INDEX_FILE):
        with open(DOCS_FILE, "r") as f:
            documents = json.load(f)
        index = faiss.read_index(INDEX_FILE)
        print(f"📚 Loaded {len(documents)} chunks from RAG store.")
    else:
        index = faiss.IndexFlatL2(768)
        print("📚 Fresh RAG store created.")

def retrieve(query: str, top_k: int = 4) -> str:
    if not FAISS_AVAILABLE or len(documents) == 0:
        return ""
    query_embedding = get_embedding(query)
    distances, indices = index.search(np.array([query_embedding]), top_k)
    results = []
    for i in indices[0]:
        if i < len(documents):
            doc = documents[i]
            results.append(f"[{doc['source']}]\n{doc['text']}")
    return "\n\n---\n\n".join(results)

def chunk_text(text, source, chunk_size=300, overlap=50):
    if not FAISS_AVAILABLE:
        return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append({"text": chunk, "source": source})
    return chunks

def add_chunks(chunks):
    global documents, index
    if not FAISS_AVAILABLE:
        return
    for chunk in chunks:
        embedding = get_embedding(chunk["text"])
        index.add(np.array([embedding]))
        documents.append(chunk)
    save_rag()

def ingest_wikipedia(topic: str):
    if not FAISS_AVAILABLE:
        return
    wiki = wikipediaapi.Wikipedia(language="en", user_agent="TravelAgent/1.0")
    page = wiki.page(topic)
    if not page.exists():
        return
    chunks = chunk_text(page.text, source=f"Wikipedia: {topic}")
    add_chunks(chunks)

def ingest_blog(url: str):
    if not FAISS_AVAILABLE:
        return
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        chunks = chunk_text(text, source=f"Blog: {url}")
        add_chunks(chunks)
    except Exception as e:
        print(f"❌ Failed to scrape {url}: {e}")

def ingest_text_file(filepath: str):
    if not FAISS_AVAILABLE:
        return
    with open(filepath, "r") as f:
        text = f.read()
    chunks = chunk_text(text, source=f"File: {filepath}")
    add_chunks(chunks)