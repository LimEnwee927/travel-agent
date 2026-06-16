from openai import OpenAI
from prompts import SYSTEM_PROMPT
from tools import TOOLS, search_web, get_weather, search_hotels
from rag import load_rag, retrieve          # ← ADD THIS
import json
import os
import numpy as np

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

load_rag()                                  # ← ADD THIS (loads index on startup)

conversation_history = []
MEMORY_FILE = "long_term_memory.json"

def load_long_term_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_long_term_memory(memories):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memories, f, indent=2)

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_embedding(text):
    response = client.embeddings.create(
        model="nomic-embed-text-v1_5",
        input=text
    )
    return response.data[0].embedding

def search_long_term_memory(query, top_k=3):
    memories = load_long_term_memory()
    if not memories:
        return []
    query_embedding = get_embedding(query)
    scored = [(cosine_similarity(query_embedding, m["embedding"]), m["text"]) for m in memories]
    scored.sort(reverse=True)
    return [text for _, text in scored[:top_k]]

def save_to_long_term_memory(text):
    memories = load_long_term_memory()
    embedding = get_embedding(text)
    memories.append({"text": text, "embedding": embedding})
    save_long_term_memory(memories)

def extract_preferences(user_message, assistant_response):
    extraction_prompt = f"""
From this conversation, extract any user travel preferences, dislikes, or personal facts worth remembering.
Return ONLY a JSON array of short strings. If nothing worth saving, return [].

User said: {user_message}
Assistant replied: {assistant_response}

Example output: ["prefers budget hotels", "vegetarian", "likes beach destinations"]
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": extraction_prompt}],
        temperature=0
    )
    raw = response.choices[0].message.content.strip().replace("```json", "").replace("```", "").strip()
    try:
        preferences = json.loads(raw)
        return preferences if isinstance(preferences, list) else []
    except:
        return []

def run_tool(tool_name, tool_args):
    print(f"  🔧 Using tool: {tool_name}({tool_args})")
    if tool_name == "search_web":
        return search_web(**tool_args)
    elif tool_name == "get_weather":
        return get_weather(**tool_args)
    elif tool_name == "search_hotels":
        return search_hotels(**tool_args)
    return "Tool not found."


class TravelAgent:
    def generate_trip(self, user_request):
        global conversation_history

        # 1. Long-term memory
        relevant_memories = search_long_term_memory(user_request)
        memory_context = ""
        if relevant_memories:
            memory_context = "User preferences from past sessions:\n" + "\n".join(f"- {m}" for m in relevant_memories) + "\n\n"

        # 2. RAG retrieval ← NEW
        rag_context = retrieve(user_request)
        rag_section = ""
        if rag_context:
            rag_section = f"Relevant travel information from knowledge base:\n{rag_context}\n\n"

        # 3. Build messages
        system = SYSTEM_PROMPT + "\n\n" + memory_context + rag_section
        messages = [{"role": "system", "content": system}]
        messages += conversation_history
        messages.append({"role": "user", "content": user_request})



        # 4. Agentic loop
        while True:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                parallel_tool_calls=False
            )

            
            msg = response.choices[0].message
            assistant_msg = {"role": "assistant",  "content": msg.content or "", "tool_calls": [
    {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.function.name,
            "arguments": tc.function.arguments
        }
    } for tc in msg.tool_calls
]}
            if not msg.tool_calls:
                answer = msg.content
                break
            messages.append(assistant_msg)
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                tool_result = run_tool(tool_name, tool_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result
                })

        # 5. Short-term memory
        conversation_history.append({"role": "user", "content": user_request})
        conversation_history.append({"role": "assistant", "content": answer})
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

        # 6. Save preferences
        for pref in extract_preferences(user_request, answer):
            save_to_long_term_memory(pref)

        return answer