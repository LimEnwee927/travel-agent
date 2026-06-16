from openai import OpenAI
from prompts import SYSTEM_PROMPT
from tools import TOOLS, search_web, get_weather, search_hotels
from rag import load_rag, retrieve
import json
import os
import numpy as np

try:
    from config import GROQ_API_KEY
except ImportError:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

load_rag()

conversation_history = []
MEMORY_FILE = "long_term_memory.json"

# =========================
# MEMORY UTILITIES
# =========================

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
    scored = []

    for m in memories:
        try:
            scored.append(
                (cosine_similarity(query_embedding, m["embedding"]), m["text"])
            )
        except:
            continue

    scored.sort(reverse=True)
    return [text for _, text in scored[:top_k]]

def save_to_long_term_memory(text):
    memories = load_long_term_memory()
    embedding = get_embedding(text)
    memories.append({"text": text, "embedding": embedding})
    save_long_term_memory(memories)

# =========================
# TOOL SAFETY
# =========================

def safe_json_loads(x):
    try:
        return json.loads(x)
    except:
        return {}

def is_valid_tool_message(msg):
    if not msg.tool_calls:
        return True

    try:
        for tc in msg.tool_calls:
            if not tc.function.name or not tc.function.arguments:
                return False

            bad = ["<function", "</function", "function="]
            if any(b in str(tc.function.name) for b in bad):
                return False
            if any(b in str(tc.function.arguments) for b in bad):
                return False

        return True
    except:
        return False

# =========================
# PREFERENCE EXTRACTION
# =========================

def extract_preferences(user_message, assistant_response):
    prompt = f"""
Extract travel preferences as JSON array only.

Return [] if nothing.

User: {user_message}
Assistant: {assistant_response}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()

        data = json.loads(raw)
        return data if isinstance(data, list) else []

    except:
        return []

# =========================
# TOOL ROUTER
# =========================

def run_tool(tool_name, tool_args):
    print(f"🔧 Tool: {tool_name}({tool_args})")

    try:
        if tool_name == "search_web":
            return search_web(**tool_args)
        elif tool_name == "get_weather":
            return get_weather(**tool_args)
        elif tool_name == "search_hotels":
            return search_hotels(**tool_args)
        return "Unknown tool"
    except Exception as e:
        return f"Tool error: {str(e)}"

# =========================
# AGENT
# =========================

class TravelAgent:

    def generate_trip(self, user_request):
        global conversation_history

        # 1. Long-term memory
        memories = search_long_term_memory(user_request)
        memory_context = ""

        if memories:
            memory_context = "User preferences:\n" + "\n".join(f"- {m}" for m in memories) + "\n\n"

        # 2. RAG
        rag_context = retrieve(user_request)
        rag_section = ""

        if rag_context:
            rag_section = f"Knowledge base:\n{rag_context}\n\n"

        # 3. Build messages
        system = SYSTEM_PROMPT + "\n\n" + memory_context + rag_section

        messages = [{"role": "system", "content": system}]
        messages += conversation_history
        messages.append({"role": "user", "content": user_request})

        # =========================
        # 4. SAFE AGENT LOOP
        # =========================

        MAX_ROUNDS = 3
        tool_count = 0

        for _ in range(MAX_ROUNDS):

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                parallel_tool_calls=False
            )

            msg = response.choices[0].message

            # CASE 1: final answer
            if not msg.tool_calls:
                answer = msg.content
                break

            # CASE 2: invalid tool format → retry safely
            if not is_valid_tool_message(msg):
                messages.append({
                    "role": "system",
                    "content": "Invalid tool format detected. Use proper tools only."
                })
                continue

            # CASE 3: tool limit
            tool_count += 1
            if tool_count > 3:
                messages.append({
                    "role": "system",
                    "content": "Stop using tools. Provide final answer now."
                })
                continue

            # CASE 4: execute tools
            assistant_msg = {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in msg.tool_calls
                ]
            }

            messages.append(assistant_msg)

            for tool_call in msg.tool_calls:
                args = safe_json_loads(tool_call.function.arguments)
                result = run_tool(tool_call.function.name, args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })

        else:
            answer = "Failed to generate trip due to tool loop."

        # =========================
        # 5. SHORT-TERM MEMORY
        # =========================

        conversation_history.append({"role": "user", "content": user_request})
        conversation_history.append({"role": "assistant", "content": answer})

        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

        # =========================
        # 6. LONG-TERM MEMORY
        # =========================

        for pref in extract_preferences(user_request, answer):
            try:
                save_to_long_term_memory(pref)
            except:
                pass

        return answer