from openai import OpenAI
from prompts import SYSTEM_PROMPT, PLANNER_PROMPT
from tools import TOOLS, search_web, get_weather, search_hotels
from rag import load_rag, multi_hop_retrieve
from embeddings import get_embedding
import json
import os
import re
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

# Short-term memory: one conversation history per session_id.
conversation_histories = {}
DEFAULT_SESSION = "default"
MEMORY_FILE = "long_term_memory.json"

# =========================
# MEMORY UTILITIES
# =========================
# Long-term memory is stored on disk as {session_id: [{"text", "embedding"}, ...]}
# so preferences persist across separate conversations for the same session_id,
# without leaking between different users/sessions sharing one server process.

def _load_all_long_term_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
        # Backwards-compat: older format was a flat list (single shared session).
        if isinstance(data, list):
            return {DEFAULT_SESSION: data}
        return data
    return {}

def _save_all_long_term_memory(data):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_long_term_memory(session_id=DEFAULT_SESSION):
    return _load_all_long_term_memory().get(session_id, [])

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_long_term_memory(session_id, query, top_k=3):
    memories = load_long_term_memory(session_id)
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

def save_to_long_term_memory(session_id, text):
    all_memories = _load_all_long_term_memory()
    session_memories = all_memories.get(session_id, [])
    embedding = get_embedding(text)
    session_memories.append({"text": text, "embedding": embedding})
    all_memories[session_id] = session_memories
    _save_all_long_term_memory(all_memories)

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

# Some models occasionally narrate a tool call as plain text instead of using
# the real function-calling mechanism (e.g. `search_web {"query": "..."}`).
# That text would otherwise be accepted as the final answer, so catch it and
# force a retry.
_FAKE_TOOL_CALL_RE = re.compile(
    r"\b(" + "|".join(t["function"]["name"] for t in TOOLS) + r")\b\s*\(?\s*\{"
)

def looks_like_fake_tool_call(content):
    return bool(content) and bool(_FAKE_TOOL_CALL_RE.search(content))

# Backstop for models that stall by announcing intent ("First, I will search
# for...") instead of actually calling a tool or finishing the itinerary.
_NARRATION_RE = re.compile(
    r"\b(I will|I'll|let'?s start|first, i|next, i|i need to|i'm going to)\b",
    re.IGNORECASE
)

def looks_like_stalling(content, tool_count):
    if tool_count > 0 or not content:
        return False
    if re.search(r"\bmorning\b", content, re.IGNORECASE):
        return False
    return bool(_NARRATION_RE.search(content))

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

    def _make_plan(self, user_request, memory_context, conversation_history):
        """Plan-and-Execute style planning step, run before any tool calls.
        Returns ("clarify", question) if the request is too ambiguous to plan
        (e.g. no destination city given), or ("plan", [sub-tasks]) otherwise.
        Falls back to ("plan", []) if the planner call fails - the main loop
        still works without an explicit plan."""
        planner_messages = [{"role": "system", "content": PLANNER_PROMPT}]
        planner_messages += conversation_history[-4:]
        planner_messages.append({
            "role": "user",
            "content": f"{memory_context}User request: {user_request}"
        })

        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=planner_messages,
                temperature=0
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)

            if isinstance(data, dict) and data.get("clarify"):
                return "clarify", str(data["clarify"])
            if isinstance(data, dict) and isinstance(data.get("plan"), list):
                return "plan", [str(s) for s in data["plan"]][:6]
        except Exception:
            pass

        return "plan", []

    def generate_trip(self, user_request, session_id=DEFAULT_SESSION):
        conversation_history = conversation_histories.setdefault(session_id, [])

        # 1. Long-term memory (per session, persisted across conversations)
        memories = search_long_term_memory(session_id, user_request)
        memory_context = ""

        if memories:
            memory_context = "User preferences:\n" + "\n".join(f"- {m}" for m in memories) + "\n\n"

        # 2. Planning step: decide whether to clarify or to plan sub-tasks
        kind, plan_result = self._make_plan(user_request, memory_context, conversation_history)

        if kind == "clarify":
            answer = plan_result
            conversation_history.append({"role": "user", "content": user_request})
            conversation_history.append({"role": "assistant", "content": answer})
            if len(conversation_history) > 20:
                conversation_histories[session_id] = conversation_history[-20:]
            return {"reply": answer, "plan": None}

        plan = plan_result
        plan_section = ""
        if plan:
            plan_section = "Plan:\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan)) + "\n\n"

        # 3. Multi-hop RAG
        rag_context = multi_hop_retrieve(user_request)
        rag_section = ""

        if rag_context:
            rag_section = f"Knowledge base:\n{rag_context}\n\n"

        # 4. Build messages
        system = SYSTEM_PROMPT + "\n\n" + memory_context + plan_section + rag_section

        messages = [{"role": "system", "content": system}]
        messages += conversation_history
        messages.append({"role": "user", "content": user_request})

        # =========================
        # 5. SAFE AGENT LOOP
        # =========================

        MAX_ROUNDS = 5
        tool_count = 0

        for _ in range(MAX_ROUNDS):

            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    parallel_tool_calls=False
                )
            except Exception:
                # Groq occasionally rejects a malformed tool-call generation
                # outright (HTTP 400) before it ever reaches our own checks.
                messages.append({
                    "role": "system",
                    "content": "Your last attempt could not be processed. Do not call any "
                                "tools this turn; give a direct, complete answer using only "
                                "the information already available."
                })
                continue

            msg = response.choices[0].message

            # CASE 1: final answer (but watch for a narrated fake tool call,
            # or stalling text that only announces intent without acting)
            if not msg.tool_calls:
                if looks_like_fake_tool_call(msg.content):
                    messages.append({
                        "role": "system",
                        "content": "Do not write tool calls as plain text. Either invoke a tool "
                                    "through the function-calling mechanism, or give the final "
                                    "itinerary directly without mentioning tool names."
                    })
                    continue
                if looks_like_stalling(msg.content, tool_count):
                    messages.append({
                        "role": "system",
                        "content": "Stop announcing intentions. Call the relevant tool right now, "
                                    "or if you already have enough information, give the complete "
                                    "final itinerary immediately."
                    })
                    continue
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
        # 6. SHORT-TERM MEMORY
        # =========================

        conversation_history.append({"role": "user", "content": user_request})
        conversation_history.append({"role": "assistant", "content": answer})

        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]
        conversation_histories[session_id] = conversation_history

        # =========================
        # 7. LONG-TERM MEMORY
        # =========================

        for pref in extract_preferences(user_request, answer):
            try:
                save_to_long_term_memory(session_id, pref)
            except:
                pass

        return {"reply": answer, "plan": plan}