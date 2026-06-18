# AI Travel Planning Agent

A conversational agent that plans a personalized 2-day trip to any city. It
decides for itself when to search the web, check the weather, or look up
hotels; remembers your preferences across sessions; and breaks each request
into a short plan before acting.

## Architecture

```
Browser/UI  ──┐
              ├─▶ FastAPI (api.py) ──▶ TravelAgent (agent.py)
REST client ──┘                            │
                                            ├─ 1. Long-term memory lookup (per session_id)
                                            ├─ 2. Planning step  ──▶ Groq (llama-3.3-70b)
                                            │     → {"clarify": "..."}  or  {"plan": [...]}
                                            ├─ 3. Multi-hop RAG (rag.py + FAISS)
                                            ├─ 4. Tool-calling loop ──▶ Groq (llama-3.3-70b)
                                            │     ├─ search_web   (Serper.dev)
                                            │     ├─ get_weather  (OpenWeatherMap)
                                            │     └─ search_hotels (Serper.dev)
                                            └─ 5. Write back short-term + long-term memory
```

Groq (OpenAI-compatible API, `llama-3.3-70b-versatile`) is the only LLM
provider; it's fast and free-tier friendly. It has no embeddings endpoint, so
embeddings (for both long-term memory and RAG) run locally via
`embeddings.py` - a dependency-free hashing-trick bag-of-words vector
(no torch/transformers), chosen to fit comfortably inside Render's free-tier
512MB memory limit. The FAISS + cosine-similarity architecture around it is
unchanged; only the vector representation is cheaper.

## Why these tools

| Tool | Provider | Purpose |
|---|---|---|
| `search_web` | Serper.dev (Google search) | Attractions, food, neighborhood tips - no single "attractions DB" covers every city, so live web search generalizes to any destination. |
| `get_weather` | OpenWeatherMap | Lets the agent favor indoor/outdoor activities and mention what to pack. |
| `search_hotels` | Serper.dev | Budget-aware accommodation suggestions (web search rather than a real booking API, since no hotel-inventory API key was available). |

The LLM decides which of these to call, in what order, and how many times
(up to 3 tool calls per request) via OpenAI-style function calling - nothing
is hardcoded.

## Planning & reasoning

Before any tool is called, `TravelAgent._make_plan()` sends the request to a
dedicated planner prompt (`PLANNER_PROMPT` in `prompts.py`) that returns one
of two things:

- `{"clarify": "<question>"}` if an essential detail (most importantly the
  destination city) is missing or ambiguous - the agent asks the user
  directly instead of guessing, and no tools are called that turn.
- `{"plan": ["sub-task 1", "sub-task 2", ...]}` - an ordered breakdown (e.g.
  check weather → research food → find hotels → draft itinerary) that's
  injected into the system prompt to guide the tool-calling loop that follows.

This is a lightweight Plan-and-Execute pattern: planning and execution are
separate LLM calls, and the plan is returned in the API response (`plan`
field) so it's visible, not just an internal implementation detail.

The tool-calling loop itself is also hardened against two real Llama/Groq
failure modes observed during testing: the model occasionally narrates a
tool call as plain text instead of using the function-calling mechanism, or
stalls by describing what it's about to do without doing it. Both are
detected and the model is forced to retry rather than returning a non-answer.

## Multi-hop RAG

`rag.multi_hop_retrieve()` (used in place of a single `retrieve()` call) does
two retrieval passes against the FAISS index of ingested Wikipedia/blog
content:

1. **Hop 1** - retrieve top-k chunks for the raw user request.
2. **Hop 2** - ask the LLM for one specific follow-up query based on what hop
   1 turned up (e.g. a neighborhood or dish it mentioned), then retrieve
   again with that more targeted query.

The two result sets are concatenated (deduplicated if identical), giving the
itinerary access to more specific detail than a single generic search would.

Content is ingested via `ingest.py` helpers: `ingest_wikipedia(topic)`,
`ingest_blog(url)`, `ingest_text_file(path)`. The FAISS index and chunk store
persist to `rag_index.faiss` / `rag_documents.json`.

## Memory design

**Short-term (conversation context):** `conversation_histories` in
`agent.py` keeps one message list per `session_id`, capped at the last 20
messages. It lives in process memory, so it resets on server restart - an
acceptable tradeoff for a single-instance demo deployment.

**Long-term (cross-session preferences):** after every reply,
`extract_preferences()` asks the LLM to pull out durable preferences (e.g.
"loves street food", "budget hotels") from the exchange. Each is embedded
locally and appended to `long_term_memory.json`, structured as
`{session_id: [{"text", "embedding"}, ...]}`. On the next request,
`search_long_term_memory()` does a cosine-similarity lookup over that
session's stored preferences and injects the closest matches into the system
prompt - this is what lets "now plan a trip to Tokyo instead" implicitly
carry over "street food, budget hotels" without the user repeating it.

Memory is partitioned by `session_id` (generated client-side and persisted
in `localStorage` for the web UI) rather than a single global store, so
concurrent users on the same deployed instance don't see each other's
conversations or preferences. This is a lightweight, file-based vector store
rather than a managed vector DB (Pinecone/standalone FAISS service) - sufficient
for this scope, but the same interface (`search_long_term_memory` /
`save_to_long_term_memory`) would swap in cleanly for a production vector DB.

## Setup guide

```bash
git clone https://github.com/LimEnwee927/travel-agent
cd travel-agent
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Set the following environment variables (or create `config.py` locally with
the same names - it's gitignored):

```
GROQ_API_KEY=...         # https://console.groq.com
SERPER_API_KEY=...       # https://serper.dev
OPENWEATHER_API_KEY=...  # https://openweathermap.org/api
```

Run the web UI + API:

```bash
uvicorn api:app --reload
# open http://localhost:8000
```

Or use the CLI:

```bash
python app.py
```

## REST API

`POST /chat`

```json
// request
{ "message": "plan a 2 day trip to Singapore", "session_id": "optional-uuid" }

// response
{
  "reply": "Day 1:\nMorning: ...",
  "plan": ["research top attractions", "find hotels", "..."],
  "session_id": "uuid-to-reuse-on-the-next-call"
}
```

If `session_id` is omitted, a new one is generated and returned - reuse it on
subsequent calls to keep both short-term context and long-term preferences.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "plan a 2 day trip to Tokyo"}'
```

## Deployment (Render)

- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn api:app --host 0.0.0.0 --port $PORT`
- Environment variables: `GROQ_API_KEY`, `SERPER_API_KEY`, `OPENWEATHER_API_KEY`
  (all three are required - without `SERPER_API_KEY`/`OPENWEATHER_API_KEY` two
  of the three tools will fail and degrade every itinerary)

`config.py` is gitignored on purpose (it held real keys locally); the code
falls back to `os.environ` when it's absent, which is what Render's build
sees.

## Known limitations / not implemented

- No explicit budget or pet-friendly constraint handling beyond what
  naturally surfaces through long-term preference memory.
- No engineered multilingual support (the base model can respond in other
  languages if asked, but there's no language detection/routing).
- Sessions are identified by a client-generated `session_id`, not an
  authenticated user account - sufficient for this single-tenant demo, not
  for a real multi-tenant product.
- Long-term memory is a flat JSON file with linear cosine-similarity search;
  fine at this scale, would need a real vector DB (Pinecone, hosted FAISS) to
  scale past a handful of sessions.

## Flowchart

```
                         +--------------------+
                         |      User          |
                         +---------+----------+
                                   |
                                   v
                      +--------------------------+
                      |  TravelAgent.generate()  |
                      +------------+-------------+
                                   |
          --------------------------------------------------
          |                        |                       |
          |                        |                       |
          v                        v                       v
+-------------------+     +-------------------+    +----------------------+
| Long-term Memory  |     |   RAG Retrieval   |    | Conversation History |
| (Vector Search)   |     | (Knowledge Base)  |    |  Short-term Memory   |
+---------+---------+     +---------+---------+    +----------+-----------+
          |                         |                         |
          |                         |                         |
          ----------- Merge Context --------------------------
                                   |
                                   v
                      +--------------------------+
                      |     System Prompt        |
                      | + Memory + RAG Context   |
                      +------------+-------------+
                                   |
                                   v
                      +--------------------------+
                      |      LLM (Groq API)      |
                      | llama-3.3-70b-versatile  |
                      +------------+-------------+
                                   |
                     Tool Calls? ---+---- No
                          |
                         Yes
                          |
                          v
                +------------------------+
                |   Tool Router          |
                +-----------+------------+
                            |
        ----------------------------------------------
        |                    |                       |
        |                    |                       |
        v                    v                       v
+----------------+   +----------------+    +----------------+
| search_web()   |   | get_weather()  |    | search_hotels()|
+----------------+   +----------------+    +----------------+
                            |
                            v
                  Tool Results Returned
                            |
                            v
                      +----------------+
                      |      LLM       |
                      +-------+--------+
                              |
                              v
                      Final Travel Plan
                              |
            ------------------------------------
            |                                  |
            v                                  v
+---------------------------+      +---------------------------+
| Conversation History      |      | Preference Extraction     |
| (Last 20 messages)        |      | (LLM JSON Extraction)     |
+-------------+-------------+      +-------------+-------------+
              |                                  |
              |                                  |
              v                                  v
      Short-term Memory              Long-term Memory Storage
                                         (Embedding Model)
```
