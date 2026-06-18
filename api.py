from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from agent import TravelAgent
import uuid

app = FastAPI()
templates = Jinja2Templates(directory="templates")
agent = TravelAgent()

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/chat")
async def chat(body: ChatRequest):
    session_id = body.session_id or str(uuid.uuid4())
    try:
        result = agent.generate_trip(body.message, session_id=session_id)
        return JSONResponse({
            "reply": result["reply"],
            "plan": result.get("plan"),
            "session_id": session_id
        })
    except Exception as e:
        return JSONResponse({"reply": f"Error: {str(e)}", "session_id": session_id}, status_code=500)