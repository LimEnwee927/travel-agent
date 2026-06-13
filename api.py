from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from agent import TravelAgent
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")
agent = TravelAgent()

class ChatRequest(BaseModel):
    message: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/chat")
async def chat(body: ChatRequest):
    try:
        answer = agent.generate_trip(body.message)
        return JSONResponse({"reply": answer})
    except Exception as e:
        return JSONResponse({"reply": f"Error: {str(e)}"}, status_code=500)