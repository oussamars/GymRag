#why we use pydantic:
# You define the data schema once, and FastAPI uses that single definition for parsing, validation, type conversion, and documentation.

from fastapi import FastAPI
from pydantic import BaseModel
from GymRag.rag_system import GymRAG
from pathlib import Path
from fastapi import HTTPException



class ChatRequest(BaseModel):
    message: str
    session_id: str

app = FastAPI()

base_dir = Path(__file__).resolve().parent / "GymRag"


@app.get("/health")
async def health():
    return {"status": "ok"}



sessions = {}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if request.session_id not in sessions:
        sessions[request.session_id] = GymRAG()
        sessions[request.session_id].index_document(str(base_dir / "gym_faq.txt"))
        sessions[request.session_id].index_document(str(base_dir / "gym_pricing.txt"))
    
    rag = sessions[request.session_id]
    
    try:
        answer = rag.chat(request.message)
        return {"answer": answer}
    except Exception:
        raise HTTPException(status_code=500, detail="Something went wrong")
