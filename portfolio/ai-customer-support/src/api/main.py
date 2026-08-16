from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.agent.rag_chain import answer_question
from src.api.schemas import AnswerResponse, QuestionRequest

# Create the FastAPI app
app = FastAPI(
    title="AI Customer Support Agent",
    description="RAG-powered customer support API",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Receive a question and return an answer with sources.
    """
    try:
        result = answer_question(request.question)

        return {
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except ValueError as e:
        # For expected validation-type errors
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        # Unexpected errors - don't expose internal details
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred while processing your request."
        )

@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    """
    return {
        "status": "ok",
        "service": "AI Customer Support Agent"
    }