from src.utils.logging import setup_logger

logger = setup_logger("api")

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
    allow_origins=["*"],        # In production, replace with specific domains (restricted to frontend domains)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Receive a question and return an answer with sources.
    """
    logger.info(f"Received question: {request.question[:80]}...")     # log only the first part

    try:
        result = answer_question(request.question)
        logger.info("Successfully generated answer")
        return {
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except ValueError as e:
        # For expected validation-type errors
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:  # noqa: BLE001
        # Unexpected errors - don't expose internal details
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="An internal error occurred while processing your request."
        )

@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    """
    logger.info("Health check requested")
    
    return {
        "status": "ok",
        "service": "AI Customer Support Agent"
    }