from typing import Any

from pydantic import BaseModel, Field


# ======================
# Class: Question Request
# ======================
class QuestionRequest(BaseModel):
    """
    Schema for the incoming question.
    """
    question: str = Field(..., min_length=1, max_length=750, description="User's question")

# ======================
# Class: Source
# ======================
class Source(BaseModel):
    """
    Schema for a single source document.
    """
    content_preview: str
    metadata: dict[str, Any]

# ======================
# Class: Answer Response
# ======================
class AnswerResponse(BaseModel):
    """
    Schema for the API response.
    """
    answer: str
    sources: list[Source]