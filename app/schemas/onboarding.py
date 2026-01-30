from typing import List
from pydantic import BaseModel, Field


class OnboardingQuestionAnswer(BaseModel):
    """Schema for a single question-answer pair."""
    question: str = Field(..., description="The onboarding question")
    answer: str = Field(..., description="The user's answer to the question")


class OnboardingRequest(BaseModel):
    """Schema for the onboarding request body."""
    responses: List[OnboardingQuestionAnswer] = Field(
        ..., 
        description="Array of question-answer pairs",
        min_length=1
    )


class OnboardingResponseData(BaseModel):
    """Schema for the onboarding response data."""
    id: str
    user_id: str
    responses: List[OnboardingQuestionAnswer]
