from pydantic import BaseModel, Field


class JourneyResponse(BaseModel):
    """Schema for the generated journey journal."""
    journey_so_far: str = Field(..., description="The generated Markdown journal entry")
