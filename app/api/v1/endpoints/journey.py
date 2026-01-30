from fastapi import APIRouter, Depends, HTTPException, status
from app.api.deps import get_current_user
from app.core.supabse_client import get_supabase_client
from app.models.user import User
from app.schemas.journey import JourneyResponse
import uuid

router = APIRouter()

@router.get("", response_model=JourneyResponse)
async def get_my_journey(
    current_user: User = Depends(get_current_user)
):
    """
    Fetch the current user's journey journal.
    """
    supabase = get_supabase_client()
    
    # Assuming 'journey' table has 'user_id' and 'journey_so_far'
    response = supabase.table("journey").select("journey_so_far").eq("user_id", str(current_user.id)).execute()
    
    if response.data and len(response.data) > 0:
        journey_text = response.data[0].get("journey_so_far", "")
        return JourneyResponse(journey_so_far=journey_text)
    
    # If no journey exists yet, return empty or initial state
    return JourneyResponse(journey_so_far="")
