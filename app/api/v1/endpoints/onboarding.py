import logging
from fastapi import APIRouter, HTTPException, status
from app.schemas.response import APIResponse
from app.schemas.onboarding import OnboardingRequest, OnboardingResponseData
from app.api.deps import SessionDep, CurrentUser
from app.models.onboarding_responses import OnboardingResponse
from app.models.journey import Journey
from app.api.v1.faces.helper import generate_journey_journal
from app.core.errors import InternalServerErrorException

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("", response_model=APIResponse[OnboardingResponseData], status_code=status.HTTP_201_CREATED)
async def submit_onboarding_responses(
    session: SessionDep,
    current_user: CurrentUser,
    request: OnboardingRequest
) -> OnboardingResponseData:

    try:
        responses_data = [
            {"question": item.question, "answer": item.answer}
            for item in request.responses
        ]
        logger.info(responses_data)
        onboarding_response = OnboardingResponse(
            user_id=current_user.id,
            responses=responses_data
        )
        
        session.add(onboarding_response)
        
        # Generate journey journal using OpenAI
        journey_so_far = await generate_journey_journal(responses_data)
        
        # Save to journey table
        journey_entry = Journey(
            user_id=current_user.id,
            journey_so_far=journey_so_far
        )
        session.add(journey_entry)
        
        current_user.is_onboarded = True
        
        await session.commit()
        await session.refresh(onboarding_response)
        
        response_data = OnboardingResponseData(
            id=str(onboarding_response.id),
            user_id=str(onboarding_response.user_id),
            responses=request.responses
        )
        
        return APIResponse.success_response(
            data=response_data,
            message="Onboarding responses saved successfully"
        )
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Failed to save onboarding responses: {str(e)}")
        await session.rollback()
        raise InternalServerErrorException(detail="Failed to save onboarding responses")