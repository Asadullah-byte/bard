from typing import List, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
import base64

import logging
from typing import Optional

from app.schemas.faces import ImageClassificationResponse, DocumentClassificationResponse, OCRResponse
from app.schemas.journey import JourneyResponse

logger = logging.getLogger(__name__)


@dataclass
class ImageInput:
    image: bytes
    platform: Optional[str]
    mime_type: str
    response_format: Union[ImageClassificationResponse, DocumentClassificationResponse, OCRResponse]


@dataclass
class MultiImageInput:
    images: List[bytes]
    mime_types: List[str]
    response_format: dict
    text: str = "Produce image comparison analysis:"


@dataclass
class JourneyInput:
    responses: List[dict]
    response_format: type[JourneyResponse]


@dataclass
class JourneyUpdateInput:
    current_journey: str
    new_content: str
    response_format: type[JourneyResponse]


class AIAnalysisStrategy(ABC):
    @abstractmethod
    async def execute(self, *,  input: Union[ImageInput, MultiImageInput, JourneyInput, JourneyUpdateInput],) -> dict:
        pass

class ImageClassificationStrategy(AIAnalysisStrategy):

    def __init__(self, provider):
        self.provider = provider

    async def execute(self, *, input: ImageInput) -> dict:
        image_base64 = base64.b64encode(input.image).decode('utf-8')
        
        user_content = [
            {
                "type": "input_text",
                "text": "Analyze this image and classify the event type and surrounding/location. Return the result in JSON format."
            },
            {
                "type": "input_image",
                "image_url": f"data:{input.mime_type};base64,{image_base64}"
            }
        ]
        
        return await self.provider.parse_json(
            model="gpt-4o",
            system_prompt="You are an expert image analyst. Classify the event happening in the image (e.g., marriage, show, concert, party, meeting, sports, casual gathering) and identify the location/surrounding (use actual location names like 'Eiffel Tower' if recognizable, otherwise describe as 'indoor office', 'outdoor park', etc., or 'unknown'). Return ONLY valid JSON.",
            user_prompt=user_content,
            response_format=input.response_format,
        )


class DocumentDetectionStrategy(AIAnalysisStrategy):

    def __init__(self, provider):
        self.provider = provider

    async def execute(self, *, input: ImageInput) -> dict:
        image_base64 = base64.b64encode(input.image).decode('utf-8')

        user_content = [
            {
                "type": "input_text",
                "text": "Determine if this image is a document (e.g., ID card, passport, driver's license, receipt, certificate, etc.). Return the result in JSON format."
            },
            {
                "type": "input_image",
                "image_url": f"data:{input.mime_type};base64,{image_base64}"
            }
        ]

        return await self.provider.parse_json(
            model="gpt-4o",
            system_prompt="You are an expert document classifier. Your goal is to determine if the provided image is a document or a standard photograph/scene. Return a JSON object with a single boolean field 'is_document'.",
            user_prompt=user_content,
            response_format=input.response_format,
        )


class DocumentOCRStrategy(AIAnalysisStrategy):

    def __init__(self, provider):
        self.provider = provider

    async def execute(self, *, input: ImageInput) -> dict:
        image_base64 = base64.b64encode(input.image).decode('utf-8')

        user_content = [
            {
                "type": "input_text",
                "text": "Extract all readable text from this document image. Return the result in JSON format."
            },
            {
                "type": "input_image",
                "image_url": f"data:{input.mime_type};base64,{image_base64}"
            }
        ]

        return await self.provider.parse_json(
            model="gpt-4o",
            system_prompt="You are a highly accurate OCR assistant. Extract all visible text from the document image and return it as a JSON object with a 'text' field containing the extracted string.",
            user_prompt=user_content,
            response_format=input.response_format,
        )


class JourneyGenerationStrategy(AIAnalysisStrategy):

    def __init__(self, provider):
        self.provider = provider

    async def execute(self, *, input: JourneyInput) -> dict:
        responses_str = "\n".join([f"Q: {r['question']}\nA: {r['answer']}" for r in input.responses])
        
        system_prompt = (
            "You are a master storyteller and biographer. Based on the user's onboarding responses, "
            "generate a beautiful Markdown journal entry describing their journey so far. "
            "Use a timeline effect with proper Markdown formatting (headings, bullet points, bold text). "
            "Make it feel inspiring and personal. Return the result as a JSON object with a 'journey_so_far' field."
        )
        
        user_prompt = f"Here are the onboarding responses:\n\n{responses_str}\n\nGenerate the journey journal."
        
        return await self.provider.parse_json(
            model="gpt-4o",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=input.response_format,
        )


class JourneyUpdateStrategy(AIAnalysisStrategy):

    def __init__(self, provider):
        self.provider = provider

    async def execute(self, *, input: JourneyUpdateInput) -> dict:
        system_prompt = (
            "You are a master storyteller managing a continuous personal journal. "
            "Your task is to take the existing journal content and weave in new events (from photos or documents) seamlessly into a timeline. "
            "STRICT RULES FOR STRUCTURE:\n"
            "1. Use the date provided with the new events as an H1 heading (e.g., '# October 25, 2023'). Group all events from the same date under this heading.\n"
            "2. Under the date heading, use H2 or H3 headings for specific events or images (e.g., '## A Wedding Celebration' or '### Important Document').\n"
            "3. Provide a vivid paragraph or explanation for each event/image, integrating it into the narrative.\n"
            "4. Maintain the voice and style of the existing journal. Use Markdown for all formatting.\n"
            "5. If the new content is a document, summarize its significance. If it's a photo/scene, describe it vividly.\n"
            "6. Ensure the full journal text is returned, including the new updates woven in correctly.\n"
            "Return the updated full journal text as a JSON object with a 'journey_so_far' field."
        )

        user_prompt = f"Current Journal:\n{input.current_journey}\n\nNew Events/Content with Dates:\n{input.new_content}\n\nUpdate the journal following the timeline structure."

        return await self.provider.parse_json(
            model="gpt-4o",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format=input.response_format,
        )
