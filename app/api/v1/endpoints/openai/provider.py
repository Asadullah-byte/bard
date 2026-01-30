from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from openai import AsyncOpenAI
from app.core.config import settings
from app.schemas.faces import ImageClassificationResponse


class AIProvider(ABC):

    @abstractmethod
    async def parse_json(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str | list[dict],
        response_format: ImageClassificationResponse
    ) -> dict:
        pass

class OpenAIProvider(AIProvider):

    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def parse_json(
            self,
            *,
            model: str,
            system_prompt: str,
            user_prompt: str | list[dict],
            response_format: dict) -> dict:
        prompt = [
            {"role": "system", "content": system_prompt},
        ]
        if user_prompt:
            prompt.append({"role": "user", "content": user_prompt})
        response = await self.client.responses.parse(
            model=model,
            input=prompt,
            text_format=response_format,
        )
        if not response.output_parsed:
            raise ValueError("Failed to parse response as JSON")
        return response.output_parsed


@dataclass(frozen=True)
class ProviderRegistry:
    openai: OpenAIProvider

    def get(self, name: str) -> AIProvider:
        if hasattr(self, name):
            return getattr(self, name)
        raise ValueError(f"Provider {name} not found in registry")

def init_providers(settings) -> ProviderRegistry:
    openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    openai_provider = OpenAIProvider(openai_client)


    return ProviderRegistry(
        openai=openai_provider,
    )


@lru_cache
def get_providers() -> ProviderRegistry:
    return init_providers(settings)
