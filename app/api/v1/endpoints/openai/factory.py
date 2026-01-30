import openai
from app.api.v1.endpoints.openai.enums import AITask
from app.api.v1.endpoints.openai.provider import ProviderRegistry
from app.api.v1.endpoints.openai.strategies import ImageClassificationStrategy, DocumentDetectionStrategy, DocumentOCRStrategy, JourneyGenerationStrategy, JourneyUpdateStrategy
from app.core.config import settings

openai_client = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

class AIAnalysisFactory:

    def __init__(self, providers: ProviderRegistry):
        self.providers = providers

    def create(self, task: AITask):
        match task:

            case AITask.IMAGE_CLASSIFICATION:
                return ImageClassificationStrategy(
                    provider=self.providers.openai
                )
            
            case AITask.DOCUMENT_DETECTION:
                return DocumentDetectionStrategy(
                    provider=self.providers.openai
                )
            
            case AITask.DOCUMENT_OCR:
                return DocumentOCRStrategy(
                    provider=self.providers.openai
                )
            
            case AITask.JOURNEY_GENERATION:
                return JourneyGenerationStrategy(
                    provider=self.providers.openai
                )

            case AITask.JOURNEY_UPDATE:
                return JourneyUpdateStrategy(
                    provider=self.providers.openai
                )

        raise ValueError(f"Unsupported task: {task}")
