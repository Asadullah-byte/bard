from typing import Optional, Union, ClassVar
from pydantic import PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from pathlib import Path
from dotenv import load_dotenv
import os
import logging

load_dotenv(override=True)
_prompts_path = Path(__file__).parent / "prompts.yaml"
prompts = yaml.safe_load(_prompts_path.read_text(encoding="utf-8"))

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    PROJECT_NAME: str = os.getenv("PROJECT_NAME")
    PREFIX: str = os.getenv("PREFIX")
    API_VERSION: str = os.getenv("API_VERSION")
    API_V1_STR: str = f"/{os.getenv('PREFIX')}/{os.getenv('API_VERSION')}"

    # Database
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB")
    POSTGRES_PORT: int = os.getenv("POSTGRES_PORT")
    DATABASE_URL: Union[str, PostgresDsn] = os.getenv("DATABASE_URL")

    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY")
    ALGORITHM: str = os.getenv("ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = os.getenv(
        "ACCESS_TOKEN_EXPIRE_MINUTES")

    # OpenAI / LLM
    OPENAI_MODEL_NAME: str = "gpt-4o"
    OPENAI_VISION_MODEL_NAME: str = "gpt-4o"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    model_config = SettingsConfigDict(
        env_file=".env", case_sensitive=True, extra="ignore")
    SYSTEM_DESCRIBE_IMAGE_PROMPT: ClassVar[str] = prompts["image_analysis"]["SYSTEM_DESCRIBE_IMAGE_PROMPT"]
    USER_DESCRIBE_IMAGE_PROMPT: ClassVar[str] = prompts["image_analysis"]["USER_DESCRIBE_IMAGE_PROMPT"]
    SYSTEM_DAY_JOURNAL_PROMPT: ClassVar[str] = prompts["image_analysis"]["SYSTEM_DAY_JOURNAL_PROMPT"]
    USER_DAY_JOURNAL_PROMPT: ClassVar[str] = prompts["image_analysis"]["USER_DAY_JOURNAL_PROMPT"]


    SUPABASE_URL:str = os.environ.get(
        "SUPABASE_URL")
    SUPABASE_KEY:str = os.environ.get(
        "SUPABASE_KEY")
    EMBED_MODEL:str = os.environ.get(
        "EMBED_MODEL")
    DISTANCE_METRIC:str = os.environ.get(
        "DISTANCE_METRIC")
    SIMILARITY_THRESHOLD:float = os.environ.get(
        "SIMILARITY_THRESHOLD")
    
    OPENAI_API_KEY:str = os.environ.get(
        "OPENAI_API_KEY")

    # Celery Configuration
    CELERY_BROKER_URL: Optional[str] = os.getenv("CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: Optional[str] = os.getenv("CELERY_RESULT_BACKEND")
    CELERY_ACCEPT_CONTENT: Optional[list[str]] = [
        "json"
    ]

    @field_validator("CELERY_ACCEPT_CONTENT", mode="before")
    @classmethod
    def assemble_list_from_str(cls, v: Union[str, list[str]]) -> list[str]:
        if isinstance(v, str) and not v.strip().startswith("["):
            return [i.strip() for i in v.split(",")]
        return v

    CELERY_TASK_SERIALIZER: Optional[str] = os.getenv(
        "CELERY_TASK_SERIALIZER"
    )
    CELERY_RESULT_SERIALIZER: Optional[str] = os.getenv(
        "CELERY_RESULT_SERIALIZER"
    )
    CELERY_TIMEZONE: Optional[str] = os.getenv("CELERY_TIMEZONE")
    WORKER_TIMEOUT: Optional[int] = int(os.getenv("WORKER_TIMEOUT", "300"))
try:
    settings = Settings()
except Exception as e:
    raise Exception(f'ERROR IN config.py: {e}')
