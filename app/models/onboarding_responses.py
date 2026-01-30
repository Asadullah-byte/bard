import uuid
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import UUID, ForeignKey, DateTime
from datetime import datetime
from sqlalchemy.dialects.postgresql import JSONB
from app.db.base import Base


class OnboardingResponse(Base):
    __tablename__ = "onboarding_responses"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    responses: Mapped[list] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now)