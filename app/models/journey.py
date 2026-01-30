import uuid
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import UUID, ForeignKey, DateTime, Text
from datetime import datetime
from app.db.base import Base


class Journey(Base):
    __tablename__ = "journey"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False
    )
    journey_so_far: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now)
