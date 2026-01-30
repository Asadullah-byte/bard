import uuid
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import UUID, String, ForeignKey
from pgvector.sqlalchemy import Vector
from app.db.base import Base

class Face(Base):
    __tablename__ = "face_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String, nullable=False)

    embedding: Mapped[Vector] = mapped_column(Vector(128), nullable=True)
    
    face_owner: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("images.id"),nullable=False
    )
