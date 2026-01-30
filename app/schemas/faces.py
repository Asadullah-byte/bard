from pydantic import BaseModel, Field
import uuid
from typing import List


class ImageClassificationResponse(BaseModel):
    event: str = Field(
        ...,
        description="Type of event happening in the image (e.g., marriage, show, concert, party, meeting, sports, casual gathering, or unknown)"
    )
    surrounding: str = Field(
        ...,
        description="The location or surrounding where the image was taken. Use actual IRL location name if identifiable (e.g., 'Eiffel Tower', 'Central Park'), otherwise use 'unknown' or descriptive location (e.g., 'indoor office', 'outdoor park', 'beach')"
    )


class ImageClassificationRequest(BaseModel):
    image_data: bytes = Field(..., description="The image data in bytes")
    mime_type: str = Field(default="image/jpeg",
                           description="MIME type of the image")


class FaceDetail(BaseModel):
    face_id: uuid.UUID = Field(...,
                               description="The UUID of the face from the faces record")
    name: str = Field(..., description="The name/label of the person")
    description: str = Field(
        ..., description="A short description of what the person is doing or where they are positioned")
    box: List[int] = Field(...,
                           description="Bounding box [x, y, w, h] of the face")


class SceneAnalysisResponse(BaseModel):
    event: str = Field(
        ...,
        description="Type of event happening in the image (e.g., marriage, show, concert, party, meeting, sports, casual gathering, or unknown)"
    )
    surrounding: str = Field(
        ...,
        description="The location or surrounding where the image was taken."
    )
    faces_id: List[uuid.UUID] = Field(
        ..., description="List of unique face IDs detected in the image")
    faces_detail: List[FaceDetail] = Field(
        ..., description="Detailed information about each face identified in the scene")

class FaceUpdate(BaseModel):
    face_id: uuid.UUID
    label: str


class FaceUpdateList(BaseModel):
    faces: List[FaceUpdate]


class DocumentClassificationResponse(BaseModel):
    is_document: bool = Field(
        ...,
        description="Whether the image is a document (e.g., ID card, passport, driver's license, receipt, certificate, document page) or a photograph/scene."
    )


class OCRResponse(BaseModel):
    text: str = Field(..., description="The extracted text from the document.")
