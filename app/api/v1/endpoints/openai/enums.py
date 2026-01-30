from enum import Enum


class AITask(Enum):
    IMAGE_CLASSIFICATION = "image_classification"
    DOCUMENT_DETECTION = "document_detection"
    DOCUMENT_OCR = "document_ocr"
    JOURNEY_GENERATION = "journey_generation"
    JOURNEY_UPDATE = "journey_update"