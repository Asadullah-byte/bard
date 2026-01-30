import logging
import numpy as np
import uuid
from app.core.supabse_client import get_supabase_client
from deepface import DeepFace
from app.core.config import settings
from retinaface import RetinaFace
import cv2
from typing import List, Any
import os
from app.schemas.faces import ImageClassificationResponse, SceneAnalysisResponse, FaceDetail, DocumentClassificationResponse, OCRResponse
from app.schemas.journey import JourneyResponse
from app.api.v1.endpoints.openai.enums import AITask
from app.api.v1.endpoints.openai.provider import ProviderRegistry, OpenAIProvider
from app.api.v1.endpoints.openai.factory import AIAnalysisFactory
from app.api.v1.endpoints.openai.strategies import ImageInput, AIAnalysisStrategy, JourneyInput, JourneyUpdateInput
from openai import AsyncOpenAI
import json
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


def calculate_similarity(embedding1, embedding2):

    return np.dot(embedding1, embedding2)


def normalize(vec):
    vec = np.array(vec, dtype=np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def add_embedding_to_index(label, embedding):
    embedding_normalized = normalize(embedding).tolist()

    try:
        supabase = get_supabase_client()
        response = supabase.table("face_embeddings").insert({
            "name": label,
            "embedding": embedding_normalized
        }).execute()

        if response.data:
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"Error inserting embedding: {e}")
        return False


def detect_faces_and_embeddings(image_data):
    try:
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = image_data

        unique_filename = f"/tmp/temp_face_{uuid.uuid4()}.jpg"
        cv2.imwrite(unique_filename, img)

        detections = RetinaFace.extract_faces(
            img_path=unique_filename, align=True)
        if os.path.exists(unique_filename):
            os.remove(unique_filename)

    except Exception as e:
        print(f"Face detection failed: {e}")
        # Ensure cleanup in case of error
        # We might need to duplicate the filename logic or just broad try/finally if we refactor more,
        # but for this specific block:
        if 'unique_filename' in locals() and os.path.exists(unique_filename):
            os.remove(unique_filename)
        return []

    results = []
    for face_crop in detections:
        try:
            emb = DeepFace.represent(
                img_path=face_crop,
                model_name=settings.EMBED_MODEL,
                enforce_detection=False
            )[0]["embedding"]
            results.append((normalize(emb), face_crop))
        except Exception as e:
            print(f"Embedding generation failed: {e}")
            continue

    return results


def detect_faces_with_metadata(image_data):
    """
    Detects faces and returns embeddings along with their bounding box metadata.
    Returns: List of (embedding, face_crop, [x, y, w, h])
    """
    try:
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            img = image_data

        unique_filename = f"/tmp/temp_face_meta_{uuid.uuid4()}.jpg"
        cv2.imwrite(unique_filename, img)

        # RetinaFace.detect_faces returns a dict with 'face_1', 'face_2' etc.
        # Each having 'facial_area': [x1, y1, x2, y2]
        detections = RetinaFace.detect_faces(img_path=unique_filename)

        # We also need the crops for DeepFace representation
        # extract_faces returns a list of aligned face crops (numpy arrays)
        face_crops = RetinaFace.extract_faces(
            img_path=unique_filename, align=True)

        if os.path.exists(unique_filename):
            os.remove(unique_filename)

        if not detections or not face_crops:
            return []

    except Exception as e:
        logger.error(f"Face detection with metadata failed: {e}")
        if 'unique_filename' in locals() and os.path.exists(unique_filename):
            os.remove(unique_filename)
        return []

    results = []
    # Match detections (boxes) with face_crops
    for i, (key, value) in enumerate(detections.items()):
        if i >= len(face_crops):
            break

        face_crop = face_crops[i]
        box = value["facial_area"]  # [x1, y1, x2, y2]
        # Convert to [x, y, w, h]
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        try:
            emb = DeepFace.represent(
                img_path=face_crop,
                model_name=settings.EMBED_MODEL,
                enforce_detection=False
            )[0]["embedding"]
            results.append((normalize(emb), face_crop, [x1, y1, w, h]))
        except Exception as e:
            logger.error(
                f"Embedding generation failed for detection {key}: {e}")
            continue

    return results


def detect_face_coordinates_and_match(
    image_data: bytes,
    face_embeddings: list[dict],
    similarity_threshold: float = 0.6
) -> list[dict]:
    """
    Detects faces in an image, generates embeddings, and matches against provided
    face_embeddings to retrieve face_id for known faces.
    
    Args:
        image_data: Raw image bytes.
        face_embeddings: List of face embedding records from DB to match against.
        similarity_threshold: Minimum similarity to consider a match (default 0.6).
    
    Returns:
        List of face dictionaries with normalized x, y, width, height and face_id if matched.
    """
    unique_filename = None
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Failed to decode image data")
            return []
        
        img_height, img_width = img.shape[:2]
        
        unique_filename = f"/tmp/temp_face_match_{uuid.uuid4()}.jpg"
        cv2.imwrite(unique_filename, img)
        
        # Detect faces with bounding boxes
        detections = RetinaFace.detect_faces(img_path=unique_filename)
        
        # Extract face crops for embedding generation
        face_crops = RetinaFace.extract_faces(img_path=unique_filename, align=True)
        
        if os.path.exists(unique_filename):
            os.remove(unique_filename)
        
        if not detections or not isinstance(detections, dict) or not face_crops:
            return []
        
        faces_list = []
        padding = 0.1  # 10% padding
        
        for i, (key, value) in enumerate(detections.items()):
            if i >= len(face_crops):
                break
            
            box = value.get("facial_area", [])
            if len(box) != 4:
                continue
            
            x1, y1, x2, y2 = box
            
            # Convert to x, y, w, h
            w = x2 - x1
            h = y2 - y1
            x = x1
            y = y1
            
            # Normalize to 0-1 range
            x = x / img_width
            y = y / img_height
            w = w / img_width
            h = h / img_height
            
            # Add padding for security margin
            pad_w = w * padding
            pad_h = h * padding
            x = max(0, x - pad_w / 2)
            y = max(0, y - pad_h / 2)
            w = min(1 - x, w + pad_w)
            h = min(1 - y, h + pad_h)
            
            # Generate embedding for this face
            face_id = None
            face_name = None
            try:
                emb = DeepFace.represent(
                    img_path=face_crops[i],
                    model_name=settings.EMBED_MODEL,
                    enforce_detection=False
                )[0]["embedding"]
                emb_normalized = normalize(emb)
                
                # Match against provided face_embeddings
                best_match = None
                max_sim = -1
                
                for face in face_embeddings:
                    existing_emb = face.get("embedding")
                    if existing_emb:
                        if isinstance(existing_emb, str):
                            try:
                                existing_emb = json.loads(existing_emb)
                            except json.JSONDecodeError:
                                continue
                        
                        sim = calculate_similarity(emb_normalized, existing_emb)
                        if sim >= similarity_threshold and sim > max_sim:
                            max_sim = sim
                            best_match = face
                
                if best_match:
                    face_id = str(best_match.get("id"))
                    face_name = best_match.get("name")
                    
            except Exception as emb_err:
                logger.warning(f"Embedding generation failed for face {i}: {emb_err}")
            
            faces_list.append({
                "face_id": face_id,
                "label": face_name,
                "x": x,
                "y": y,
                "width": w,
                "height": h
            })
        
        return faces_list
        
    except Exception as e:
        logger.error(f"Face coordinate detection with matching failed: {e}", exc_info=True)
        if unique_filename and os.path.exists(unique_filename):
            os.remove(unique_filename)
        return []


async def recognize_and_store_faces(faces: list, user_id: str, image_id: str, supabase_client: Any) -> List[FaceDetail]:
    """
    Identifies faces and manages the face_embeddings table with strict deduplication.
    If a face matches an existing one with similarity >= 0.6, no new entry is created.
    """
    recognized_faces = []

    # Fetch existing faces for THIS user
    existing_faces_response = supabase_client.table("face_embeddings") \
        .select("*") \
        .eq("face_owner", str(user_id)) \
        .execute()
    existing_faces_data = existing_faces_response.data if existing_faces_response.data else []

    for emb, _, box in faces:
        best_match = None
        max_sim = -1

        for face in existing_faces_data:
            existing_emb = face.get("embedding")
            if existing_emb:
                if isinstance(existing_emb, str):
                    try:
                        existing_emb = json.loads(existing_emb)
                    except json.JSONDecodeError:
                        continue

                sim = calculate_similarity(emb, existing_emb)
                if sim >= 0.6 and sim > max_sim:
                    max_sim = sim
                    best_match = face

        face_id = None
        matched_label = "unknown"

        if best_match:
            # Face is known (similarity >= 0.6)
            matched_label = best_match.get("name")
            logger.info(f"Face matched existing '{matched_label}' (sim: {max_sim:.4f}). Inserting new entry for this image.")
            
            # Insert new entry linking this face instance to this image, but preserving the identity name
            data = {
                "name": matched_label,
                "embedding": emb.tolist(),
                "face_owner": str(user_id),
                "image_id": image_id
            }
            res = supabase_client.table("face_embeddings").insert(data).execute()
            if res.data:
                face_id = uuid.UUID(res.data[0].get("id"))

        else:
            # New face (similarity < 0.6 or no existing faces)
            data = {
                "name": "unknown",
                "embedding": emb.tolist(),
                "face_owner": str(user_id),
                "image_id": image_id
            }
            res = supabase_client.table("face_embeddings").insert(data).execute()
            if res.data:
                face_id = uuid.UUID(res.data[0].get("id"))
                logger.info(f"New face detected. Created entry with ID: {face_id}")

        recognized_faces.append(FaceDetail(
            face_id=face_id,
            name=matched_label,
            description="",  # GPT will fill this later
            box=box
        ))

    return recognized_faces


@dataclass
class SceneInput(ImageInput):
    faces_detail: List[FaceDetail]


class SceneAnalysisStrategy(AIAnalysisStrategy):
    def __init__(self, provider):
        self.provider = provider

    async def execute(self, *, input: ImageInput) -> SceneAnalysisResponse:
        import base64
        image_base64 = base64.b64encode(input.image).decode('utf-8')

        # Extract face info for prompt
        faces_context = ""
        if hasattr(input, 'faces_detail') and input.faces_detail:
            faces_context = "\nDetected faces with their IDs, names, and bounding boxes [x, y, w, h]:\n"
            for f in input.faces_detail:
                faces_context += f"- ID: {f.face_id}, Name: {f.name}, Box: {f.box}\n"

        system_prompt = (
            "You are a master storyteller and visual analyst. Your task is to analyze the provided image and weave a narrative about the scene, "
            "identifying the event and describing the surroundings in vivid detail. "
            "Crucially, I have provided coordinates for faces detected in the image. "
            "You must identify which person corresponds to which face ID based on their position, and then incorporate them into the story. "
            "For each face ID, describe their role, action, or emotion in the scene (e.g., 'The groom, standing proudly in the center', 'A guest laughing joyfully on the left'). "
            "If a name is provided and is not 'unknown', refer to them by name to personalize the narrative. "
            "Return the result as a JSON object matching the SceneAnalysisResponse schema."
        )

        user_content = [
            {
                "type": "input_text",
                "text": f"Analyze this image and map the faces to the scene. {faces_context}"
            },
            {
                "type": "input_image",
                "image_url": f"data:{input.mime_type};base64,{image_base64}"
            }
        ]

        result = await self.provider.parse_json(
            model="gpt-4o",
            system_prompt=system_prompt,
            user_prompt=user_content,
            response_format=SceneAnalysisResponse,
        )
        print(f"OPENAI response: {result}")
        return result


async def analyze_scene(image_data: bytes, mime_type: str, faces_detail: List[FaceDetail]) -> SceneAnalysisResponse:
    try:
        openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        provider_registry = ProviderRegistry(
            openai=OpenAIProvider(client=openai_client)
        )
        provider = provider_registry.get("openai")
        strategy = SceneAnalysisStrategy(provider=provider)

        scene_input = SceneInput(
            image=image_data,
            platform=None,
            mime_type=mime_type,
            response_format=SceneAnalysisResponse,
            faces_detail=faces_detail
        )

        return await strategy.execute(input=scene_input)

    except Exception as e:
        logger.error(f"Scene analysis failed: {e}", exc_info=True)
        return SceneAnalysisResponse(
            event="unknown",
            surrounding="unknown",
            faces_id=[f.face_id for f in faces_detail],
            faces_detail=faces_detail
        )


async def image_classification(image_data: bytes, mime_type: str) -> ImageClassificationResponse:

    try:
        openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        provider_registry = ProviderRegistry(
            openai=OpenAIProvider(client=openai_client)
        )
        factory = AIAnalysisFactory(providers=provider_registry)

        strategy = factory.create(AITask.IMAGE_CLASSIFICATION)

        image_input = ImageInput(
            image=image_data,
            platform=None,
            mime_type=mime_type,
            response_format=ImageClassificationResponse
        )

        result = await strategy.execute(input=image_input)
        logger.warning(f"Image classification strategy result: {result}")
        if isinstance(result, ImageClassificationResponse):
            return result
        elif isinstance(result, dict):
            return ImageClassificationResponse(**result)
        else:
            return ImageClassificationResponse(
                event="unknown",
                surrounding="unknown",
            )

    except Exception as e:
        logger.error(f"Image classification failed: {e}", exc_info=True)
        return ImageClassificationResponse(
            event="unknown",
            surrounding="unknown",
        )


async def classify_image_as_document(image_data: bytes, mime_type: str) -> bool:
    """
    Determines if an image is a document using OpenAI.
    """
    try:
        openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        provider_registry = ProviderRegistry(
            openai=OpenAIProvider(client=openai_client)
        )
        factory = AIAnalysisFactory(providers=provider_registry)
        strategy = factory.create(AITask.DOCUMENT_DETECTION)

        image_input = ImageInput(
            image=image_data,
            platform=None,
            mime_type=mime_type,
            response_format=DocumentClassificationResponse
        )

        result = await strategy.execute(input=image_input)
        if isinstance(result, DocumentClassificationResponse):
            return result.is_document
        elif isinstance(result, dict):
            return result.get("is_document", False)
        return False
    except Exception as e:
        logger.error(f"Document classification failed: {e}", exc_info=True)
        return False


async def perform_ocr_on_image(image_data: bytes, mime_type: str) -> str:
    """
    Extracts text from a document image using OpenAI OCR.
    """
    try:
        openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        provider_registry = ProviderRegistry(
            openai=OpenAIProvider(client=openai_client)
        )
        factory = AIAnalysisFactory(providers=provider_registry)
        strategy = factory.create(AITask.DOCUMENT_OCR)

        image_input = ImageInput(
            image=image_data,
            platform=None,
            mime_type=mime_type,
            response_format=OCRResponse
        )

        result = await strategy.execute(input=image_input)
        if isinstance(result, OCRResponse):
            return result.text
        elif isinstance(result, dict):
            return result.get("text", "")
        return ""
    except Exception as e:
        logger.error(f"OCR failed: {e}", exc_info=True)
        return ""


async def generate_journey_journal(responses: List[dict]) -> str:
    """
    Generates a Markdown journal entry from onboarding responses using OpenAI.
    """
    try:
        openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        provider_registry = ProviderRegistry(
            openai=OpenAIProvider(client=openai_client)
        )
        factory = AIAnalysisFactory(providers=provider_registry)
        strategy = factory.create(AITask.JOURNEY_GENERATION)

        journey_input = JourneyInput(
            responses=responses,
            response_format=JourneyResponse
        )

        result = await strategy.execute(input=journey_input)
        if isinstance(result, JourneyResponse):
            return result.journey_so_far
        elif isinstance(result, dict):
            return result.get("journey_so_far", "")
        return ""
    except Exception as e:
        logger.error(f"Journey generation failed: {e}", exc_info=True)
        return ""


def get_face_embeddings_by_image_id(image_id: str, supabase_client) -> list[dict]:

    response = supabase_client.table("face_embeddings").select("*").eq("image_id", image_id).execute()
    
    return response.data if response.data else []


def get_image_by_id(image_id: str, supabase_client) -> dict | None:

    response = supabase_client.table("images").select("*").eq("id", image_id).execute()
    
    if response.data and len(response.data) > 0:
        return response.data[0]
    return None


def get_image_public_url(file_path: str, supabase_client) -> str | None:

    if not file_path:
        return None
    return supabase_client.storage.from_("images").get_public_url(file_path)


def fetch_image_dimensions(public_url: str) -> tuple[int | None, int | None]:
    
    from PIL import Image
    import io
    import httpx
    
    try:
        with httpx.Client() as client:
            resp = client.get(public_url)
            pil_img = Image.open(io.BytesIO(resp.content))
            return pil_img.width, pil_img.height
    except Exception as fetch_err:
        logger.warning(f"Could not fetch image dimensions from URL: {fetch_err}")
        return None, None


def extract_face_coordinates(
    faces_detail: list,
    img_width: int | None,
    img_height: int | None,
    public_url: str | None,
    padding: float = 0.1
) -> list[dict]:

    faces_list = []
    
    for face in faces_detail:
        box = face.get("box", [])
        if len(box) != 4:
            continue
        
        x, y, w, h = box
        
        # Normalize coordinates if dimensions are available
        if img_width and img_height:
            x = x / img_width
            y = y / img_height
            w = w / img_width
            h = h / img_height
        
        # Add padding for security margin
        pad_w = w * padding
        pad_h = h * padding
        x = max(0, x - pad_w / 2)
        y = max(0, y - pad_h / 2)
        w = min(1 - x, w + pad_w)  # Clamp to image bounds
        h = min(1 - y, h + pad_h)
        
        faces_list.append({
            "id": face.get("face_id"),
            "face_id": face.get("face_id"),
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "label": face.get("name", ""),
            "public_url": public_url
        })
    
    return faces_list


async def update_journey_context(user_id: str, new_content: str) -> str:
    """
    Updates the user's journey journal by appending new content processing via LLM.
    """
    try:
        supabase = get_supabase_client()
        
        # 1. Fetch existing journey
        response = supabase.table("journey").select("id, journey_so_far").eq("user_id", user_id).execute()
        current_journey = ""
        journey_id = None
        
        if response.data and len(response.data) > 0:
            current_journey = response.data[0].get("journey_so_far", "")
            journey_id = response.data[0].get("id")

        # 2. Call LLM to update
        openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        provider_registry = ProviderRegistry(
            openai=OpenAIProvider(client=openai_client)
        )
        factory = AIAnalysisFactory(providers=provider_registry)
        strategy = factory.create(AITask.JOURNEY_UPDATE)
        
        journey_input = JourneyUpdateInput(
            current_journey=current_journey,
            new_content=new_content,
            response_format=JourneyResponse
        )
        
        result = await strategy.execute(input=journey_input)
        updated_text = ""
        
        if isinstance(result, JourneyResponse):
            updated_text = result.journey_so_far
        elif isinstance(result, dict):
            updated_text = result.get("journey_so_far", "")
            
        # 3. Update DB
        if updated_text:
            data = {
                "user_id": user_id,
                "journey_so_far": updated_text,
                "updated_at": datetime.now().isoformat()
            }
            if journey_id:
                # data["id"] = journey_id # Don't update ID
                supabase.table("journey").update(data).eq("id", journey_id).execute()
            else:
                supabase.table("journey").insert(data).execute()
                
        return updated_text

    except Exception as e:
        logger.error(f"Failed to update journey context: {e}", exc_info=True)
        return ""
