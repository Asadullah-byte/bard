from typing import Any, Dict, List
from fastapi import APIRouter, UploadFile, File, Depends, status, Form, HTTPException
import logging
import asyncio
import uuid
from app.core.ingestImages import extract_image_metadata
from app.schemas.response import APIResponse
from app.core.errors import BadRequestException, InternalServerErrorException
from app.api import deps
from app.models.user import User
from app.core.supabse_client import get_supabase_client
from app.api.v1.faces.tasks import process_images_task
from app.schemas.faces import FaceUpdateList

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/register-face", response_model=APIResponse[Dict[str, Any]], status_code=status.HTTP_201_CREATED)
async def register_face_endpoint(
    label: str = Form(...),
    images: List[UploadFile] = File(...),
    current_user: User = Depends(deps.get_current_active_user)
):

    if not label:
        raise BadRequestException(detail="Label (name) is required")

    if not images:
        raise BadRequestException(detail="At least one image is required")

    total_embeddings = 0
    failed_images = 0

    try:
        supabase = get_supabase_client()
        from app.api.v1.faces.helper import detect_faces_with_metadata, calculate_similarity

        # Fetch existing faces for THIS user
        existing_faces_response = supabase.table("face_embeddings") \
            .select("*") \
            .eq("face_owner", str(current_user.id)) \
            .execute()
        existing_faces_data = existing_faces_response.data if existing_faces_response.data else []

        for image_file in images:
            try:
                # 1. Process Image Upload (Store in bucket & DB)
                upload_result = await process_image_upload(image_file, current_user.id, supabase)
                
                if upload_result.get("status") == "error":
                    logger.error(f"Failed to upload image {image_file.filename}: {upload_result.get('error')}")
                    failed_images += 1
                    continue
                
                image_id = upload_result.get("image_id")
                
                # 2. Reset file pointer and read for face detection
                await image_file.seek(0)
                image_data = await image_file.read()
                
                faces_meta = detect_faces_with_metadata(image_data)

                if not faces_meta:
                    failed_images += 1
                    continue

                for emb, _, _ in faces_meta:
                    best_match = None
                    max_sim = -1

                    for face in existing_faces_data:
                        existing_emb = face.get("embedding")
                        if existing_emb:
                            if isinstance(existing_emb, str):
                                import json
                                try:
                                    existing_emb = json.loads(existing_emb)
                                except json.JSONDecodeError:
                                    continue

                            sim = calculate_similarity(emb, existing_emb)
                            if sim >= 0.6 and sim > max_sim:
                                max_sim = sim
                                best_match = face

                    # Since this is explicit registration, we want to register the face with the provided label.
                    # Logic: 
                    # If match found, check if it's the SAME name. If so, just add new embedding entry linked to this image.
                    # If match found with DIFFERENT name, log warning but still perhaps add it? 
                    #   The user explicitly said "Register this as X". If it looks like Y, maybe we should update or add instance.
                    #   For now, sticking to: If explicitly registering "Label", we create an entry with "Label".
                    # But if we want simple behavior:
                    
                    # Store the new embedding linked to this image
                    data = {
                        "name": label,
                        "embedding": emb.tolist(),
                        "face_owner": str(current_user.id),
                        "image_id": image_id
                    }
                    res = supabase.table("face_embeddings").insert(data).execute()
                    if res.data:
                        total_embeddings += 1
                        # Update local list to avoid duplicates in same batch if needed, 
                        # though for explicit registration we might just want to count it.
                        existing_faces_data.append(res.data[0]) 

            except Exception as e:
                logger.error(f"Error processing image {image_file.filename}: {e}")
                failed_images += 1
                continue

        return APIResponse.success_response(
            message=f"Successfully registered {label}",
            data={
                "total_embeddings": total_embeddings,
                "failed_images": failed_images
            }
        )
    except Exception as e:
        logger.error(f"Error registering face: {e}")
        raise InternalServerErrorException(detail="Failed to register face")


async def process_image_upload(
    file: UploadFile,
    user_id: uuid.UUID,
    supabase_client: Any
) -> Dict[str, Any]:
    try:
        content = await file.read()
        filename = file.filename
        content_size = len(content)

        metadata_result = await extract_image_metadata(content)
        metadata = metadata_result.get("metadata", {})
        gps = metadata_result.get("gps")
        if gps:
            metadata["gps"] = gps

        # Extract image dimensions using PIL
        from PIL import Image
        import io
        try:
            img = Image.open(io.BytesIO(content))
            metadata["width"] = img.width
            metadata["height"] = img.height
        except Exception as dim_err:
            logger.warning(f"Could not extract image dimensions: {dim_err}")

        image_id = str(uuid.uuid4())
        file_path = f"images/{user_id}/{image_id}"

        supabase_client.storage.from_("images").upload(
            file_path,
            content,
            {"content-type": file.content_type or "application/octet-stream"}
        )

        public_url_res = supabase_client.storage.from_(
            "images").get_public_url(file_path)
        public_url = public_url_res

        image_data_db = {
            "id": image_id,
            "file_owner": str(user_id),
            "file_stored": file_path,
            "metadata": metadata
        }

        supabase_client.table("images").insert(image_data_db).execute()

        return {
            "filename": filename,
            "status": "success",
            "image_id": image_id,
            "file_stored": file_path,
            "metadata": metadata,
            "public_url": public_url,
            "size_bytes": content_size
        }

    except Exception as e:
        logger.error(
            f"Error processing file {file.filename}: {e}", exc_info=True)
        return {
            "filename": file.filename,
            "status": "error",
            "error": str(e)
        }


@router.post("/upload", response_model=APIResponse[Dict[str, Any]], status_code=status.HTTP_201_CREATED)
async def upload_images(
    images: List[UploadFile] = File(...),
    current_user: User = Depends(deps.get_current_active_user)
) -> Any:
    try:
        if not images:
            raise BadRequestException(detail="No image files provided.")
        if len(images) > 10:
            raise BadRequestException(
                detail="Maximum 10 images can be uploaded at a time.")
        logger.info(f"User {current_user.id} uploading {len(images)} images.")

        supabase = get_supabase_client()

        tasks = [process_image_upload(
            file, current_user.id, supabase) for file in images]
        results = await asyncio.gather(*tasks)
        logger.info(f"Upload images with results: {results}")

        # Trigger background processing task
        successful_uploads = [r for r in results if r["status"] == "success"]
        if successful_uploads:
            process_images_task.delay(successful_uploads, str(current_user.id))

        return APIResponse.success_response(
            message="Images Uploaded Successfully. Processing started in background.",
            data={
                "images": results,
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error uploading images: {e}", exc_info=True)
        raise InternalServerErrorException(detail="Failed to upload images")


@router.get("", response_model=APIResponse[Dict[str, Any]], status_code=status.HTTP_200_OK)
async def get_images(
    current_user: User = Depends(deps.get_current_active_user)
) -> Any:
    try:
        supabase = get_supabase_client()

        response = supabase.table("images").select(
            "*").eq("file_owner", str(current_user.id)).execute()

        images_data = response.data if response.data else []

        results = []
        for img in images_data:
            file_path = img.get("file_stored")
            if file_path:
                public_url = supabase.storage.from_(
                    "images").get_public_url(file_path)

                img_response = {
                    "id": img.get("id"),
                    "filename": file_path.split('/')[-1] if file_path else "unknown",
                    "public_url": public_url,
                    "metadata": img.get("metadata"),
                    "created_at": img.get("created_at")
                }
                results.append(img_response)

        return APIResponse.success_response(
            message="Successfully retrieved user images",
            data={
                "images": results,
            }
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error fetching images: {e}", exc_info=True)
        raise InternalServerErrorException(detail="Failed to fetch images")


@router.get("/{image_id}/faces", response_model=APIResponse[Dict[str, Any]], status_code=status.HTTP_200_OK)
async def get_faces_by_image_id(
    image_id: str,
    current_user: User = Depends(deps.get_current_active_user)
) -> Any:

    from app.api.v1.faces.helper import (
        get_image_by_id,
        get_image_public_url,
        get_face_embeddings_by_image_id,
        detect_face_coordinates_and_match
    )
    from app.core.errors import NotFoundException
    import httpx

    try:
        supabase = get_supabase_client()

        # 1. Check if image exists and belongs to current user
        image_data = get_image_by_id(image_id, supabase)

        if not image_data:
            raise NotFoundException(
                detail=f"Image with ID '{image_id}' not found")

        if image_data.get("file_owner") != str(current_user.id):
            raise NotFoundException(
                detail=f"Image with ID '{image_id}' not found")

        # 2. Check if face_embeddings exist for this image
        face_embeddings = get_face_embeddings_by_image_id(image_id, supabase)

        # Get public URL for the image
        file_path = image_data.get("file_stored")
        public_url = get_image_public_url(file_path, supabase)

        if not face_embeddings:
            return APIResponse.success_response(
                message="No faces detected in this image",
                data={"faces": [], "public_url": public_url}
            )

        if not public_url:
            raise InternalServerErrorException(
                detail="Image file path not found")

        # 3. Fetch image bytes from storage
        async with httpx.AsyncClient() as client:
            response = await client.get(public_url)
            if response.status_code != 200:
                raise InternalServerErrorException(
                    detail="Failed to fetch image from storage")
            image_bytes = response.content

        # 4. Run DeepFace on image and match against face_embeddings for this image
        faces_list = detect_face_coordinates_and_match(
            image_bytes, face_embeddings)

        if not faces_list:
            return APIResponse.success_response(
                message="No faces detected in this image",
                data={"faces": [], "public_url": public_url}
            )

        return APIResponse.success_response(
            message="Successfully retrieved face data",
            data={"faces": faces_list, "public_url": public_url}
        )

    except NotFoundException:
        raise
    except InternalServerErrorException:
        raise
    except Exception as e:
        logger.error(
            f"Error fetching faces for image {image_id}: {e}", exc_info=True)
        raise InternalServerErrorException(detail="Failed to fetch faces")


@router.put("/{image_id}/faces", response_model=APIResponse[Dict[str, Any]], status_code=status.HTTP_200_OK)
async def update_face_labels(
    image_id: str,
    faces_data: FaceUpdateList,
    current_user: User = Depends(deps.get_current_active_user)
) -> Any:

    from app.api.v1.faces.helper import get_image_by_id
    from app.core.errors import NotFoundException

    try:
        supabase = get_supabase_client()

        # Verify image exists and belongs to current user
        image_data = get_image_by_id(image_id, supabase)

        if not image_data:
            raise NotFoundException(
                detail=f"Image with ID '{image_id}' not found")

        if image_data.get("file_owner") != str(current_user.id):
            raise NotFoundException(
                detail=f"Image with ID '{image_id}' not found")

        if not faces_data.faces:
            raise BadRequestException(detail="No faces provided for update")

        updated_count = 0
        failed_updates = []

        for face in faces_data.faces:
            face_id = str(face.face_id)
            label = face.label

            if not face_id or not label:
                failed_updates.append(
                    {"face_id": face_id, "error": "Missing face_id or label"})
                continue

            try:
                # Update name for all face_embeddings with same id
                result = supabase.table("face_embeddings").update(
                    {"name": label}
                ).eq("id", face_id).execute()

                if result.data:
                    updated_count += 1
                else:
                    failed_updates.append(
                        {"face_id": face_id, "error": "Face not found"})
            except Exception as update_err:
                logger.error(f"Failed to update face {face_id}: {update_err}")
                failed_updates.append(
                    {"face_id": face_id, "error": str(update_err)})

        return APIResponse.success_response(
            message=f"Successfully updated {updated_count} face(s)",
            data={
                "updated_count": updated_count,
                "failed_updates": failed_updates
            }
        )

    except NotFoundException:
        raise
    except BadRequestException:
        raise
    except Exception as e:
        logger.error(
            f"Error updating face labels for image {image_id}: {e}", exc_info=True)
        raise InternalServerErrorException(
            detail="Failed to update face labels")
