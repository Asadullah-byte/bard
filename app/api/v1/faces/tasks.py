from app.celery_app import celery
from app.api.v1.faces.helper import (
    detect_faces_with_metadata, 
    recognize_and_store_faces, 
    analyze_scene,
    classify_image_as_document,
    perform_ocr_on_image,
    update_journey_context
)
from app.core.supabse_client import get_supabase_client
import logging
import asyncio

logger = logging.getLogger(__name__)

@celery.task(name="app.api.v1.faces.tasks.process_images")
def process_images_task(image_records: list, user_id: str) -> dict:
    return asyncio.run(process_images_internal(image_records, user_id))


async def process_images_internal(image_records: list, user_id: str) -> dict:
    supabase = get_supabase_client()
    
    async def process_single_image(record: dict):
        image_id = record.get("image_id")
        file_path = record.get("file_stored")
        metadata = record.get("metadata") or {}
        mime_type = metadata.get("mime_type", "image/jpeg")
        
        try:
            # 1. Fetch image data from Supabase Storage
            image_bytes = supabase.storage.from_("images").download(file_path)
            
            # 2. Document Classification (Concurrent via outer asyncio.gather)
            is_document = await classify_image_as_document(image_bytes, mime_type)
            
            description = ""
            
            if is_document:
                # 3a. OCR for Documents
                ocr_text = await perform_ocr_on_image(image_bytes, mime_type)
                description = f"Document: {ocr_text}"
                
                scene_data = {
                    "is_document": True,
                    "ocr_text": ocr_text,
                    "event": "document",
                    "surrounding": "unknown",
                    "faces_id": [],
                    "faces_detail": []
                }
                status_msg = "document_processed"
            else:
                # 3b. Photo Flow: Face Detection + Recognition + Scene Analysis
                faces_meta = detect_faces_with_metadata(image_bytes)
                faces_detail = await recognize_and_store_faces(faces_meta, user_id, image_id, supabase)
                scene_result = await analyze_scene(image_bytes, mime_type, faces_detail)
                
                faces_str = ", ".join([f"{f.name} ({f.description})" for f in scene_result.faces_detail])
                description = f"Photo Event: {scene_result.event}. Surrounding: {scene_result.surrounding}. People: {faces_str}"
                
                scene_data = {
                    "is_document": False,
                    "event": scene_result.event,
                    "surrounding": scene_result.surrounding,
                    "faces_id": [str(f.face_id) for f in faces_detail],
                    "faces_detail": [
                        {
                            "face_id": str(f.face_id),
                            "name": f.name,
                            "description": f.description,
                            "box": f.box
                        } for f in scene_result.faces_detail
                    ]
                }
                status_msg = "photo_processed"
            
            # 4. Update images table
            supabase.table("images").update({"scene_data": scene_data}).eq("id", image_id).execute()
            
            # Extract date from metadata
            capture_date = metadata.get("DateTimeOriginal") or metadata.get("DateTime")
            if not capture_date:
                from datetime import datetime
                capture_date = datetime.now().strftime("%Y:%m:%d")
            else:
                # Keep only the date part if it's "YYYY:MM:DD HH:MM:SS"
                capture_date = capture_date.split(' ')[0]

            return {
                "image_id": image_id,
                "status": "success",
                "processed_as": status_msg,
                "is_document": is_document,
                "description": description,
                "date": capture_date
            }
            
        except Exception as e:
            logger.error(f"Error processing image {image_id}: {e}", exc_info=True)
            return {
                "image_id": image_id,
                "status": "error",
                "error": str(e)
            }

    # Process all images concurrently
    results = await asyncio.gather(*(process_single_image(record) for record in image_records))
    
    # 5. Batch Update Journey Journal
    try:
        from collections import defaultdict
        content_by_date = defaultdict(list)
        
        success_count = 0
        for res in results:
            if res.get("status") == "success" and res.get("description"):
                date = res.get("date", "Unknown Date")
                content_by_date[date].append(res.get("description"))
                success_count += 1
        
        if success_count > 0:
            # Format content by date for the LLM
            formatted_parts = []
            for date in sorted(content_by_date.keys()):
                formatted_parts.append(f"Date: {date}")
                for desc in content_by_date[date]:
                    formatted_parts.append(f"- {desc}")
                formatted_parts.append("") # Empty line between dates
            
            combined_new_content = "\n".join(formatted_parts)
            logger.info(f"Updating journey for user {user_id} with {success_count} new image descriptions across {len(content_by_date)} dates.")
            updated_journey = await update_journey_context(user_id, combined_new_content)
            if updated_journey:
                logger.info(f"Successfully updated journey for user {user_id}.")
            else:
                logger.warning(f"Journey update returned empty for user {user_id}.")
        else:
            logger.info(f"No new content to update for journey for user {user_id}.")
            
    except Exception as journey_err:
        logger.error(f"Error updating journey for user {user_id}: {journey_err}", exc_info=True)
    
    return {"processed_count": len(results), "results": results}