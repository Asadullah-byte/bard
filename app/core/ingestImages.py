import base64
import asyncio
import io
import httpx
import logging
from typing import Tuple, List, Union
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import pillow_heif
from app.core.generator_base import JournalGenerator

# Configure logger
logger = logging.getLogger(__name__)

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

def extract_metadata(image: Image.Image) -> dict:
    """
    Extracts all available EXIF metadata including GPS, Make, Model, Flash, etc.
    """
    metadata = {}
    
    # 1. Try standard EXIF
    exif_data = image._getexif()
    if exif_data:
        for tag, value in exif_data.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_info = {}
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = value[t]
                metadata["GPS"] = gps_info
            else:
                # Filter out very long binary data or maker notes to keep it clean
                if isinstance(value, (bytes, bytearray)) and len(value) > 100:
                    continue
                metadata[decoded] = value
    
    # 2. If no EXIF or to supplement, check standard image info
    # (PNGs often store metadata in .info, not _getexif)
    if not metadata and hasattr(image, "info"):
        for k, v in image.info.items():
            if k == "exif": 
                # Raw EXIF bytes in PNG/WEBP, specialized parsing needed if we want to support it without _getexif,
                # but PIL usually handles it. If strictly bytes, skip or try to parse ? 
                # For now, skip raw bytes to avoid JSON errors
                continue
            if isinstance(v, (str, int, float)):
                 metadata[k] = v

    return metadata

def parse_gps_location(gps_info: dict) -> Tuple[Union[float, None], Union[float, None]]:
    """
    Parses GPS dictionary and returns (Latitude, Longitude) in decimal degrees.
    Returns (None, None) if parsing fails.
    """
    def _to_degrees(value):
        # Handle IFDRational or tuple
        if isinstance(value, tuple) and len(value) == 3:
            d, m, s = value
        elif hasattr(value, 'numerator') and hasattr(value, 'denominator'):
             return float(value)
        else:
             # Fallback if it's already a float or int, though unlikely for DMS
             try:
                 return float(value)
             except (ValueError, TypeError):
                 return 0.0
        
        # Convert rational tuples to floats if needed
        d = float(d)
        m = float(m)
        s = float(s)
        return d + (m / 60.0) + (s / 3600.0)

    try:
        lat = _to_degrees(gps_info.get('GPSLatitude', (0,0,0)))
        lat_ref = gps_info.get('GPSLatitudeRef', 'N')
        lon = _to_degrees(gps_info.get('GPSLongitude', (0,0,0)))
        lon_ref = gps_info.get('GPSLongitudeRef', 'E')
        
        if lat_ref != "N": lat = -lat
        if lon_ref != "E": lon = -lon
        
        return lat, lon
    except Exception:
        return None, None

def clean_metadata_for_json(metadata: dict) -> dict:
    """
    Converts metadata values to JSON-serializable formats (str, float, int).
    """
    cleaned = {}
    for k, v in metadata.items():
        if k == "GPS":
             # Handle GPS Separately or skip (it's handled by parse_gps_location)
             continue
             
        if isinstance(v, (int, float, str, bool, type(None))):
            cleaned[k] = v
        elif isinstance(v, bytes):
            cleaned[k] = f"<binary data: {len(v)} bytes>"
        elif hasattr(v, '__str__'):
            cleaned[k] = str(v)
        else:
            cleaned[k] = repr(v)
    return cleaned

async def extract_image_metadata(image_input: Union[str, bytes]) -> dict:
    """
    Extracts metadata and GPS from a single image bytes/path.
    Returns a dictionary suitable for JSON response.
    """
    result = {
        "gps": None,
        "metadata": {},
        "error": None
    }
    
    try:
        raw_bytes = await fetch_image_bytes(image_input)
        image = Image.open(io.BytesIO(raw_bytes))
        
        raw_metadata = extract_metadata(image)
        
        # Extract GPS
        if "GPS" in raw_metadata:
            lat, lon = parse_gps_location(raw_metadata["GPS"])
            if lat is not None and lon is not None:
                result["gps"] = {"latitude": lat, "longitude": lon}
        
        # Clean rest of metadata
        result["metadata"] = clean_metadata_for_json(raw_metadata)
        
    except Exception as e:
        result["error"] = str(e)
        
    return result

async def ingest_images_metadata(image_inputs: List[Union[str, bytes]]) -> List[dict]:
    """
    Concurrent processing for metadata extraction only (No LLM).
    """
    tasks = [extract_image_metadata(img) for img in image_inputs]
    return await asyncio.gather(*tasks)

def _format_metadata(metadata: dict) -> str:
    """
    Formats the metadata dictionary into a readable string for the LLM.
    """
    lines = []
    
    # Handle GPS specifically
    if "GPS" in metadata:
        gps_info = metadata["GPS"]
        try:
            lat, lon = parse_gps_location(gps_info)
            if lat is not None and lon is not None:
                lines.append(f"Location: {lat}, {lon}")
        except Exception:
            pass # GPS parsing failed, ignore
            
        # We don't delete "GPS" here to avoid side effects if metadata is reused
        
    # Add other interesting fields with priority
    priority_keys = ["Make", "Model", "DateTimeOriginal", "Flash", "Software", "LensModel"]
    for key in priority_keys:
        if key in metadata:
            lines.append(f"{key}: {metadata[key]}")
    
    # Add remaining keys (excluding priority ones)
    for key, value in metadata.items():
        if key not in priority_keys and key != "GPS": # Skip GPS here as we handled it
             lines.append(f"{key}: {value}")
             
    return "\n".join(lines)

def convert_to_jpeg(image_data: bytes) -> bytes:
    """
    Converts image bytes (any format supported by Pillow, including HEIC) to JPEG bytes.
    Raises exception if conversion fails.
    """
    img = Image.open(io.BytesIO(image_data))
    
    # Optimization: If already JPEG, return original bytes
    if img.format == "JPEG":
        return image_data
    
    # Handle RGBA/P to RGB conversion (JPEG doesn't support alpha)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
        
    output_buffer = io.BytesIO()
    img.save(output_buffer, format="JPEG")
    return output_buffer.getvalue()

async def fetch_image_bytes(image_input: Union[str, bytes]) -> bytes:
    """
    Fetches raw image bytes from URL or file path.
    """
    if isinstance(image_input, str):
        if image_input.startswith("http"):
            async with httpx.AsyncClient() as client:
                response = await client.get(image_input)
                response.raise_for_status()
                return response.content
        else:
            with open(image_input, "rb") as image_file:
                return image_file.read()
    else:
        return image_input

async def process_single_image(image_input: Union[str, bytes]) -> str:
    """
    Takes an image (URL or file path), extracts metadata from RAW bytes,
    then converts to JPEG for OpenAI description.
    """
    # 1. Fetch RAW bytes
    raw_image_bytes = await fetch_image_bytes(image_input)

    # 2. Extract Metadata (from RAW bytes)
    metadata_str = ""
    try:
        # Open raw image for metadata extraction
        image = Image.open(io.BytesIO(raw_image_bytes))
        metadata = extract_metadata(image)
        metadata_str = _format_metadata(metadata)
        logger.debug(f"Metadata extracted: {metadata_str}")
        if not metadata_str:
             metadata_str = "No specific metadata found."
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        metadata_str = f"Error extracting metadata: {str(e)}"
    
    # 3. Convert to JPEG & Encode (for LLM)
    try:
        jpeg_bytes = convert_to_jpeg(raw_image_bytes)
        base64_image = base64.b64encode(jpeg_bytes).decode('utf-8')
    except Exception as e:
        logger.error(f"Image conversion failed: {str(e)}")
        return f"Error converting image for analysis: {str(e)}"

    # 4. Delegate to Generator Class
    generator = JournalGenerator()
    return await generator.describe_image(base64_image, metadata_str)

async def ingest_single_day(image_inputs: List[Union[str, bytes]]) -> List[Tuple[Union[str, bytes], str]]:
    """
    Processes multiple images in parallel and returns a list of (image_input, description) tuples.
    """
    tasks = [process_single_image(img) for img in image_inputs]
    
    # Run all description tasks in parallel
    results_descriptions = await asyncio.gather(*tasks, return_exceptions=True)
    
    final_results = []
    for img, desc_or_exc in zip(image_inputs, results_descriptions):
        if isinstance(desc_or_exc, Exception):
            logger.error(f"Error processing image {img}: {str(desc_or_exc)}")
            final_results.append((img, f"Error processing image: {str(desc_or_exc)}"))
        else:
            final_results.append((img, desc_or_exc))
            
    return final_results

async def generate_day_journal(image_inputs: List[Union[str, bytes]]) -> str:
    """
    Full pipeline: Process images parallelly -> Get Descriptions -> Generate Journal Entry.
    """
    # 1. Ingest images to get descriptions
    processed_images = await ingest_single_day(image_inputs)
    
    # 2. Extract just the descriptions
    descriptions = [desc for _, desc in processed_images]
    
    # 3. Generate entry using the Base Class
    generator = JournalGenerator()
    return await generator.generate_entry(descriptions)

if __name__ == "__main__":
    import asyncio
    import os

    # Ensure you are running this from the project root using:
    # uv run python -m app.core.ingestImages
    
    test_dir = "E:/github/bard/test-images"
    
    if not os.path.exists(test_dir):
        print(f"Directory not found: {test_dir}")
    else:
        # Get all files in directory
        image_paths = [
            os.path.join(test_dir, f) 
            for f in os.listdir(test_dir) 
            if os.path.isfile(os.path.join(test_dir, f))
        ]
        
        print(f"Found {len(image_paths)} images. Processing class based approach...")
        
        try:
            # Generate journal from all images
            result = asyncio.run(generate_day_journal(image_paths))
            print("\n--- Generated Journal Entry ---\n")
            print(result)
        except Exception as e:
            print(f"Error: {e}")