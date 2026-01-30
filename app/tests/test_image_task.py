import asyncio
import uuid
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.api.v1.faces.tasks import process_images_internal
from app.core.supabse_client import get_supabase_client

async def test_processing():
    # This script assumes you have a user and an image record in your local/dev Supabase
    # 1. Manually specify a user_id and image_id if you want to run this for real
    user_id = "c7b242d2-5687-4089-bc38-056729e1c060"
    image_id = "0397a385-7004-40fa-94b2-f8f5e1e3f72c"
    
    if user_id == "YOUR_USER_ID":
        print("Skipping real run. Please set user_id and image_id in the script to test.")
        return

    supabase = get_supabase_client()
    
    # Fetch record to mock the input
    response = supabase.table("images").select("*").eq("id", image_id).execute()
    if not response.data:
        print(f"Image {image_id} not found.")
        return
    
    record = response.data[0]
    
    print(f"Starting processing for image {image_id}...")
    result = await process_images_internal([record], user_id)
    print("Result:", result)

if __name__ == "__main__":
    asyncio.run(test_processing())
