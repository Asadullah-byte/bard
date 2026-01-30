from supabase import Client, create_client,AsyncClient
from app.core.config import settings

def get_supabase_client() -> Client:
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)

async def get_async_supabase_client() -> AsyncClient:
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)