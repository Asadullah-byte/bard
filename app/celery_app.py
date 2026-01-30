from celery import Celery
from app.core.config import settings
celery = Celery(
    "worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)
celery.conf.update(
    task_serializer=settings.CELERY_TASK_SERIALIZER,
    accept_content=settings.CELERY_ACCEPT_CONTENT,
    result_serializer=settings.CELERY_RESULT_SERIALIZER,
    timezone=settings.CELERY_TIMEZONE,
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.WORKER_TIMEOUT,
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=20,
)

celery.autodiscover_tasks(["app.api.v1.faces.tasks", "app.api.v1.faces.tasks"])
celery.conf.task_routes = {
    "app.api.v1.faces.tasks.process_images": {"queue": "process_images"},
}
