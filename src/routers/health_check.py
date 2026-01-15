import asyncio
import psutil
import os
from fastapi import status
from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse

from src.app import logger_instance
from src.utils.config import settings
from src.utils.config_loader import ConfigReaderInstance


router = APIRouter()
logger = logger_instance.get_logger(__name__)
api_config = ConfigReaderInstance.yaml.read_config_from_file(settings.API_CONFIG_FILENAME)


@router.get('/ping', responses={200: {
            'description': 'Healthcheck Service',
            'content': {
                'application/json': {
                    'example': {'REVISION': '1.0.0'}
                }
            }
        }})

async def health_check() -> JSONResponse:
    logger.info('event=health-check-success message="Successful health check. "')
    content = {'REVISION': api_config.get('API_VERSION')}
    return JSONResponse(content=content, status_code=status.HTTP_200_OK)


@router.get('/debug/resources')
async def get_resource_usage() -> JSONResponse:
    """
    Debug endpoint to monitor resource usage and active tasks.
    Helps identify CPU/memory issues.
    """
    try:
        process = psutil.Process(os.getpid())

        # Get CPU and memory
        cpu_percent = process.cpu_percent(interval=0.1)
        memory_info = process.memory_info()

        # Get active asyncio tasks
        all_tasks = asyncio.all_tasks()
        task_names = {}
        for task in all_tasks:
            name = task.get_name()
            # Group by name prefix
            prefix = name.split('-')[0] if '-' in name else name
            task_names[prefix] = task_names.get(prefix, 0) + 1

        # Get WebSocket connections if available
        ws_info = {}
        try:
            from src.core.websocket_manager import global_connection_manager
            ws_info = {
                "total_topics": len(global_connection_manager.topic_subscriptions),
                "total_clients": len(global_connection_manager.client_topics),
                "topics": list(global_connection_manager.topic_subscriptions.keys())[:20],  # First 20
            }
        except Exception:
            ws_info = {"error": "Could not get WebSocket info"}

        # Get thread count
        thread_count = process.num_threads()

        return JSONResponse(content={
            "pid": os.getpid(),
            "cpu_percent": cpu_percent,
            "memory_mb": round(memory_info.rss / 1024 / 1024, 2),
            "memory_percent": round(process.memory_percent(), 2),
            "thread_count": thread_count,
            "asyncio_tasks": {
                "total": len(all_tasks),
                "by_type": task_names,
            },
            "websocket": ws_info,
        }, status_code=status.HTTP_200_OK)

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
