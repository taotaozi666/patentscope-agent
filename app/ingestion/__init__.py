"""ingestion 模块的公共接口。"""

from .ingestion import (get_ingestion_manager, JobNotFoundError, IngestionManager)

__all__ = ['get_ingestion_manager', 'JobNotFoundError', 'IngestionManager']
