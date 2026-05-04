from __future__ import annotations
from capacity_eval.src.models.base import BaseModelClient
from capacity_eval.src.models.api_model import APIModelClient
from capacity_eval.src.models.local_model import LocalModelClient
from capacity_eval.src.config import ModelConfig
from capacity_eval.src.logging_utils import setup_logger

logger = setup_logger()


def create_model_client(config: ModelConfig) -> BaseModelClient:
    if config.type == "api":
        return APIModelClient(config)
    elif config.type == "local":
        return LocalModelClient(config)
    else:
        raise ValueError(f"Unknown model type: {config.type}")


def create_clients_from_configs(configs: list[ModelConfig]) -> dict[str, BaseModelClient]:
    clients: dict[str, BaseModelClient] = {}
    for cfg in configs:
        try:
            clients[cfg.name] = create_model_client(cfg)
        except Exception as e:
            logger.error(f"Failed to create client for {cfg.name}: {e}")
    return clients
