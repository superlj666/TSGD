from abc import ABC, abstractmethod
from typing import Any, Dict
from src.tools.utils import Config, get_chat_model

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the TSGD system.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def _init_llm(self, agent_name: str):
        """
        Helper to initialize LLM with agent-specific settings.
        agent_name: e.g., 'solver', 'optimizer', 'initializer', 'regularizer'
        """
        prefix = agent_name.upper()
        
        # 1. Model Name (CLI/Config Override > Agent Default > Global Default)
        model_key = f"{agent_name}_model"
        default_model = getattr(Config, f"{prefix}_MODEL", Config.MODEL)
        model_name = self.config.get(model_key, default_model)
        
        # 2. Temperature (CLI/Config Override > Agent Default > Global Default)
        temp_key = f"{agent_name}_temperature"
        default_temp = getattr(Config, f"{prefix}_TEMPERATURE", Config.TEMPERATURE)
        temp = self.config.get(temp_key, default_temp)
        
        # 3. Other params (CLI/Config Override > Global Default)
        max_tokens = self.config.get(f"max_tokens") or self.config.get("max_tokens", Config.MAX_TOKENS)
        timeout = self.config.get(f"{agent_name}_timeout") or self.config.get("timeout", Config.TIMEOUT)
        max_retries = self.config.get(f"{agent_name}_max_retries") or self.config.get("max_retries", Config.MAX_RETRIES)
        
        llm = get_chat_model(
            model_name=model_name,
            temperature=temp, 
            max_tokens=max_tokens, 
            timeout=timeout,
            max_retries=max_retries
        )
        
        # Log initialization details
        import logging
        logging.info(f"[{agent_name.capitalize()}] Initialized with Model: {model_name}, Temp: {temp}")
        
        return llm

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        pass
