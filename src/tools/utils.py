import os
import random
import json
import sys
import argparse
import logging
import httpx
import threading
import time
import functools
import re
import yaml
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from src/.env file
# Using absolute path for robustness
current_dir = os.path.dirname(os.path.abspath(__file__))
# Adjusted for src/tools/ location - look one level up for .env if it's in src/
# If .env is in src/, then it is at ../.env relative to src/tools/
load_dotenv(dotenv_path=os.path.join(current_dir, "..", ".env"))
# load_dotenv() # Fallback or if needed globally

from datasets import load_dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from enum import Enum, auto

class DatasetType(Enum):
    MATH_500 = auto()
    OMNI_MATH = auto()
    GSM8K = auto()
    LOCAL_GENERIC = auto()

    @staticmethod
    def identify(dataset_name: str) -> 'DatasetType':
        name_lower = dataset_name.lower()
        if any(x in name_lower for x in ["math-500", "math_precalculus_5", "math_ia_5", "aime"]):
            return DatasetType.MATH_500
        elif "omni-math" in name_lower:
            return DatasetType.OMNI_MATH
        elif "gsm8k" in name_lower:
            return DatasetType.GSM8K
        return DatasetType.LOCAL_GENERIC

# Global lock for synchronized logging across threads
LOG_LOCK = threading.Lock()

def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
    retryable_subjects: list = None
):
    """
    Decorator for retrying a function with exponential backoff.
    """
    if retryable_subjects is None:
        retryable_subjects = ["connection", "timeout", "rate limit", "api error", "500", "502", "503", "504", "overloaded"]

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract item_id from kwargs if it exists
            item_id = kwargs.get("item_id", "N/A")
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    err_msg = str(e).lower()
                    
                    is_retryable = any(kw in err_msg for kw in retryable_subjects)
                    if not is_retryable or attempt == max_retries:
                        break
                    
                    with LOG_LOCK:
                        logging.warning(f"[{item_id}] {func.__name__} attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                    
                    time.sleep(delay)
                    delay *= backoff_factor
            
            # If we reach here, all retries failed
            with LOG_LOCK:
                logging.error(f"[{item_id}] {func.__name__} failed after {max_retries + 1} attempts: {last_exception}")
            raise last_exception
        return wrapper
    return decorator

# --- Logging Configuration ---
def suppress_external_logging():
    """
    Suppress third-party logs (HTTP requests from httpx, urllib3, datasets, etc.)
    to avoid "HTTP Request: POST ..." noise in terminal.
    """
    for logger_name in [
        "httpx", 
        "httpcore", 
        "openai", 
        "urllib3", 
        "datasets", 
        "langchain",
        "langchain_openai",
        "langchain_core"
    ]:
        target_logger = logging.getLogger(logger_name)
        target_logger.setLevel(logging.WARNING)
        target_logger.propagate = False
        # Also ensure handlers on these loggers don't output if they exist
        if target_logger.hasHandlers():
            target_logger.handlers.clear()

def setup_logging(log_file: Optional[str] = None):
    """
    Sets up logging to console and optionally to a file.
    """
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    if log_file:
        # Ensure the directory for the provided log file exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define common format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)

    # File handler (only if log_file is provided)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Suppress third-party logs
    suppress_external_logging()

    if log_file:
        logging.info(f"Logging initialized. Level: {log_level_str}, Log file: {log_file}")
    else:
        logging.info(f"Logging initialized. Level: {log_level_str} (Console only)")
    
    return log_file

def setup_phoenix(project_name: str = "default_project"):
    """
    Configures Phoenix tracing if available.
    """
    # Check if Phoenix is enabled via env
    if os.getenv("PHOENIX_ENABLED", "true").lower() not in ("true", "1", "yes"):
        return

    try:
        from phoenix.trace.langchain import LangChainInstrumentor
        LangChainInstrumentor().instrument(project_name=project_name)
        logging.info(f"Phoenix tracing enabled for project: {project_name}")
    except ImportError:
        logging.debug("Phoenix not installed. Tracing disabled.")
    except Exception as e:
        logging.warning(f"Failed to setup Phoenix tracing: {e}")

# --- Load Environment Variables ---
# Already loaded at module level

def load_prompts(path: str = None) -> Dict[str, Any]:
    """
    Loads prompts from a YAML file.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.yaml")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.warning(f"Failed to load prompts from {path}: {e}. Using defaults.")
        return {}

class Config:
    """
    Configuration class that manages project-wide settings.
    Priority: Programmatic Overrides (CLI) > Environment Variables > Defaults.
    """
    def __init__(self):
        self._overrides = {}
        self._prompts = None
        # Core defaults for environment/basic settings - only things not in CLI
        self._defaults = {}
        # Type mapping for automatic casting of environment variables
        self._type_map = {
            "float": [
                "TEMPERATURE", "TRAIN_RATIO", "VAL_RATIO", "TEST_RATIO",
                "SIMILARITY_THRESHOLD", "ACCEPTANCE_THRESHOLD", "REGULARIZATION_LAMBDA", 
                "REGULARIZATION_EPSILON", "SOLVER_TEMPERATURE", "OPTIMIZER_TEMPERATURE",
                "INITIALIZER_TEMPERATURE", "REGULARIZER_TEMPERATURE"
            ],
            "int": [
                "SEED", "RETRIEVAL_TOP_K", "TOTAL_LIMIT", "MAX_OPTIMIZATION_STEPS", 
                "SUBJECT_LIMIT", "MAX_WORKERS", "MAX_RETRIES", "K", "TIMEOUT", "MAX_TOKENS"
            ]
        }
        # Add hardcoded defaults for regularization constants if not in env
        self._defaults.update({
            "REGULARIZATION_LAMBDA": 0.1,
            "REGULARIZATION_EPSILON": 1e-6,
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
            "ACCEPTANCE_THRESHOLD": 0.0,
        })

    def update(self, config_dict: Dict[str, Any]):
        """Update configuration with values from a dictionary (e.g., argparse)."""
        if not config_dict:
            return
        # Filter out None values to avoid overriding with nulls
        filtered_config = {k.upper(): v for k, v in config_dict.items() if v is not None}
        self._overrides.update(filtered_config)
        
        # Also sync to environment variables for tools that read from os.environ
        for k, v in filtered_config.items():
            if v is not None:
                os.environ[k] = str(v)
        
        # Special sync for OpenTelemetry / Parallelism
        if "MAX_WORKERS" in filtered_config:
            mw = str(filtered_config["MAX_WORKERS"])
            os.environ["OTEL_PYTHON_OTLP_HTTP_MAX_POOL_SIZE"] = mw
            os.environ["OTEL_EXPORTER_OTLP_HTTP_MAX_CONNECTIONS"] = mw
            
        logging.info(f"Config updated with {len(filtered_config)} overrides.")

    def _get_cast_type(self, key: str) -> type:
        if key in self._type_map["float"]: return float
        if key in self._type_map["int"]: return int
        return str

    def __getattr__(self, name: str) -> Any:
        """Dynamic attribute lookup for config values and prompts."""
        # 1. Handle Prompt access: e.g. Config.SOLVER_SYSTEM_PROMPT
        # We check this first because prompts are always uppercase strings ending in _PROMPT
        if "_PROMPT" in name:
            # Format: AGENTNAME_PROMPTTYPE_PROMPT (e.g., SOLVER_SYSTEM_PROMPT or SOLVER_EXP_SYSTEM_PROMPT)
            parts = name.lower().split("_")
            if len(parts) >= 3:
                # The last two parts are always [type]_prompt (e.g., system_prompt)
                # Everything before that is the agent key in prompts.yaml (e.g., solver, solver_exp)
                p_type = f"{parts[-2]}_prompt" # e.g., 'system_prompt'
                agent_key = "_".join(parts[:-2]) # e.g., 'solver' or 'solver_exp'
                prompt = self.PROMPTS.get(agent_key, {}).get(p_type)
                if prompt is not None:
                    return prompt
                # If name looks like a prompt but isn't found, we'll fall through to normal lookup
                # in case it's actually an environment variable with _PROMPT in its name.

        # 2. Handle Uppercase Config Names (e.g., Config.TEMPERATURE)
        if name.isupper():
            # Special Alias
            if name == "MAX_EXPERIENCE_POOL_SIZE":
                return self.TOTAL_LIMIT
            
            # Special logic for agent-specific temperatures/models defaulting to global
            if (name.endswith("_TEMPERATURE") and name != "TEMPERATURE") or \
               (name.endswith("_MODEL_NAME") and name != "MODEL_NAME"):
                val = self._get_val_with_priority(name)
                if val is not None:
                    return val
                # Fallback to global
                global_key = "TEMPERATURE" if name.endswith("_TEMPERATURE") else "MODEL_NAME"
                return getattr(self, global_key)

            return self._get_val_with_priority(name)

        raise AttributeError(f"'Config' object has no attribute '{name}'")

    def _get_val_with_priority(self, key: str) -> Any:
        """Internal helper to get config with priority logic."""
        cast_type = self._get_cast_type(key)
        
        # 1. Check programmatic overrides (from CLI/Trainer)
        if key in self._overrides:
            val = self._overrides[key]
            try:
                return cast_type(val) if val is not None else None
            except (ValueError, TypeError):
                return val

        # 2. Check environment variables
        val = os.getenv(key)
        if val is not None:
            try:
                return cast_type(val)
            except (ValueError, TypeError):
                pass
        
        # 3. Fallback to default
        return self._defaults.get(key)

    @property
    def PROMPTS(self):
        if self._prompts is None:
            self._prompts = load_prompts()
        return self._prompts


# Instantiate for easy access
Config = Config()


def add_common_arguments(parser: argparse.ArgumentParser):
    """
    Adds standard TSGD arguments to an ArgumentParser.
    Priority: CLI Argument > Environment Variable (.env) > Hardcoded Default.
    """
    def get_env_default(key, default):
        val = os.getenv(key)
        if val is None:
            return default
        # Try to cast to the same type as the default value
        if isinstance(default, bool):
            return val.lower() in ("true", "1", "yes")
        if isinstance(default, int):
            return int(val)
        if isinstance(default, float):
            return float(val)
        return val

    # 1. Environment & Model
    env_group = parser.add_argument_group("Environment & Model")
    env_group.add_argument("--model_name", type=str, 
                         default=get_env_default("MODEL_NAME", "grok-4-1-fast-non-reasoning"), 
                         help="LLM model name")
    env_group.add_argument("--embedding_model", type=str, 
                         default=get_env_default("EMBEDDING_MODEL", "text-embedding-3-large"),
                         help="Embedding model name")
    
    # Agent Specific Models
    env_group.add_argument("--solver_model", type=str, 
                         default=get_env_default("SOLVER_MODEL", None),
                         help="Model name for SolverAgent")
    env_group.add_argument("--optimizer_model", type=str, 
                         default=get_env_default("OPTIMIZER_MODEL", None),
                         help="Model name for OptimizerAgent")
    env_group.add_argument("--initializer_model", type=str, 
                         default=get_env_default("INITIALIZER_MODEL", None),
                         help="Model name for InitializerAgent")
    env_group.add_argument("--regularizer_model", type=str, 
                         default=get_env_default("REGULARIZER_MODEL", None),
                         help="Model name for RegularizerAgent")

    env_group.add_argument("--project_name", type=str, 
                         default=get_env_default("PROJECT_NAME", None),
                         help="Phoenix project name")
    env_group.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    # 2. LLM Hyperparameters
    llm_group = parser.add_argument_group("LLM Hyperparameters")
    llm_group.add_argument("--temperature", type=float, 
                         default=get_env_default("TEMPERATURE", 0.3),
                         help="Global LLM temperature")
    llm_group.add_argument("--max_tokens", type=int, 
                         default=get_env_default("MAX_TOKENS", 4096),
                         help="LLM max_tokens for generation")
    llm_group.add_argument("--timeout", type=int, 
                         default=get_env_default("TIMEOUT", 300),
                         help="LLM request timeout")
    llm_group.add_argument("--max_retries", type=int, 
                         default=get_env_default("MAX_RETRIES", 3),
                         help="LLM max retries")

    # Agent Specific Temperatures
    llm_group.add_argument("--solver_temperature", type=float, 
                         default=get_env_default("SOLVER_TEMPERATURE", None),
                         help="Temperature for SolverAgent")
    llm_group.add_argument("--optimizer_temperature", type=float, 
                         default=get_env_default("OPTIMIZER_TEMPERATURE", None),
                         help="Temperature for OptimizerAgent")
    llm_group.add_argument("--initializer_temperature", type=float, 
                         default=get_env_default("INITIALIZER_TEMPERATURE", None),
                         help="Temperature for InitializerAgent")
    llm_group.add_argument("--regularizer_temperature", type=float, 
                         default=get_env_default("REGULARIZER_TEMPERATURE", None),
                         help="Temperature for RegularizerAgent")

    # 3. Data & Paths
    data_group = parser.add_argument_group("Data & Paths")
    data_group.add_argument("--dataset_name", type=str, 
                         default=get_env_default("DATASET_NAME", None),
                         help="Dataset name or path")
    data_group.add_argument("--experience_dir", type=str, 
                         default=get_env_default("EXPERIENCE_DIR", None),
                         help="Directory to save experience library (experience_pool.jsonl and question_meta.json)")
    data_group.add_argument("--max_samples", type=int, 
                         default=get_env_default("MAX_SAMPLES", None),
                         help="Limit number of samples to process")
    data_group.add_argument("--k", type=int, 
                         default=get_env_default("K", 1),
                         help="Number of samples per problem (Pass@k)")

    # 4. Experience Pool & Retrieval
    pool_group = parser.add_argument_group("Experience Pool & Retrieval")
    pool_group.add_argument("--retrieval_top_k", type=int, 
                         default=get_env_default("RETRIEVAL_TOP_K", 3),
                         help="Number of experiences to retrieve")
    pool_group.add_argument("--similarity_threshold", type=float, 
                         default=get_env_default("SIMILARITY_THRESHOLD", 0.4),
                         help="Similarity threshold for retrieval")
    pool_group.add_argument("--total_limit", type=int, 
                         default=get_env_default("TOTAL_LIMIT", 50),
                         help="Total experience limit for the library")
    pool_group.add_argument("--subject_limit", type=int, 
                         default=get_env_default("SUBJECT_LIMIT", 10),
                         help="Per-subject experience limit")
    pool_group.add_argument("--exp_max_tokens", type=int, 
                         default=get_env_default("EXP_MAX_TOKENS", 1024),
                         help="Max tokens for a single experience")
    
    # 5. Execution
    exec_group = parser.add_argument_group("Execution")
    exec_group.add_argument("--max_workers", type=int, 
                         default=get_env_default("MAX_WORKERS", 10),
                         help="Max parallel workers")
    exec_group.add_argument("--seed", type=int, 
                         default=get_env_default("SEED", 42),
                         help="Random seed")

    return parser

# --- Shared HTTP Client ---
# Using a global client to reuse connections and prevent port exhaustion
_HTTP_CLIENT = None
_CLIENT_LOCK = threading.Lock()

def get_shared_http_client():
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None:
        with _CLIENT_LOCK:
            if _HTTP_CLIENT is None:
                # Set connection pool size to match MAX_WORKERS or a reasonable default
                max_connections = Config.MAX_WORKERS * 2
                limits = httpx.Limits(
                    max_connections=max_connections, 
                    max_keepalive_connections=max_connections
                )
                _HTTP_CLIENT = httpx.Client(limits=limits, timeout=Config.TIMEOUT)
    return _HTTP_CLIENT

# --- LangChain Helpers ---

def get_provider_config(model_name: str):
    """
    Returns (api_key, base_url) based on the model name and project rules.
    """
    # Normalize model name
    m = model_name.lower()
    openrouter_list = ["x-ai/grok-4.1-fast"]
    aihubmix_list = ["gpt-4o-mini", "gpt-4o", "grok-4-1-fast-non-reasoning"]
    
    # 1. OpenRouter (Model C rule: e.g. Grok, x-ai)
    if m in openrouter_list:
        # User requested automatic routing for "Model C" (OpenRouter)
        # We assume Grok/Claude/Gemini usually go via OpenRouter in this setup
        key = os.getenv("OPENROUTER_API_KEY")
        base = os.getenv("OPENROUTER_BASE_URL")
        if key and base:
            logging.info(f"Routing {model_name} to OpenRouter")
            return key, base
    elif m in aihubmix_list:
        # Prioritize AIHubMix for GPT-4o family if keys exist
        key = os.getenv("AIHUBMIX_API_KEY")
        base = os.getenv("AIHUBMIX_BASE_URL")
        if key and base:
             logging.info(f"Routing {model_name} to AIHubMix")
             return key, base
    else:
        key = os.getenv("OPENAI_API_KEY")
        base = os.getenv("OPENAI_BASE_URL")
        if key and base:
             logging.info(f"Routing {model_name} to V3 API")
             return key, base


def get_chat_model(temperature=None, model_name=None, timeout=None, max_tokens=None, max_retries=None):
    http_client = get_shared_http_client()
    
    final_model_name = model_name or Config.MODEL_NAME
    final_max_tokens = max_tokens if max_tokens is not None else Config.MAX_TOKENS
    
    # Determine Provider
    api_key, base_url = get_provider_config(final_model_name)

    kwargs = {
        "model": final_model_name,
        "openai_api_key": api_key,
        "openai_api_base": base_url,
        "timeout": timeout if timeout is not None else Config.TIMEOUT,
        "max_tokens": final_max_tokens,
        "max_retries": max_retries if max_retries is not None else Config.MAX_RETRIES,
        "http_client": http_client
    }

    # Handle models that don't like both temperature and top_p (e.g., Claude)
    target_model = kwargs["model"].lower()

    # Special handling for x-ai/grok-4.1-fast
    if "grok-4.1-fast" in target_model:
        # Add extra_body as a top-level argument to avoid warnings
        extra_body = kwargs.get("extra_body", {})
        # Ensure reasoning is disabled
        if "reasoning" not in extra_body:
             extra_body["reasoning"] = {"enabled": False}
        kwargs["extra_body"] = extra_body

    # Remove top_p for models that don't support it (e.g., Xiaomi gpt-4o-mini)
    # The user memory says Xiaomi/AIHubMix rejects top_p.
    # For now, we only pass temperature if it's not None.
    if temperature is not None:
        kwargs["temperature"] = temperature
    
    return ChatOpenAI(**kwargs)

# --- Data Loading ---

class DataLoader:
    """
    Handles loading and partitioning datasets (Omni-MATH, MATH-500).
    """
    @staticmethod
    def _extract_math_id(item: Dict[str, Any], index: int) -> str:
        """Helper to extract MATH_xxx style IDs."""
        # Use item_id or fallback to unique_id if present in raw data
        raw_id = item.get("item_id") or item.get("unique_id") or item.get("idx") or item.get("id") or str(index)
        
        if isinstance(raw_id, str) and ("MATH_" in raw_id or "/" in raw_id):
            match = re.search(r'(\d+)', os.path.basename(str(raw_id)))
            if match:
                return f"MATH_{match.group(1)}"
        
        if str(raw_id).startswith("MATH_"):
            return str(raw_id)
        
        return f"MATH_{raw_id}"

    @staticmethod
    def partition_data(data: List[Dict[str, Any]], max_samples: int = None) -> List[Dict[str, Any]]:
        """
        Partitions or truncates the data.
        """
        if max_samples is not None:
            logging.info(f"Partitioning data: truncated to {max_samples} samples.")
            return data[:max_samples]
        return data

    @staticmethod
    def load_data(
        dataset_name: str, 
        split: str = "test", 
        max_samples: int = None, 
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Loads data from HuggingFace or local files.
        """
        # Handle aliases
        if dataset_name == "aime2024":
            dataset_name = "HuggingFaceH4/aime_2024"
            if split == "test":
                split = "train" # AIME 2024 usually has only train split
        
        if dataset_name == "yentinglin/aime_2025":
            if split == "test":
                split = "train" # AIME 2025 usually has only train split
        
        logging.info(f"Loading dataset: {dataset_name} ({split})")
        
        try:
            # 1. Try loading from HuggingFace
            dataset = load_dataset(dataset_name, split=split)
        except Exception as e:
            # 2. Fallback: Check if it's a local file
            if os.path.exists(dataset_name):
                ext = dataset_name.split(".")[-1]
                if ext == "jsonl":
                    dataset = load_dataset("json", data_files=dataset_name, split="train")
                elif ext == "csv":
                    dataset = load_dataset("csv", data_files=dataset_name, split="train")
                else:
                    raise ValueError(f"Unsupported local file format: {ext}")
            else:
                raise e

        # Convert to list of dicts and ensure item_id
        data = []
        for i, item in enumerate(dataset):
            # Ensure item_id exists
            if "item_id" not in item or item["item_id"] is None:
                item["item_id"] = DataLoader._extract_math_id(item, i)
            
            # Ensure ground_truth exists
            if "ground_truth" not in item or item["ground_truth"] is None:
                if "answer" in item:
                    item["ground_truth"] = item["answer"]
                elif "solution" in item:
                    item["ground_truth"] = item["solution"]

            data.append(item)
        
        # Shuffle if needed
        if seed is not None:
            random.seed(seed)
            random.shuffle(data)
        
        # Truncate
        if max_samples:
            data = data[:max_samples]
            logging.info(f"Truncated dataset to {max_samples} samples.")
            
        logging.info(f"Loaded {len(data)} samples.")
        return data

    # Alias for compatibility with train.py
    load_dataset = load_data

def get_embeddings_model(model_name: str = "text-embedding-3-large"):
    """
    Returns the configured embeddings model.
    """
    http_client = get_shared_http_client()
    
    # Default to OpenAI settings
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=base_url,
        http_client=http_client,
        check_embedding_ctx_length=False
    )
