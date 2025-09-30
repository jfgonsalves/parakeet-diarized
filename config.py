# Configuration settings for Parakeet
import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path

from pydantic.types import T

# Set up logging
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    # Load environment variables from .env file if it exists
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.warning("dotenv package not installed. Environment variables will only be loaded from system.")

# API settings
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEBUG_MODE = os.environ.get("DEBUG", "0") == "1"

# Model settings
DEFAULT_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_CHUNK_DURATION = 500  # 5 minutes in seconds

# Hugging Face configuration
HF_TOKEN = os.environ.get("HUGGINGFACE_ACCESS_TOKEN")

# Diarization settings
DEFAULT_DIARIZE = True
DEFAULT_NUM_SPEAKERS = None  # None means auto-detection
DEFAULT_INCLUDE_DIARIZATION_IN_TEXT = True  # Whether to include speaker labels in the text


class Config:
    """Global configuration for Parakeet"""

    # Singleton instance
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize configuration with default values"""
        # API settings
        self.host = os.environ.get("HOST", DEFAULT_HOST)
        self.port = int(os.environ.get("PORT", DEFAULT_PORT))
        self.debug = DEBUG_MODE

        # Model settings
        self.model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
        self.temperature = float(os.environ.get("TEMPERATURE", DEFAULT_TEMPERATURE))
        self.chunk_duration = int(os.environ.get("CHUNK_DURATION", DEFAULT_CHUNK_DURATION))

        # Diarization settings
        self.hf_token = HF_TOKEN
        self.enable_diarization = os.environ.get("ENABLE_DIARIZATION", str(DEFAULT_DIARIZE)).lower() == "true"
        self.include_diarization_in_text = os.environ.get("INCLUDE_DIARIZATION_IN_TEXT", str(DEFAULT_INCLUDE_DIARIZATION_IN_TEXT)).lower() == "true"
        self.default_num_speakers = DEFAULT_NUM_SPEAKERS


        # File paths
        self.temp_dir = os.environ.get("TEMP_DIR", "/tmp/parakeet")
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

        logger.debug(f"Initialized configuration: debug={self.debug}, model={self.model_id}")

    def update_hf_token(self, token: str) -> None:
        """Update the HuggingFace token"""
        self.hf_token = token
        logger.info("Updated HuggingFace token")

    def get_hf_token(self) -> Optional[str]:
        """Get the HuggingFace token"""
        return self.hf_token

    def as_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary (for API responses)"""
        return {
            "host": self.host,
            "port": self.port,
            "debug": self.debug,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "chunk_duration": self.chunk_duration,
            "enable_diarization": self.enable_diarization,
            "include_diarization_in_text": self.include_diarization_in_text,
            "has_hf_token": self.hf_token is not None
        }


# Create a global instance
config = Config()

def get_config() -> Config:
    """Get the global configuration instance"""
    return config
