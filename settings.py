from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration loaded from environment and optional .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    # Expose as `pico_api_key` in code, read from env var `PICO_API_KEY`
    pico_api_key: str = Field(
        default="",
        description="PICO API Key used by the application",
        validation_alias="PICO_API_KEY",
    )

    log_level: str = Field(
        default="INFO",
        description="Root log level (DEBUG, INFO, WARNING, ERROR)",
        validation_alias="LOG_LEVEL",
    )

    log_file: str | None = Field(
        default=None,
        description="Optional path to a log file (rotated daily)",
        validation_alias="LOG_FILE",
    )

    # Optional audio device IDs
    input_device_id: int | None = Field(
        default=None,
        description="PyAudio input device index to use (optional)",
        validation_alias="INPUT_DEVICE_ID",
    )

    output_device_id: int | None = Field(
        default=None,
        description="PyAudio output device index to use (optional)",
        validation_alias="OUTPUT_DEVICE_ID",
    )

    # Wake word model path (Porcupine .ppn file), optional for now
    wake_word_path: str | None = Field(
        default=None,
        description="Path to the wake word model file (.ppn)",
        validation_alias="WAKE_WORD_PATH",
    )

    # OpenAI API key (read from env var OPENAI_API_KEY)
    openai_api_key: str = Field(
        default="",
        description="OpenAI API Key used for OpenAI services",
        validation_alias="OPENAI_API_KEY",
    )

    # Agent instructions file path
    agent_instructions_path: str | None = Field(
        default=None,
        description="Path to a file containing system/agent instructions",
        validation_alias="AGENT_INSTRUCTIONS_PATH",
    )

    # Agent voice selection (defaults to 'shimmer')
    agent_voice: str = Field(
        default="shimmer",
        description="Voice name for the agent (e.g., shimmer)",
        validation_alias="AGENT_VOICE",
    )


# A singleton-style instance for convenient imports
settings = Settings()
