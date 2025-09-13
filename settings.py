import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = os.getenv("AURORA_ENV_FILE", str(PROJECT_DIR / ".env"))


class Settings(BaseSettings):
    """Application configuration loaded from environment and optional .env file."""

    model_config = SettingsConfigDict(
    # Resolve .env relative to this file (override with AURORA_ENV_FILE)
    env_file=ENV_FILE,
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

    # Perplexity API key (read from env var PERPLEXITY_API_KEY)
    perplexity_api_key: str = Field(
        default="",
        description="Perplexity API Key used for Perplexity services",
        validation_alias="PERPLEXITY_API_KEY",
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

    # UI implementation to use (default: Debug)
    ui: str = Field(
        default="Debug",
        description="UI implementation to use (e.g., Debug)",
        validation_alias="UI",
    )

    # UI asset paths (optional)
    alarm_font_path: Path | None = Field(
        default=None,
        description="Path to the TTF/OTF font used for alarm display",
        validation_alias="ALARM_FONT_PATH",
    )
    image_talk_path: Path | None = Field(
        default=None,
        description="Path to the image used for TALK state",
        validation_alias="IMAGE_TALK_PATH",
    )
    image_listen_path: Path | None = Field(
        default=None,
        description="Path to the image used for LISTEN state",
        validation_alias="IMAGE_LISTEN_PATH",
    )
    image_sleep_path: Path | None = Field(
        default=None,
        description="Path to the image used for SLEEP state",
        validation_alias="IMAGE_SLEEP_PATH",
    )  
    image_tool_path: Path | None = Field(
        default=None,
        description="Path to the image used for TOOL/processing state",
        validation_alias="IMAGE_TOOL_PATH",
    )

    kid_name_a: str | None = Field(
        default=None,
        description="Display name for Kid A",
        validation_alias="KID_NAME_A",
    )
    kid_name_b: str | None = Field(
        default=None,
        description="Display name for Kid B",
        validation_alias="KID_NAME_B",
    )

    # Todoist integration (optional)
    todoist_api_key: str = Field(
        default="",
        description="Todoist API Key",
        validation_alias="TODOIST_API_KEY",
    )
    todoist_todo_project_id: str | None = Field(
        default="",
        description="Todoist project ID for 'To Do' items",
        validation_alias="TODOIST_TODO_PROJECT_ID",
    )
    todoist_shopping_project_id: str | None = Field(
        default="",
        description="Todoist project ID for 'Shopping' items",
        validation_alias="TODOIST_SHOPPING_PROJECT_ID",
    )
    todoist_todo_due_details: str = Field(
        default="Today",
        description="Default due string for 'To Do' tasks (e.g., Today, tomorrow 6pm)",
        validation_alias="TODOIST_TODO_DUE_DETAILS",
    )
    todoist_shopping_due_details: str = Field(
        default="Saturday",
        description="Default due string for 'Shopping' tasks",
        validation_alias="TODOIST_SHOPPING_DUE_DETAILS",
    )

    # Bay Area 511 transit tool (optional)
    bay_area_511_api_key: str = Field(
        default="",
        description="Bay Area 511 API key",
        validation_alias="BAY_AREA_511_API_KEY",
    )
    bay_area_511_agency: str | None = Field(
        default=None,
        description="Transit agency code for Bay Area 511",
        validation_alias="BAY_AREA_511_AGENCY",
    )
    bay_area_511_stop_code: str | None = Field(
        default=None,
        description="Stop code for Bay Area 511",
        validation_alias="BAY_AREA_511_STOP_CODE",
    )
    bay_area_511_friendly_name: str | None = Field(
        default=None,
        description="Friendly name for the transit line (e.g., 'L train' or '38 bus')",
        validation_alias="BAY_AREA_511_FRIENDLY_NAME",
    )


# A singleton-style instance for convenient imports
settings = Settings()
