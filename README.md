# Aurora

Aurora is an Alexa-style assistant designed for Raspberry PI. It uses the OpenAI Realtime API and Picovoice Porcupine for wake word detection. 

## Configuration

This project uses Pydantic Settings to load configuration from environment variables and an optional local `.env` file.

Copy `.env.example` to `.env` and follow the instructions in the file (or set the appropriate environment variables).

Create a virtual environment (recommended) and then install dependencies:

```
pip install -r requirements.txt
```

### Environment variables

- `PICO_API_KEY` (required): API key for Picovoice Porcupine.
- `LOG_LEVEL` (optional): DEBUG, INFO, WARNING, ERROR. Default: INFO.
- `LOG_FILE` (optional): Path to a log file; if set, logs will also be written there with daily rotation.
- `INPUT_DEVICE_ID` (optional): PyAudio input device index.
- `OUTPUT_DEVICE_ID` (optional): PyAudio output device index.
- `WAKE_WORD_PATH` (optional): Full path to a Porcupine wake word model (.ppn). Not used yet.

### Logging

Logging is configured via `logging_config.configure_logging()` using the standard library. By default it logs to stdout with timestamps. Control verbosity with `LOG_LEVEL` and optionally write to a file with `LOG_FILE`.

To discover device IDs, set `LOG_LEVEL=DEBUG` and run the app; it will enumerate available input/output devices in the logs.
