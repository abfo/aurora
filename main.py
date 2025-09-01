import base64
import json
import logging
import struct
import time
from settings import settings
from logging_config import configure_logging
import pvporcupine
import pyaudio
import asyncio
import websockets

def main():
    configure_logging(settings.log_level, settings.log_file)
    log = logging.getLogger("aurora")

    agent_instructions = "You are a helpful assistant."
    if settings.agent_instructions_path:
        try:
            with open(settings.agent_instructions_path, "r") as f:
                agent_instructions = f.read()
                log.info(f"Loaded agent instructions from {settings.agent_instructions_path}")
        except FileNotFoundError:
            log.warning(f"Agent instructions file not found: {settings.agent_instructions_path}")

    try:
        audio = pyaudio.PyAudio()
        _log_audio_devices(audio)

        # Main loop
        while True:
            try:
                porcupine = pvporcupine.create(
                    access_key=settings.pico_api_key,
                    keyword_paths=[settings.wake_word_path]
                )
                stream = audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=porcupine.sample_rate,
                    input=True,
                    frames_per_buffer=porcupine.frame_length,
                    input_device_index=settings.input_device_id
                )
                # Wake Word loop
                while True:
                    try:
                        sample = stream.read(porcupine.frame_length, exception_on_overflow = False)
                        sample = struct.unpack_from("h" * porcupine.frame_length, sample)
                        keyword_index = porcupine.process(sample)
                        if keyword_index >= 0:
                            break

                    except Exception as e:
                        log.exception(f"Error in wake word loop: {e}")
                        time.sleep(5)
                        break

                # wake word detected
                if stream:
                    stream.stop_stream()
                    stream.close()
                if porcupine:
                    porcupine.delete()
                log.info("Wake word detected")
                asyncio.run(run_realtime_conversation(audio, agent_instructions, log))

            except Exception as e:
                log.exception(f"Error in main loop: {e}")
                time.sleep(5)

    except Exception as e:
        log.exception(f"Error initializing audio: {e}")

    finally:
        if audio:
            audio.terminate()

async def run_realtime_conversation(audio: pyaudio.PyAudio, agent_instructions: str, log: logging.Logger):
    loop = asyncio.get_running_loop()
    frame_queue: asyncio.Queue[bytes] = asyncio.Queue()

    connect_task = asyncio.create_task(_connect_realtime(agent_instructions))
    input_stream_task = asyncio.create_task(_open_input_stream_async(audio, loop, frame_queue))
    output_stream_task = asyncio.create_task(_open_output_stream_async(audio))

    log.info("Connecting to OpenAI Realtime API and opening audio streams...")
    ws, input_stream, output_stream = await asyncio.gather(connect_task, input_stream_task, output_stream_task)

    log.info("Starting audio...")
    input_stream.start_stream()
    output_stream.start_stream()
    send_audio_task = asyncio.create_task(_send_audio_loop(ws, frame_queue))

    log.info("Ready for conversation.")

    try:
        while True:
            async for response in ws:
                res_1=json.loads(response)

                if res_1.get("type") == "response.audio.delta":
                    base64_audio_data = res_1.get("delta")
                    if base64_audio_data:
                        pcm_data = base64.b64decode(base64_audio_data)
                        output_stream.write(pcm_data)

                if res_1.get("type") == "response.function_call_arguments.done":
                    function_name = res_1.get("name")
                    arguments = json.loads(res_1.get("arguments"))
                    call_id = res_1.get("call_id")
                    output = None

                    if function_name == "go_to_sleep":
                        # throw an exception
                        log.info("User asked us to go to sleep, ending session.")
                        raise Exception("User asked us to go to sleep...")
                    
    except Exception as e:
        log.info(f"Error in conversation: {e}")
    finally:
        log.info("Conversation over, cleaning up.")
        if send_audio_task:
            send_audio_task.cancel()
        if input_stream:
            input_stream.stop_stream()
            input_stream.close()
        if output_stream:
            output_stream.stop_stream()
            output_stream.close()
        if ws:
            await ws.close()


async def _connect_realtime(agent_instructions: str):
    additional_headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "OpenAI-Beta": "realtime=v1"
    }
    ws = await websockets.connect("wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview", additional_headers=additional_headers)
    event = {
        "type": "session.update",
        "session": {
            "modalities": ['audio', 'text'],
            "instructions": agent_instructions,
            "voice": settings.agent_voice,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {
                # "enabled": True,
                "model": "whisper-1"
            },
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.1,
                "prefix_padding_ms": 10,
                "silence_duration_ms": 999
            },
            "tools": [
                {
                    "name": "go_to_sleep",
                    "type": "function",
                    "description": "Puts the assistant to sleep and ends the curent session, call if the user asks you to go to sleep, shut up, stop it, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                        },
                        "required": []
                    }
                }
            ],
            "tool_choice": "auto",
        }
    }
    await ws.send(json.dumps(event))
    return ws
    
async def _send_audio_loop(ws: websockets.ClientConnection, frame_queue: asyncio.Queue):
    while True:
        frame = await frame_queue.get()
        if frame:
            audio_event = {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(frame).decode('utf-8')
            }
            await ws.send(json.dumps(audio_event))         

def _make_audio_callback(loop, frame_queue: asyncio.Queue):
    def _callback(in_data, frame_count, time_info, status_flags):
        # Push from PortAudio thread into the asyncio loop without blocking
        try:
            loop.call_soon_threadsafe(frame_queue.put_nowait, in_data)
        except Exception:
            pass
        return (None, pyaudio.paContinue)
    return _callback

async def _open_input_stream_async(audio: pyaudio.PyAudio, loop, frame_queue: asyncio.Queue):
    def _open():
        return audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            input=True,
            frames_per_buffer=4096,
            input_device_index=settings.input_device_id,
            stream_callback=_make_audio_callback(loop, frame_queue),
            start=False,  # open but don't start yet
        )
    return await asyncio.to_thread(_open)

async def _open_output_stream_async(audio: pyaudio.PyAudio):
    def _open():
        return audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=24000,
            output=True,
            frames_per_buffer=4096,
            output_device_index=settings.output_device_id,
            start=False,  # open but don't start yet
        )
    return await asyncio.to_thread(_open)

# log available audio devices
def _log_audio_devices(audio: pyaudio.PyAudio):
    """Log available PyAudio devices with safe formatting and error guards."""
    log = logging.getLogger("aurora")
    try:
        device_count = audio.get_device_count()
        log.debug("Detected %s audio devices", device_count)
    except Exception:
        log.exception("Failed to get audio device count")
        return

    for i in range(device_count):
        try:
            info = audio.get_device_info_by_index(i)
            name = info.get('name', f'Device {i}')
            max_input = int(info.get('maxInputChannels') or 0)
            max_output = int(info.get('maxOutputChannels') or 0)

            if max_input > 0:
                log.info("Input Device id %s - %s (channels=%s)", i, name, max_input)
            if max_output > 0:
                log.info("Output Device id %s - %s (channels=%s)", i, name, max_output)
            if max_input == 0 and max_output == 0:
                log.debug("Device id %s - %s has no I/O channels reported", i, name)
        except Exception:
            log.exception("Failed to get info for audio device index %s", i)

if __name__ == "__main__":
    main()