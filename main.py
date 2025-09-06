import base64
import json
import logging
import os
import struct
import time
import wave
from datetime import datetime, timedelta
import requests
from settings import settings
from logging_config import configure_logging
import pvporcupine
import pyaudio
import asyncio
import websockets
from tools.base import load_plugins, Tool
from audio_manager import AudioManager, ScheduledAudio
from ui.base import AssistantUIBase, AssistantUIState

_AUDIO_PLAYBACK_CHUNK = 1024
_REALTIME_SAMPLERATE = 24000
_REALTIME_FRAMESPERBUFFER = 4096

def main():
    configure_logging(settings.log_level, settings.log_file)
    log = logging.getLogger("aurora")

    # Initialize UI (lazy imports to avoid hardware deps on Windows)
    if settings.ui == "Debug":
        from ui.debug import DebugUI
        ui = DebugUI(log)
    elif settings.ui == "Braincraft":
        try:
            from ui.braincraft import BraincraftUI
            ui = BraincraftUI(log)
        except Exception:
            log.exception("Braincraft UI not available on this platform")
            return
    else:
        log.warning(f"Unknown UI implementation: {settings.ui}")
        return

    # create audio manager and load tool plugins with it
    audio_manager = AudioManager(log)
    tools = load_plugins(log=log, audio_manager=audio_manager)

    agent_instructions = "You are a helpful assistant."
    if settings.agent_instructions_path:
        try:
            with open(settings.agent_instructions_path, "r") as f:
                agent_instructions = f.read()
                log.info(f"Loaded agent instructions from {settings.agent_instructions_path}")
        except FileNotFoundError:
            log.warning(f"Agent instructions file not found: {settings.agent_instructions_path}")

    try:
        # wait for internet
        _wait_for_internet_connection()
        ui.update_state(AssistantUIState.LOAD_INTERNET, reason="Internet connection established")

        # open audio and log available devices
        audio = pyaudio.PyAudio()
        _log_audio_devices(audio)
        ui.update_state(AssistantUIState.LOAD_AUDIO, reason="Audio initialized")

        # Main loop
        while True:
            try:
                scheduled_audio = audio_manager.get_audio()
                if scheduled_audio:
                    ui.update_state(AssistantUIState.TALKING, reason="Playing scheduled audio")
                    _play_scheduled_audio(audio, scheduled_audio, ui)

                # Start listening for wake word
                ui.update_state(AssistantUIState.SLEEPING, reason="Listening for wake word")

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
                due_audio = False
                next_timer_update = datetime.now() + timedelta(seconds=1)
                while True:
                    try:
                        sample = stream.read(porcupine.frame_length, exception_on_overflow = False)
                        sample = struct.unpack_from("h" * porcupine.frame_length, sample)
                        keyword_index = porcupine.process(sample)
                        if keyword_index >= 0:
                            break
                        if audio_manager.has_due_audio():
                            due_audio = True
                            break
                        if datetime.now() > next_timer_update:
                            next_timer_update = datetime.now() + timedelta(seconds=1)
                            if audio_manager.has_any_audio():
                                ui.set_timer_text(audio_manager.audio_to_text())

                    except Exception as e:
                        log.exception(f"Error in wake word loop: {e}")
                        time.sleep(5)
                        break

                # cleanup wake word loop
                if stream:
                    stream.stop_stream()
                    stream.close()
                if porcupine:
                    porcupine.delete()

                if due_audio:
                    # got back to start of loop and play audio
                    continue
                
                log.info("Wake word detected")
                asyncio.run(run_realtime_conversation(
                    audio, 
                    agent_instructions, 
                    log, 
                    tools, 
                    audio_manager,
                    ui
                ))

            except Exception as e:
                log.exception(f"Error in main loop: {e}")
                time.sleep(5)

    except Exception as e:
        log.exception(f"Error initializing audio: {e}")

    finally:
        if audio:
            audio.terminate()

async def run_realtime_conversation(
        audio: pyaudio.PyAudio, 
        agent_instructions: str, 
        log: logging.Logger, 
        tools: list[Tool],
        audio_manager: AudioManager,
        ui: AssistantUIBase):
    loop = asyncio.get_running_loop()
    frame_queue: asyncio.Queue[bytes] = asyncio.Queue()
    mic_gate = {"capture": True}

    connect_task = asyncio.create_task(_connect_realtime(agent_instructions, tools))
    input_stream_task = asyncio.create_task(_open_input_stream_async(audio, loop, frame_queue, lambda: mic_gate["capture"]))
    output_stream_task = asyncio.create_task(_open_output_stream_async(audio))

    log.info("Connecting to OpenAI Realtime API and opening audio streams...")
    ws, input_stream, output_stream = await asyncio.gather(connect_task, input_stream_task, output_stream_task)

    log.info("Starting audio...")
    input_stream.start_stream()
    output_stream.start_stream()
    send_audio_task = asyncio.create_task(_send_audio_loop(ws, frame_queue))
    due_audio_task = asyncio.create_task(_due_audio_loop(ws, audio_manager, ui))

    log.info("Ready for conversation.")
    ui.update_state(AssistantUIState.LISTENING, reason="Listening for user input")

    try:
        while True:
            async for response in ws:
                res_1=json.loads(response)
                #log.info(f"Received: {res_1.get('type')}")

                # error, log it
                if res_1.get("type") == "error":
                    error = res_1.get("error")
                    log.warning(f'Realtime Error: {error}')

                # generating a response, update state
                if res_1.get("type") == "response.created":
                    mic_gate["capture"] = False
                    ui.update_state(AssistantUIState.TALKING, reason="Assistant responding")

                # finished a response, update state
                if res_1.get("type") == "response.done":
                    mic_gate["capture"] = True
                    if ui.state != AssistantUIState.TOOL_CALLING: # if tool calling don't update
                        ui.update_state(AssistantUIState.LISTENING, reason="Assistant finished")

                # audio chunk received, play it
                if res_1.get("type") == "response.output_audio.delta":
                    base64_audio_data = res_1.get("delta")
                    if base64_audio_data:
                        pcm_data = base64.b64decode(base64_audio_data)
                        output_stream.write(pcm_data)

                # need to call a function (tool call)
                if res_1.get("type") == "response.function_call_arguments.done":
                    ui.update_state(AssistantUIState.TOOL_CALLING, reason="Starting a tool call")
                    function_name = res_1.get("name")
                    arguments = (res_1.get("arguments"))
                    call_id = res_1.get("call_id")
                    output = None

                    if function_name == "go_to_sleep":
                        # throw an exception
                        log.info("User asked us to go to sleep, ending session.")
                        raise Exception("User asked us to go to sleep...")
                    else:
                        for tool in tools:
                            try:
                                output = tool.handle(function_name, arguments)
                                if output:
                                    log.info(f"Tool {tool.name} handled function call {function_name}")
                                    break
                            except Exception as e:
                                log.exception(f"Error in tool {tool.name}: {e}")

                    if output:
                        message = {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "output": output,
                                "call_id": call_id
                            }
                        }
                        await ws.send(json.dumps(message))

                        # force generation after tool call
                        message = {
                                "type": "response.create"
                            }
                        await ws.send(json.dumps(message))

    except Exception as e:
        log.info(f"Error in conversation: {e}")
    finally:
        log.info("Conversation over, cleaning up.")
        if due_audio_task:
            due_audio_task.cancel()
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

async def _connect_realtime(agent_instructions: str, tools: list[Tool]):
    all_tools = [tool.manifest() for tool in tools]

    # always include the go_to_sleep tool
    all_tools.append({
        "name": "go_to_sleep",
        "type": "function",
        "description": "Puts the assistant to sleep and ends the current session.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    })

    additional_headers = {
        "Authorization": f"Bearer {settings.openai_api_key}"
    }
    
    ws = await websockets.connect(
        "wss://api.openai.com/v1/realtime?model=gpt-realtime",
        additional_headers=additional_headers,
        ping_interval=30,
        ping_timeout=25
    )
    
    event = {
        "type": "session.update",
        "session": {
            "model": "gpt-realtime",
            "type": "realtime",
            "audio": {
                "input" : {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000
                    },
                    "noise_reduction" : {
                        "type": "far_field"
                    },
                    "turn_detection": {
                        "create_response": True,
                        "interrupt_response": True,
                        "eagerness": "auto",
                        "type": "semantic_vad"
                    }
                },
                "output": {
                    "format": {
                        "type": "audio/pcm",
                        "rate": 24000
                    },
                    "speed": 1,
                    "voice": settings.agent_voice
                }
            },
            "instructions": agent_instructions,
            "output_modalities": ['audio'],
            "tools": all_tools,
            "tool_choice": "auto",
        }
    }
    await ws.send(json.dumps(event))
    return ws
    
async def _due_audio_loop(ws: websockets.ClientConnection, audio_manager:AudioManager, ui: AssistantUIBase):
    while True:
        await asyncio.sleep(1)
        if audio_manager.has_due_audio():
            message = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Call the go_to_sleep function"
                            }
                        ]
                    }
                }
            await ws.send(json.dumps(message))
            messageRespond = {
                    "type": "response.create"
                }
            await ws.send(json.dumps(messageRespond))
            await asyncio.sleep(1)
            message2 = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "This is important - you need to go to sleep now."
                            }
                        ]
                    }
                }
            await ws.send(json.dumps(message2))
            messageRespond2 = {
                    "type": "response.create"
                }
            await ws.send(json.dumps(messageRespond2))

        elif audio_manager.has_any_audio():
            ui.set_timer_text(audio_manager.audio_to_text())

async def _send_audio_loop(ws: websockets.ClientConnection, frame_queue: asyncio.Queue):
    while True:
        try:
            frame = await frame_queue.get()
            if frame:
                audio_event = {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(frame).decode('utf-8')
                }
                await ws.send(json.dumps(audio_event))       
        except Exception:
            pass

def _make_audio_callback(loop, frame_queue: asyncio.Queue, should_capture):
    def _callback(in_data, frame_count, time_info, status_flags):
        # Push from PortAudio thread into the asyncio loop without blocking
        try:
            if should_capture():
                asyncio.run_coroutine_threadsafe(frame_queue.put(in_data), loop)
                # loop.call_soon_threadsafe(frame_queue.put_nowait, in_data)
        except Exception:
            pass
        return (in_data, pyaudio.paContinue)
    return _callback

async def _open_input_stream_async(audio: pyaudio.PyAudio, loop, frame_queue: asyncio.Queue, should_capture):
    def _open():
        return audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=_REALTIME_SAMPLERATE,
            input=True,
            frames_per_buffer=_REALTIME_FRAMESPERBUFFER,
            input_device_index=settings.input_device_id,
            stream_callback=_make_audio_callback(loop, frame_queue, should_capture),
            start=False,  # open but don't start yet
        )
    return await asyncio.to_thread(_open)

async def _open_output_stream_async(audio: pyaudio.PyAudio):
    def _open():
        return audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=_REALTIME_SAMPLERATE,
            output=True,
            frames_per_buffer=_REALTIME_FRAMESPERBUFFER,
            output_device_index=settings.output_device_id,
            start=False,  # open but don't start yet
        )
    return await asyncio.to_thread(_open)

def _play_scheduled_audio(audio: pyaudio.PyAudio, scheduled_audio: ScheduledAudio, ui: AssistantUIBase):
    log = logging.getLogger("aurora")
    try:
        wf = wave.open(scheduled_audio.path, 'rb')

        stream = audio.open(format=audio.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output_device_index=settings.output_device_id,
            output=True)
        
        data = wf.readframes(_AUDIO_PLAYBACK_CHUNK)
        while data != b'':
            stream.write(data)
            data = wf.readframes(_AUDIO_PLAYBACK_CHUNK)

            # audio playback is interrupted
            if ui.is_cancel_pressed():
                break
        
        stream.stop_stream()
        stream.close()
        wf.close()

        if (scheduled_audio.delete_after_play):
            try:
                os.remove(scheduled_audio.path)
            except FileNotFoundError:
                log.debug("Audio file already removed: %s", scheduled_audio.path)
            except Exception:
                log.exception("Failed to remove audio file: %s", scheduled_audio.path)

    except Exception as e:
        log.exception("Failed to play scheduled audio: %s", e)

def _wait_for_internet_connection():
    while True:
        try:
            requests.get('https://api.openai.com/').status_code
            break
        except:
            time.sleep(5)
            pass

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