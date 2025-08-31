import logging
import struct
from time import time
from settings import settings
from logging_config import configure_logging
import pvporcupine
import pyaudio

def main():
    configure_logging(settings.log_level, settings.log_file)
    log = logging.getLogger("aurora")

    pico_api_key = settings.pico_api_key
    if not pico_api_key:
        log.error("PICO_API_KEY is not set.")
        return
        
    try:
        audio = pyaudio.PyAudio()
        log_audio_devices(audio)

        # Main loop
        while True:
            try:
                porcupine = pvporcupine.create(
                    access_key=pico_api_key,
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
                        log.exception("Error in wake word loop", e)
                        time.sleep(5)
                        break

                # wake word detected
                if stream:
                    stream.stop_stream()
                    stream.close()
                if porcupine:
                    porcupine.delete()
                log.info("Wake word detected")

            except Exception as e:
                log.exception("Error in main loop", e)
                time.sleep(5)

    except Exception as e:
        log.exception("Error initializing audio", e)

    finally:
        if audio:
            audio.terminate()

# log available audio devices
def log_audio_devices(audio):
    log = logging.getLogger("aurora")
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            log.info("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
        else:
            log.info("Output Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

if __name__ == "__main__":
    main()