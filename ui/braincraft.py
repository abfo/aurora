from __future__ import annotations
from enum import Enum
import logging
import threading
import board
from digitalio import DigitalInOut, Direction, Pull
import adafruit_dotstar
import digitalio
from PIL import Image, ImageDraw, ImageFont
from adafruit_rgb_display import st7789 
from adafruit_rgb_display.rgb import image_to_data 
from settings import settings

from .base import AssistantUIBase, AssistantUIState

_BUTTON_PIN = board.D17
_JOYDOWN_PIN = board.D27
_JOYLEFT_PIN = board.D22
_JOYUP_PIN = board.D23
_JOYRIGHT_PIN = board.D24
_JOYSELECT_PIN = board.D16
_DOTSTAR_DATA = board.D5
_DOTSTAR_CLOCK = board.D6
_DISPLAY_BAUDRATE = 24000000
_DISPLAY_HEIGHT = 240
_DISPLAY_ROTATION = 180
_DISPLAY_XOFFSET = 80

class ImageState(Enum):
    SLEEP = 1
    TALK = 2
    LISTEN = 3
    TOOL = 4
    BLANK = 5

class BraincraftUI(AssistantUIBase):
    """Skeleton UI for Raspberry Pi 4 with Braincraft display.

    This class provides the minimal concrete implementation required to
    subclass AssistantUIBase, but intentionally contains no specific
    behavior yet. It can be wired up later to the Braincraft hardware
    (display, buttons, LEDs, etc.).
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        # Set logger early so we can log initialization failures
        self._log = logger or logging.getLogger("aurora.ui.braincraft")
        try:
            super().__init__()

            # initialize buttons
            buttons = [_BUTTON_PIN, _JOYUP_PIN, _JOYDOWN_PIN,
               _JOYLEFT_PIN, _JOYRIGHT_PIN, _JOYSELECT_PIN]
            for i, pin in enumerate(buttons):
                buttons[i] = DigitalInOut(pin)
                buttons[i].direction = Direction.INPUT
                buttons[i].pull = Pull.UP
            self._button, self._joyup, self._joydown, self._joyleft, self._joyright, self._joyselect = buttons

            # initialize dotstar
            self._dotstar = adafruit_dotstar.DotStar(
                _DOTSTAR_CLOCK, _DOTSTAR_DATA, 3, brightness=0.05
            )

            # all red until we start initializing 
            self._dotstar[0] = (255, 0, 0)
            self._dotstar[1] = (255, 0, 0)
            self._dotstar[2] = (255, 0, 0)

            # initialize display
            self._cs_pin = digitalio.DigitalInOut(board.CE0)
            self._dc_pin = digitalio.DigitalInOut(board.D25)
            self._reset_pin = digitalio.DigitalInOut(board.D24)
            self._spi = board.SPI()

            self._display = st7789.ST7789(
                self._spi,
                height=_DISPLAY_HEIGHT,
                y_offset=_DISPLAY_XOFFSET,
                rotation=_DISPLAY_ROTATION,
                cs=self._cs_pin,
                dc=self._dc_pin,
                rst=self._reset_pin,
                baudrate=_DISPLAY_BAUDRATE,
            )

            self._font = ImageFont.truetype(settings.alarm_font_path, 27)

            self._image_black = image_to_data(self._create_black_image())
            self._image_sleep = image_to_data(self._load_image(settings.image_sleep_path))
            self._image_listen = image_to_data(self._load_image(settings.image_listen_path))
            self._image_talk = image_to_data(self._load_image(settings.image_talk_path))
            self._image_tool = image_to_data(self._load_image(settings.image_tool_path))
            self._image_state = ImageState.BLANK

            # black image to start
            self._send_image_data(self._image_black)

            # Redraws run on a worker thread. A state change is two full-screen
            # SPI frames (~230 KB at 24 MHz) plus bit-banged LED updates, which
            # takes long enough that doing it on the caller - the asyncio loop -
            # stalled audio. The callers now only flag that a redraw is due.
            # _draw_lock serialises all display and LED access, since the
            # training cues still draw directly from the caller.
            self._draw_lock = threading.Lock()
            self._redraw = threading.Event()
            self._closing = False
            # Set while a training cue owns the screen and LEDs, so a redraw
            # that was already pending can't land on top of the cue. Cleared by
            # the next state change, when training is over.
            self._cue_showing = False
            self._worker = threading.Thread(target=self._draw_loop, name="braincraft-ui", daemon=True)
            self._worker.start()

        except Exception:
            self._log.exception("Failed to initialize BraincraftUI")
            raise

    def on_state_changed(
        self,
        previous: AssistantUIState,
        current: AssistantUIState,
        reason: str | None = None,
    ) -> None:
        self._log.info("UI state changed from %s to %s: %s", previous, current, reason)
        self._cue_showing = False
        self._redraw.set()

    def on_timer_text_changed(self, text: str) -> None:
        self._redraw.set()

    def _draw_loop(self) -> None:
        """Redraw whenever a change has been flagged, always from the latest state.

        Changes that land while a redraw is in progress just set the flag again,
        so a burst of them costs one more redraw of wherever things ended up,
        not one redraw per change.
        """
        while True:
            self._redraw.wait()
            self._redraw.clear()
            if self._closing:
                return
            with self._draw_lock:
                if not self._cue_showing:
                    self._update_state()

    # Wake-word training cues: all LEDs green for positives, red for negatives.
    # These draw synchronously, so the cue is showing by the time the caller
    # starts its lead-in timer; nothing is playing during training capture.
    # Capture starts straight after a state change, so the worker may still be
    # about to redraw for it - _cue_showing stops that redraw replacing the cue.
    def show_training_prompt(self, label: str, index: int, total: int) -> None:
        try:
            with self._draw_lock:
                self._cue_showing = True
                color = (0, 255, 0) if label == "positives" else (255, 0, 0)
                self._dotstar[0] = color
                self._dotstar[1] = color
                self._dotstar[2] = color
                if label == "positives":
                    self._display_text(f"Say Aurora\n{index}/{total}")
                else:
                    self._display_text(f"Say any word\n{index}/{total}")
        except Exception:
            self._log.exception("show_training_prompt failed")

    def clear_training_lights(self) -> None:
        try:
            with self._draw_lock:
                self._dotstar[0] = (0, 0, 0)
                self._dotstar[1] = (0, 0, 0)
                self._dotstar[2] = (0, 0, 0)
        except Exception:
            self._log.exception("clear_training_lights failed")

    # The joystick press doubles as the training button: it cancels during
    # playback and starts wake-word training while idle. The directions are
    # unreachable once the Pi is in its case, so this is the only usable input.
    def is_train_pressed(self) -> bool:
        try:
            return not self._joyselect.value
        except Exception:
            self._log.exception("is_train_pressed check failed")
            return False

    def is_cancel_pressed(self) -> bool:
        try:
            return not self._joyselect.value
        except Exception:
            self._log.exception("is_cancel_pressed check failed")
            return False

    def is_shutdown_pressed(self) -> bool:
        try:
            return not self._joydown.value
        except Exception:
            self._log.exception("is_shutdown_pressed check failed")
            return False

    # Lifecycle
    def shutdown(self) -> None:
        self._closing = True
        self._redraw.set()
        self._worker.join(timeout=2)
        with self._draw_lock:
            self._send_image_data(self._image_black)
            self._dotstar.deinit()
            self._spi.deinit()

    def _update_state(self):
        try:
            # timer text is the highest priority
            if self._timer_text:
                self._display_text(self._timer_text)

            next_image_state = ImageState.BLANK

            if self._state == AssistantUIState.LOAD_START:
                # initalizing, all LEDs red
                self._dotstar[0] = (255, 0, 0)
                self._dotstar[1] = (255, 0, 0)
                self._dotstar[2] = (255, 0, 0)
            elif self._state == AssistantUIState.LOAD_INTERNET:
                # internet connected, first LED green
                self._dotstar[0] = (0, 255, 0)
                self._dotstar[1] = (255, 0, 0)
                self._dotstar[2] = (255, 0, 0)
            elif self._state == AssistantUIState.LOAD_DISPLAY:
                # display connected, second LED green
                self._dotstar[0] = (0, 255, 0)
                self._dotstar[1] = (0, 255, 0)
                self._dotstar[2] = (255, 0, 0)
            elif self._state == AssistantUIState.LOAD_AUDIO:
                # audio connected, third LED green
                self._dotstar[0] = (0, 255, 0)
                self._dotstar[1] = (0, 255, 0)
                self._dotstar[2] = (0, 255, 0)
            elif self._state == AssistantUIState.SLEEPING:
                next_image_state = ImageState.SLEEP
                self._dotstar[0] = (0, 0, 0)
                self._dotstar[1] = (0, 0, 0)
                self._dotstar[2] = (0, 0, 0)
            elif self._state == AssistantUIState.TALKING:
                next_image_state = ImageState.TALK
                self._dotstar[0] = (0, 0, 0)
                self._dotstar[1] = (0, 0, 0)
                self._dotstar[2] = (0, 0, 255)
            elif self._state == AssistantUIState.LISTENING:
                next_image_state = ImageState.LISTEN
                self._dotstar[0] = (255, 0, 0)
                self._dotstar[1] = (0, 0, 0)
                self._dotstar[2] = (0, 0, 0)
            elif self._state == AssistantUIState.TOOL_CALLING:
                next_image_state = ImageState.TOOL
                self._dotstar[0] = (0, 0, 0)
                self._dotstar[1] = (0, 255, 0)
                self._dotstar[2] = (0, 0, 0)

            if not self._timer_text:
                if not next_image_state == ImageState.BLANK:
                    self._display_image(next_image_state)
        except Exception:
            self._log.exception("_update_state failed")

    def _display_image(self, image_state: ImageState):
        try:
            if self._image_state != image_state:
                self._image_state = image_state

                # black first
                self._send_image_data(self._image_black)

                # then the reqested image
                if (self._image_state == ImageState.SLEEP):
                    self._send_image_data(self._image_sleep)
                elif (self._image_state == ImageState.LISTEN):
                    self._send_image_data(self._image_listen)
                elif (self._image_state == ImageState.TALK):
                    self._send_image_data(self._image_talk)
                elif (self._image_state == ImageState.TOOL):
                    self._send_image_data(self._image_tool)
        except Exception:
            self._log.exception("_display_image failed")

    def _send_image_data(self, data):
        try:
            self._display._block(0, 0, _DISPLAY_HEIGHT - 1, _DISPLAY_HEIGHT - 1, data)
        except Exception:
            self._log.exception("_send_image_data failed")

    def _display_text(self, text):
        try:
            if self._display.rotation % 180 == 90:
                height = self._display.width  # we swap height/width to rotate it to landscape!
                width = self._display.height
            else:
                width = self._display.width  # we swap height/width to rotate it to landscape!
                height = self._display.height

            image = Image.new('RGB', (width, height), color=(0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.text((20, 20), text, font=self._font, fill=(255, 255, 255))
            self._display.image(image)
        except Exception:
            self._log.exception("_display_text failed")

    def _to565(self, image: Image.Image) -> bytes | None:
        """Convert PIL Image to RGB565 byte stream once; use library helper if available."""
        try:
            if image_to_data:
                # Uses the Adafruit helper for correct 565 packing
                return bytes(image_to_data(image, self._display.rotation))
            # Fallback: let driver do conversion (returns None so caller can fallback)
            return None
        except Exception:
            self._log.exception("_to565 conversion failed")
            return None

    def _load_image(self, filename):
        try:
            if self._display.rotation % 180 == 90:
                height = self._display.width  # we swap height/width to rotate it to landscape!
                width = self._display.height
            else:
                width = self._display.width  # we swap height/width to rotate it to landscape!
                height = self._display.height

            image = Image.open(filename)
            image.load()

            # Scale the image to the smaller screen dimension
            image_ratio = image.width / image.height
            screen_ratio = width / height
            if screen_ratio < image_ratio:
                scaled_width = image.width * height // image.height
                scaled_height = height
            else:
                scaled_width = width
                scaled_height = image.height * width // image.width
            image = image.resize((scaled_width, scaled_height), Image.BICUBIC)

            # Crop and center the image
            x = scaled_width // 2 - width // 2
            y = scaled_height // 2 - height // 2
            image = image.crop((x, y, x + width, y + height))
            image = image.transpose(Image.ROTATE_180) 
            return image
        except Exception:
            self._log.exception("_load_image failed for %s", filename)
            raise

    def _create_black_image(self):
        try:
            if self._display.rotation % 180 == 90:
                height = self._display.width  # we swap height/width to rotate it to landscape!
                width = self._display.height
            else:
                width = self._display.width  # we swap height/width to rotate it to landscape!
                height = self._display.height

            image = Image.new("RGB", (width, height))

            # Get drawing object to draw on image.
            draw = ImageDraw.Draw(image)

            # Draw a black filled box to clear the image.
            draw.rectangle((0, 0, width, height), outline=0, fill=(0, 0, 0))
            return image
        except Exception:
            self._log.exception("_create_black_image failed")
            raise
