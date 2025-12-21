
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from .schemas import FramePacket


@dataclass
class UIElement:
    id: str
    text: str
    bbox: tuple[float, float, float, float]
    role: str
    action: dict[str, Any] = field(default_factory=dict)
    selected: bool = False
    enabled: bool = True


class GUITarget(ABC):
    """
    Abstract protocol for any pointer-centric GUI surface.

    Capability properties tell the Brain what motor strategies are available.
    Every concrete backend must implement at least get_new_frame and send_hid.
    """

    def __init__(self, key: str, width: int = 320, height: int = 640) -> None:
        self.key = key
        self.width = width
        self.height = height

    # ── capability properties ───────────────────────────────────────────

    @property
    def has_direct_cursor(self) -> bool:
        """True if cursor position is directly observable (desktop, simulator)."""
        return False

    @property
    def supports_keyboard(self) -> bool:
        """True if the target accepts keyboard / key-combo input."""
        return False

    @property
    def supports_absolute_move(self) -> bool:
        """True if the target supports move_cursor_to(x, y) instead of velocity HID."""
        return False

    # ── screen capture ──────────────────────────────────────────────────

    @abstractmethod
    def get_new_frame(self) -> Optional[FramePacket]:
        """Capture the current screen and return a FramePacket."""
        ...

    # ── motor commands ──────────────────────────────────────────────────

    @abstractmethod
    def send_hid(self, vx: float, vy: float, contact: bool, button_mask: int = 1) -> None:
        """Issue a velocity-based HID command (always supported)."""
        ...

    def move_cursor_to(self, x: float, y: float) -> None:
        """Move cursor to an absolute position.  Only when supports_absolute_move."""
        raise NotImplementedError(f"{type(self).__name__} does not support absolute cursor move")

    def click(self, x: float, y: float, button: str = "left") -> None:
        """Click at an absolute position.  Only when supports_absolute_move."""
        raise NotImplementedError(f"{type(self).__name__} does not support click")

    def send_key(self, key: str, modifiers: list[str] | None = None) -> None:
        """Send a keyboard key event.  Only when supports_keyboard."""
        raise NotImplementedError(f"{type(self).__name__} does not support keyboard input")

    def send_text(self, text: str) -> None:
        """Type a string of text.  Only when supports_keyboard."""
        raise NotImplementedError(f"{type(self).__name__} does not support text input")

    def get_cursor_position(self) -> Optional[tuple[float, float]]:
        """Direct cursor readout.  Returns None when not observable."""
        return None

    # ── polling (sensible defaults) ─────────────────────────────────────

    def get_hid_ack(self) -> Optional[dict]:
        """Return acknowledgement of the last motor command, if any."""
        return None

    def get_task_instruction(self) -> Optional[str]:
        """Poll for a new task instruction, if any."""
        return None

    # ── streaming I/O (sensible defaults) ────────────────────────────

    def get_text_input(self) -> Optional[str]:
        """Poll for incremental text input (typed, pasted, dictated).
        Returns None if no new text. Can return partial chunks."""
        return None

    def get_audio_input(self) -> Optional[dict]:
        """Poll for audio input frame.
        Returns None if no audio. Otherwise dict with:
          'samples': list[float] or bytes — raw PCM samples
          'sample_rate': int
          't_ns': int — capture timestamp
        """
        return None

    def send_text_output(self, text: str) -> None:
        """Emit text response to the user (chat, status, narration)."""
        pass

    def send_speech_output(self, text: str, **kwargs) -> None:
        """Emit speech output (text-to-speech). kwargs may include voice, rate, etc."""
        pass


class DesktopTarget(GUITarget):
    """Base class for desktop (mouse + keyboard) GUI surfaces."""

    @property
    def has_direct_cursor(self) -> bool:
        return True

    @property
    def supports_keyboard(self) -> bool:
        return True

    @property
    def supports_absolute_move(self) -> bool:
        return True

    def send_hid(self, vx: float, vy: float, contact: bool, button_mask: int = 1) -> None:
        # Desktop targets use absolute positioning; velocity HID is a no-op fallback.
        pass
