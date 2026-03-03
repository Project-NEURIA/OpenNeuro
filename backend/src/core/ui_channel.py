"""UI channel types for bidirectional component-to-frontend communication.

UISender/UIReceiver extend Sender/Receiver — component code uses .send() and
``for item in receiver(self):`` unchanged.  The concrete class name (returned by
the API as e.g. ``"UIVideoSender"``) tells the frontend which widget to render.
"""

from __future__ import annotations

from src.core.channel import Receiver, Sender
from src.core.frames import TextFrame


# -- marker bases --

class UISender[T](Sender[T]):
    """Marker base: data flows from component to the frontend node UI."""


class UIReceiver[T](Receiver[T]):
    """Marker base: data flows from the frontend node UI to the component."""


# -- concrete output types (backend -> frontend) --

class UITextSender(UISender[TextFrame]):
    """Component sends text for display in the node UI."""


class UIVideoSender(UISender[bytes]):
    """Component sends JPEG bytes for display in the node UI."""


# -- concrete input types (frontend -> backend) --

class UITextReceiver(UIReceiver[TextFrame]):
    """Component receives text typed by the user in the node UI."""


class UIKeystrokeReceiver(UIReceiver[TextFrame]):
    """Component receives individual keystrokes from the node UI."""
