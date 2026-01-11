from ..target import GUITarget, DesktopTarget, UIElement
from .mock import MockPhone
from .mock_desktop import MockDesktop
from .real import RealPhone
from .simulator import SimulatorPhone

__all__ = [
    "GUITarget",
    "DesktopTarget",
    "UIElement",
    "MockPhone",
    "MockDesktop",
    "RealPhone",
    "SimulatorPhone",
]

# PlaywrightPhone and PlaywrightDesktop have heavy optional deps — import explicitly:
#   from belief_graph_orchestrator.backends.playwright import PlaywrightPhone
#   from belief_graph_orchestrator.backends.playwright_desktop import PlaywrightDesktop
