from belief_graph_orchestrator.runtime import Brain
from belief_graph_orchestrator.backends.mock_desktop import MockDesktop
from belief_graph_orchestrator.backends.mock import MockPhone
from belief_graph_orchestrator.target import GUITarget, DesktopTarget


def test_desktop_brain_smoke():
    desktop = MockDesktop("mock-desktop")
    brain = Brain(target_key="mock-desktop", target_instance=desktop, use_metadata_hints=True)
    for _ in range(20):
        brain.step()
    s = brain.summary()
    assert s["num_events"] > 0
    assert s["num_nodes"] > 0


def test_desktop_capabilities():
    desktop = MockDesktop("test")
    assert desktop.has_direct_cursor is True
    assert desktop.supports_keyboard is True
    assert desktop.supports_absolute_move is True


def test_phone_capabilities():
    phone = MockPhone("test")
    assert phone.has_direct_cursor is False
    assert phone.supports_keyboard is False
    assert phone.supports_absolute_move is False


def test_gui_target_is_common_base():
    phone = MockPhone("test")
    desktop = MockDesktop("test")
    assert isinstance(phone, GUITarget)
    assert isinstance(desktop, GUITarget)
    assert isinstance(desktop, DesktopTarget)
    assert not isinstance(phone, DesktopTarget)


def test_desktop_direct_cursor():
    desktop = MockDesktop("test")
    desktop.move_cursor_to(100.0, 200.0)
    pos = desktop.get_cursor_position()
    assert pos == (100.0, 200.0)


def test_desktop_click_navigates():
    desktop = MockDesktop("test", width=1280, height=800)
    assert desktop.current_screen == "main"
    # Click the "Submit" button — its bbox starts at (230, 200) when sidebar is open
    desktop.click(300.0, 225.0)
    assert desktop.current_screen == "form"


def test_desktop_send_text():
    desktop = MockDesktop("test")
    desktop.send_text("hello")
    assert desktop.text_buffer == "hello"
    desktop.send_text(" world")
    assert desktop.text_buffer == "hello world"


def test_desktop_send_key_escape():
    desktop = MockDesktop("test")
    desktop.click(300.0, 225.0)  # go to form
    assert desktop.current_screen == "form"
    desktop.send_key("Escape")
    assert desktop.current_screen == "main"


def test_desktop_brain_pointer_uncertainty_zero():
    """On desktop the pointer should always be perfectly known."""
    desktop = MockDesktop("mock-desktop")
    brain = Brain(target_key="mock-desktop", target_instance=desktop, use_metadata_hints=True)
    for _ in range(10):
        brain.step()
    assert brain.state.pointer_uncertainty == 0.0


def test_desktop_absolute_tap_phases():
    """Desktop tap chunks should use move_to + click phases, not velocity approach."""
    desktop = MockDesktop("test")
    brain = Brain(target_key="test", target_instance=desktop, use_metadata_hints=True)
    for _ in range(30):
        brain.step()
    # Check that any action_issued event has absolute-style cmd
    action_events = [e for e in brain.journal.events if e.type == "action_issued"]
    assert len(action_events) > 0
    # At least some should have absolute command types
    abs_cmds = [e for e in action_events if isinstance(e.payload.get("cmd"), dict) and e.payload["cmd"].get("type") in ("move_to", "click", "key", "noop")]
    assert len(abs_cmds) > 0


def test_backward_compat_legacy_cls():
    """Brain(legacy cls/key aliases) still works via backward-compat params."""
    # exercises the deprecated legacy key/cls aliases
    brain = Brain(target_key="mock", target_cls=MockPhone, use_metadata_hints=True)
    for _ in range(5):
        brain.step()
    assert brain.state.num_events if hasattr(brain.state, "num_events") else len(brain.journal.events) > 0
