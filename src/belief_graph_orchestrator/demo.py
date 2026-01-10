
from __future__ import annotations

from .runtime import Brain
from .backends.mock import MockPhone
from .backends.mock_desktop import MockDesktop


def main(desktop: bool = False) -> None:
    if desktop:
        brain = Brain(target_key="mock-desktop", target_instance=MockDesktop("mock-desktop"), use_metadata_hints=True)
    else:
        brain = Brain("mock-key", target_cls=MockPhone, use_metadata_hints=True)
    for i in range(120):
        brain.step()
        if i % 10 == 0:
            print(f"step={i}", brain.summary())
    print("final:", brain.summary())


if __name__ == "__main__":
    main()
