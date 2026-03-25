from __future__ import annotations

from typing import Any, Protocol

from schemas import OrcaAction


class OrcaExecutor(Protocol):
    """
    The interface OrcaSlicer needs to implement/map (for integration).

    Since OrcaSlicer is not part of this repository, this file only provides
    an "action -> method" contract example.
    """

    def load(self, args: dict[str, Any]) -> None: ...

    def load_from_net(self, args: dict[str, Any]) -> None: ...

    def slice_current(self, args: dict[str, Any]) -> None: ...

    def slice_all(self, args: dict[str, Any]) -> None: ...


def apply_actions(executor: OrcaExecutor, actions: list[OrcaAction]) -> None:
    """
    Reference executor implementation on the OrcaSlicer side:
    call the corresponding method for each action.
    """
    op_to_method = {
        "load": executor.load,
        "load_from_net": executor.load_from_net,
        "slice_current": executor.slice_current,
        "slice_all": executor.slice_all,
    }

    for a in actions:
        fn = op_to_method.get(a.action)
        if not fn:
            raise RuntimeError(f"Unsupported action: {a.action}")
        fn(a.params)

