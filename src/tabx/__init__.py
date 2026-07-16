"""Public TABX API with lazy imports.

Keeping package initialization lightweight lets command-line modules configure
JAX before any backend or optional accelerator plugin is initialized.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = (
    "TABX",
    "TABXHeuristicParam",
    "UnitScenario",
    "VectorizedScenario",
    "ZoneScenario",
    "build_batched_env_params_and_config",
)

_EXPORT_MODULES = {
    "TABX": "src.tabx.tabx",
    "TABXHeuristicParam": "src.tabx.heuristic_policy.params",
    "UnitScenario": "src.tabx.scenarios.scenario",
    "VectorizedScenario": "src.tabx.scenarios.scenario",
    "ZoneScenario": "src.tabx.scenarios.scenario",
    "build_batched_env_params_and_config": "src.tabx.utils",
}


def __getattr__(name: str) -> Any:
    """Load public objects on first access while preserving the existing API."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
