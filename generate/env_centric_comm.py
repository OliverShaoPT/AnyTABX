"""Env-centric generation that also dumps ``attack_target`` (comm schema).

Production path: same GPU workers as ``generate.parallel_records``, with
``dump_attack_target=true`` compiled into the scan (no monkeypatch).

  python -m generate.env_centric_comm --config generate/configs/record_gen.yaml
  python -m generate.env_centric_comm --coach_root ... --output_root ...

``python -m generate.env_centric`` / ``parallel_records`` stay on the old
schema unless ``dump_attack_target: true``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_one_record(*args: Any, **kwargs: Any) -> Path:
    """Same signature as ``generate.env_centric.generate_one_record``."""

    from generate.env_centric import generate_one_record as _orig

    kwargs.setdefault("dump_attack_target", True)
    return _orig(*args, **kwargs)


def generate_one_record_scan(*args: Any, **kwargs: Any) -> Path:
    from generate.scan_rollout import generate_one_record_scan as _orig

    kwargs.setdefault("dump_attack_target", True)
    return _orig(*args, **kwargs)


def build_scan_rollout_fn(*args: Any, **kwargs: Any):
    from generate.scan_rollout import build_scan_rollout_fn as _orig

    kwargs.setdefault("dump_attack_target", True)
    return _orig(*args, **kwargs)


def main(argv: list[str] | None = None) -> None:
    from generate.parallel_records import main as parallel_main

    parallel_main(argv, dump_attack_target=True)


if __name__ == "__main__":
    main()
