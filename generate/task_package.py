"""Discover and pack per-task packages (task.json + oracle checkpoint).

Preferred layout (written by ``marl_baseline``): each coach run leaf is
self-contained::

    {coach_root}/.../task-000000-<id>/seed-<seed>/
      task.json
      meta.json
      config.json
      best.safetensors
      final.safetensors

Legacy packed layout still works::

    {root}/task_00000_<id>/
      task.json
      oracle/{config.json, best.safetensors}
      meta.json
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from src.tabx.sample_task import load_task_bank, save_task_bank


@dataclass(frozen=True)
class TaskPackage:
    path: Path
    root: Path
    task_index: int
    task_id: str
    task_json: Path
    oracle_dir: Path
    oracle_config: Path
    oracle_ckpt: Path

    @property
    def name(self) -> str:
        """Unique id under ``root`` (stable for worker lookup / jobs)."""

        rel = self.path.resolve().relative_to(self.root.resolve())
        return "__".join(rel.parts) if rel.parts else self.path.name


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_ckpt(oracle_dir: Path) -> Path:
    for name in ("best.safetensors", "final.safetensors"):
        candidate = oracle_dir / name
        if candidate.exists():
            return candidate
    matches = sorted(oracle_dir.glob("*.safetensors"))
    if not matches:
        raise FileNotFoundError(f"No safetensors checkpoint under {oracle_dir}")
    return matches[0]


def _oracle_dir_for(path: Path) -> Path | None:
    """Return directory that holds coach weights + config, if any."""

    nested = path / "oracle"
    if nested.is_dir() and (
        list(nested.glob("*.safetensors")) or (nested / "config.json").exists()
    ):
        return nested
    if list(path.glob("*.safetensors")):
        return path
    return None


def _is_package_leaf(path: Path) -> bool:
    return (path / "task.json").is_file() and _oracle_dir_for(path) is not None


def _iter_package_dirs(root: Path) -> Iterator[Path]:
    """Yield self-contained package directories under ``root`` (recursive)."""

    # Prefer deeper leaves: skip a dir if a descendant is also a package leaf.
    candidates = [
        path
        for path in sorted(root.rglob("task.json"))
        if path.is_file() and _is_package_leaf(path.parent)
    ]
    dirs = [path.parent.resolve() for path in candidates]
    for directory in dirs:
        if any(
            other != directory and directory in other.parents for other in dirs
        ):
            continue
        yield directory


def discover_task_packages(
    root: str | Path,
    *,
    seed: int | None = None,
) -> list[TaskPackage]:
    """Scan ``root`` for self-contained task+coach packages."""

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"coach_root / task_packages_root not found: {root}")

    packages: list[TaskPackage] = []
    for path in _iter_package_dirs(root):
        task_json = path / "task.json"
        oracle_dir = _oracle_dir_for(path)
        assert oracle_dir is not None
        meta_path = path / "meta.json"
        meta = _read_json(meta_path) if meta_path.exists() else {}
        if seed is not None:
            meta_seed = meta.get("seed")
            if meta_seed is not None and int(meta_seed) != int(seed):
                continue
            # Flat coach leaf named seed-N / nested seed dir.
            if meta_seed is None and f"seed-{seed}" not in path.name and f"seed_{seed}" not in str(path):
                continue
        bank = _read_json(task_json)
        tasks = bank.get("tasks") or [bank]
        task = tasks[0]
        task_id = str(meta.get("task_id") or task.get("task_id") or path.name)
        task_index = int(meta.get("task_index", _infer_index_from_path(path)))
        config_path = oracle_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing oracle config: {config_path}")
        packages.append(
            TaskPackage(
                path=path,
                root=root,
                task_index=task_index,
                task_id=task_id,
                task_json=task_json,
                oracle_dir=oracle_dir,
                oracle_config=config_path,
                oracle_ckpt=_find_ckpt(oracle_dir),
            )
        )
    if not packages:
        raise FileNotFoundError(
            f"No valid task packages under {root} "
            "(need task.json + *.safetensors, or task.json + oracle/)"
        )
    packages.sort(key=lambda package: (package.task_index, package.name))
    return packages


def _infer_index_from_path(path: Path) -> int:
    for part in reversed(path.parts):
        match = re.search(r"task[_-](\d+)", part)
        if match:
            return int(match.group(1))
    return 0


def load_package_task_bank(package: TaskPackage) -> dict[str, Any]:
    return load_task_bank(package.task_json)


def iter_ckpt_candidates(
    ckpt_root: Path,
    *,
    task_index: int,
    task_id: str,
    algorithm: str | None,
    seed: int | None,
) -> Iterator[Path]:
    """Yield likely checkpoint directories under a training save tree."""

    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
    patterns = [
        f"**/task-{task_index:06d}-{safe_id}/seed-*/",
        f"**/task-{task_index:06d}-*/seed-*/",
        f"**/task_{task_index:05d}_seed_*/",
        f"**/task_{task_index:05d}_*/",
    ]
    if algorithm:
        patterns = [f"**/{algorithm}/" + p.lstrip("**/") for p in patterns] + patterns
    seen: set[Path] = set()
    for pattern in patterns:
        for match in ckpt_root.glob(pattern):
            if not match.is_dir():
                continue
            resolved = match.resolve()
            if resolved in seen:
                continue
            if seed is not None and f"seed-{seed}" not in match.name and f"seed_{seed}" not in match.name:
                # Allow nested seed dirs.
                if not any(match.glob(f"**/seed-{seed}")) and not (match / f"seed-{seed}").exists():
                    if f"seed_{seed}" not in str(match):
                        continue
            seen.add(resolved)
            yield match


def resolve_oracle_dir(
    ckpt_root: Path,
    *,
    task_index: int,
    task_id: str,
    algorithm: str | None,
    seed: int | None,
) -> Path:
    for candidate in iter_ckpt_candidates(
        ckpt_root,
        task_index=task_index,
        task_id=task_id,
        algorithm=algorithm,
        seed=seed,
    ):
        # Prefer leaf that already contains safetensors.
        if list(candidate.glob("*.safetensors")):
            return candidate
        nested = sorted(candidate.glob("**/best.safetensors"))
        if nested:
            return nested[0].parent
        nested = sorted(candidate.glob("**/*.safetensors"))
        if nested:
            return nested[0].parent
    raise FileNotFoundError(
        f"No checkpoint found for task_index={task_index} task_id={task_id!r} under {ckpt_root}"
    )


def pack_task_packages(
    *,
    task_bank_path: str | Path,
    ckpt_root: str | Path,
    output_root: str | Path,
    algorithm: str | None = None,
    seed: int | None = None,
    task_indices: list[int] | None = None,
) -> list[Path]:
    """Pack each task + matching oracle ckpt into a task package directory.

    Legacy helper for ckpt trees that were trained before ``task.json`` was
    written next to weights. Prefer pointing generate-record at ``coach_root``
    directly when trainers already emit self-contained leaves.
    """

    bank = load_task_bank(task_bank_path)
    ckpt_root = Path(ckpt_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for index, task in enumerate(bank["tasks"]):
        if task_indices is not None and index not in task_indices:
            continue
        task_id = str(task.get("task_id", f"task_{index:06d}"))
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
        pkg_dir = output_root / f"task_{index:05d}_{safe_id}"
        pkg_dir.mkdir(parents=True, exist_ok=True)

        save_task_bank(
            pkg_dir / "task.json",
            [task],
            seed=int(bank["manifest"].get("seed", 0)),
            physics=str(bank["manifest"].get("physics", "default")),
            heuristic=str(bank["manifest"].get("heuristic", "medium")),
            max_n_ally=int(bank["manifest"]["schema"]["max_n_ally"]),
            max_n_enemy=int(bank["manifest"]["schema"]["max_n_enemy"]),
            max_n_zone=int(bank["manifest"]["schema"]["max_n_zone"]),
            filter_protocol=bank["manifest"].get("filter_protocol"),
        )

        oracle_src = resolve_oracle_dir(
            ckpt_root,
            task_index=index,
            task_id=task_id,
            algorithm=algorithm,
            seed=seed,
        )
        oracle_dst = pkg_dir / "oracle"
        oracle_dst.mkdir(parents=True, exist_ok=True)
        for item in oracle_src.iterdir():
            if item.is_file() and (
                item.suffix == ".safetensors"
                or item.name in {"config.json", "trainer_config.json", "task.json", "meta.json"}
            ):
                shutil.copy2(item, oracle_dst / item.name)

        # Also keep flat copies at package root for the preferred layout.
        for name in ("task.json", "meta.json"):
            src = oracle_src / name
            if src.exists() and not (pkg_dir / name).exists():
                shutil.copy2(src, pkg_dir / name)

        config_dst = oracle_dst / "config.json"
        if not config_dst.exists():
            # Fall back to trainer_config or synthesize minimal config.
            alt = oracle_dst / "trainer_config.json"
            if alt.exists():
                alt.replace(config_dst)
            else:
                config_dst.write_text(
                    json.dumps(
                        {
                            "algorithm": algorithm or "mappo",
                            "HIDDEN_SIZE": 128,
                            "task_index": index,
                            "task_id": task_id,
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )

        meta = {
            "task_index": index,
            "task_id": task_id,
            "source_task_bank": str(Path(task_bank_path).resolve()),
            "source_oracle_dir": str(oracle_src.resolve()),
            "algorithm": algorithm,
            "seed": seed,
        }
        (pkg_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written.append(pkg_dir)
    return written
