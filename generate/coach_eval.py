"""Oracle-pure vs heuristic-advanced win-rate eval for coach packages.

Used after training (write ``tasks[0].metadata.coach_eval``) and by
``tools/compare_oracle_vs_advanced.py``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from generate.task_package import TaskPackage, _find_ckpt, _oracle_dir_for, _read_json

ORACLE_SPEC = {
    "policy_id": 7,
    "name": "oracle_pure",
    "kind": "oracle",
    "weight": 1.0,
    "epsilon": 0.0,
}
ADVANCED_SPEC = {
    "policy_id": 4,
    "name": "heuristic_advanced",
    "kind": "heuristic",
    "weight": 1.0,
    "heuristic": "advanced",
}

BEST_ORACLE = "oracle_pure"
BEST_ADVANCED = "heuristic_advanced"
DEFAULT_TIE_EPS = 0.02
# Cap default batch width to limit VRAM; override via parallel_envs.
DEFAULT_PARALLEL_ENVS = 32


def win_rate_from_counts(counts: dict[str, int]) -> float:
    wins = int(counts.get("win", 0))
    losses = int(counts.get("loss", 0))
    decisive = wins + losses
    if decisive <= 0:
        return 0.0
    return wins / decisive


def resolve_best_policy(
    *,
    oracle_wr: float,
    advanced_wr: float,
    tie_eps: float = DEFAULT_TIE_EPS,
) -> str:
    """Pick best teacher; prefer oracle when within ``tie_eps`` (inclusive)."""

    delta = float(oracle_wr) - float(advanced_wr)
    if abs(delta) <= float(tie_eps):
        return BEST_ORACLE
    return BEST_ORACLE if delta > 0 else BEST_ADVANCED


def task_package_from_leaf(package_dir: Path) -> TaskPackage:
    """Build a ``TaskPackage`` from a coach leaf directory."""

    package_dir = Path(package_dir).resolve()
    task_json = package_dir / "task.json"
    if not task_json.is_file():
        raise FileNotFoundError(f"task.json missing under {package_dir}")
    oracle_dir = _oracle_dir_for(package_dir)
    if oracle_dir is None:
        raise FileNotFoundError(f"No oracle weights under {package_dir}")
    bank = _read_json(task_json)
    task = bank["tasks"][0]
    meta_path = package_dir / "meta.json"
    if meta_path.is_file():
        meta = _read_json(meta_path)
        task_index = int(meta.get("task_index", 0))
        task_id = str(meta.get("task_id", task.get("task_id", package_dir.name)))
    else:
        task_index = 0
        task_id = str(task.get("task_id", package_dir.name))
    return TaskPackage(
        path=package_dir,
        root=package_dir.parent,
        task_index=task_index,
        task_id=task_id,
        task_json=task_json,
        oracle_dir=oracle_dir,
        oracle_config=oracle_dir / "config.json",
        oracle_ckpt=_find_ckpt(oracle_dir),
    )


def _init_last_visible(n_ally: int):
    from src.tabx.heuristic_policy import LastVisibleTarget

    return [LastVisibleTarget() for _ in range(n_ally)]


def _build_batched_episode_fn(
    *,
    ctx: Any,
    spec_dict: dict[str, Any],
    max_episode_steps: int,
):
    """Return ``jit(vmap(run_one_episode))`` over a batch of episode seeds."""

    import jax
    import jax.numpy as jnp
    from generate.behavior_mix import BehaviorSpec
    from generate.policies import act_shared_jax
    from src.tabx.eval_task import load_heuristic_params

    spec = BehaviorSpec(
        policy_id=int(spec_dict["policy_id"]),
        name=str(spec_dict["name"]),
        kind=str(spec_dict["kind"]),
        weight=float(spec_dict.get("weight", 1.0)),
        heuristic=spec_dict.get("heuristic"),
        epsilon=None
        if spec_dict.get("epsilon") is None
        else float(spec_dict["epsilon"]),
    )
    env = ctx.env
    env_params = ctx.env_params
    ally_keys = list(ctx.ally_keys)
    n_ally = len(ally_keys)
    n_agents = int(ctx.n_units_total)
    max_n_zone = int(ctx.env.max_n_zone)
    oracle = ctx.oracle
    heur_params = None
    if spec.kind == "heuristic":
        if not spec.heuristic:
            raise ValueError(f"heuristic spec missing preset: {spec}")
        heur_params = load_heuristic_params(
            spec.heuristic, epsilon_override=spec.epsilon
        )
    max_steps = int(max_episode_steps)

    def run_one_episode(ep_key: jax.Array):
        """One episode → (is_win, truncated, finished) uint8 scalars."""

        ep_key, reset_key = jax.random.split(ep_key)
        obs, state = env.reset(reset_key, env_params)
        last_visible = _init_last_visible(n_ally)

        def body(carry, _t):
            obs_i, state_i, key_i, last_i, done_i, win_i, trunc_i = carry

            def step_fn(operands):
                obs_s, state_s, key_s, last_s, win_s, trunc_s = operands
                key_s, bkey, skey = jax.random.split(key_s, 3)
                avail = env.get_avail_actions(state_s)
                behavior, new_last, _ = act_shared_jax(
                    key=bkey,
                    spec=spec,
                    oracle=oracle,
                    obs_by_agent=obs_s,
                    avail_by_agent=avail,
                    ally_keys=ally_keys,
                    n_agents=n_agents,
                    max_n_zone=max_n_zone,
                    physics_params=state_s["physics_params"],
                    heuristic_params=heur_params,
                    last_visible=last_s,
                )
                obs_n, state_n, _rew, dones, info = env.step(
                    skey, state_s, dict(behavior)
                )
                ep_done = jnp.asarray(dones["__all__"]).reshape(()).astype(jnp.bool_)
                is_win = jnp.asarray(info["is_win"]).reshape(-1)[0].astype(jnp.bool_)
                trunc = jnp.asarray(info["truncation"]).reshape(-1)[0].astype(jnp.bool_)
                # Latch terminal flags on the finishing transition.
                win_out = jnp.where(ep_done, is_win, win_s)
                trunc_out = jnp.where(ep_done, trunc, trunc_s)
                return (
                    obs_n,
                    state_n,
                    key_s,
                    new_last,
                    ep_done,
                    win_out,
                    trunc_out,
                )

            def skip_fn(operands):
                obs_s, state_s, key_s, last_s, win_s, trunc_s = operands
                return (
                    obs_s,
                    state_s,
                    key_s,
                    last_s,
                    jnp.bool_(True),
                    win_s,
                    trunc_s,
                )

            new_carry = jax.lax.cond(
                done_i,
                skip_fn,
                step_fn,
                (obs_i, state_i, key_i, last_i, win_i, trunc_i),
            )
            return new_carry, None

        init = (
            obs,
            state,
            ep_key,
            last_visible,
            jnp.bool_(False),
            jnp.bool_(False),
            jnp.bool_(False),
        )
        (obs_f, state_f, key_f, last_f, done_f, win_f, trunc_f), _ = jax.lax.scan(
            body, init, jnp.arange(max_steps, dtype=jnp.int32)
        )
        del obs_f, state_f, key_f, last_f
        # Never finished → count as truncated loss.
        finished = done_f
        trunc_f = jnp.where(finished, trunc_f, jnp.bool_(True))
        win_f = jnp.where(finished, win_f, jnp.bool_(False))
        return (
            win_f.astype(jnp.uint8),
            trunc_f.astype(jnp.uint8),
            finished.astype(jnp.uint8),
        )

    return jax.jit(jax.vmap(run_one_episode))


def run_policy_episodes(
    *,
    ctx: Any,
    spec_dict: dict[str, Any],
    num_episodes: int,
    seed: int,
    max_episode_steps: int,
    parallel_envs: int | None = None,
) -> dict[str, int]:
    """Roll out episodes with ``vmap`` over parallel envs; score via ``is_win``."""

    import jax
    import numpy as np

    n = max(1, int(num_episodes))
    if parallel_envs is None:
        batch = min(n, DEFAULT_PARALLEL_ENVS)
    else:
        batch = max(1, min(n, int(parallel_envs)))

    batched_fn = _build_batched_episode_fn(
        ctx=ctx,
        spec_dict=spec_dict,
        max_episode_steps=max_episode_steps,
    )
    key = jax.random.key(int(seed) % (2**32))
    wins = losses = draws = truncated = 0
    episodes = 0

    for start in range(0, n, batch):
        valid = min(batch, n - start)
        key, sub = jax.random.split(key)
        ep_keys = jax.random.split(sub, batch)
        win_u8, trunc_u8, finished_u8 = batched_fn(ep_keys)
        win_np = np.asarray(win_u8[:valid], dtype=np.uint8)
        trunc_np = np.asarray(trunc_u8[:valid], dtype=np.uint8)
        # finished is informational; unfinished already forced to trunc loss.
        del finished_u8
        for w, t in zip(win_np.tolist(), trunc_np.tolist()):
            if int(w):
                wins += 1
            else:
                losses += 1
            episodes += 1
            if int(t):
                truncated += 1

    return {
        "win": wins,
        "draw": draws,
        "loss": losses,
        "episodes": episodes,
        "truncated": truncated,
    }


def evaluate_package_oracle_vs_advanced(
    package: TaskPackage,
    *,
    num_episodes: int = 64,
    seed: int = 0,
    max_episode_steps: int = 512,
    tie_eps: float = DEFAULT_TIE_EPS,
    parallel_envs: int | None = None,
    ctx: Any | None = None,
    train_like_sample: bool = False,
) -> dict[str, Any]:
    """Compare oracle_pure vs heuristic_advanced; include ``best_policy``.

    Pass an existing ``RecordGenContext`` as ``ctx`` to reuse env/oracle (e.g.
    inside ``record_worker`` before generate).
    """

    if ctx is None:
        from generate.env_centric import build_record_gen_context

        ctx = build_record_gen_context(
            package, train_like_sample=bool(train_like_sample)
        )
    n = max(1, int(num_episodes))
    if parallel_envs is None:
        batch = min(n, DEFAULT_PARALLEL_ENVS)
    else:
        batch = max(1, min(n, int(parallel_envs)))

    oracle_counts = run_policy_episodes(
        ctx=ctx,
        spec_dict=ORACLE_SPEC,
        num_episodes=num_episodes,
        seed=seed,
        max_episode_steps=max_episode_steps,
        parallel_envs=batch,
    )
    advanced_counts = run_policy_episodes(
        ctx=ctx,
        spec_dict=ADVANCED_SPEC,
        num_episodes=num_episodes,
        seed=seed + 10_000_003,
        max_episode_steps=max_episode_steps,
        parallel_envs=batch,
    )
    o_wr = win_rate_from_counts(oracle_counts)
    a_wr = win_rate_from_counts(advanced_counts)
    best = resolve_best_policy(oracle_wr=o_wr, advanced_wr=a_wr, tie_eps=tie_eps)
    return {
        "package_name": package.name,
        "task_id": package.task_id,
        "task_index": int(package.task_index),
        "package_path": str(package.path),
        "enemy_heuristic": (ctx.manifest or {}).get("heuristic"),
        "num_episodes": int(num_episodes),
        "max_episode_steps": int(max_episode_steps),
        "parallel_envs": int(batch),
        "seed": int(seed),
        "tie_eps": float(tie_eps),
        "oracle_pure": {**oracle_counts, "win_rate": o_wr},
        "heuristic_advanced": {**advanced_counts, "win_rate": a_wr},
        "delta_oracle_minus_advanced": o_wr - a_wr,
        "best_policy": best,
        "train_like_sample": bool(getattr(ctx.oracle, "train_like_sample", False)),
        "train_update_steps": getattr(ctx.oracle, "train_update_steps", None),
        "train_update_source": getattr(ctx.oracle, "train_update_source", ""),
        "train_epsilon": float(getattr(ctx.oracle, "train_epsilon", 0.0) or 0.0),
        "oracle_act": (
            "train_like"
            if getattr(ctx.oracle, "train_like_sample", False)
            else "greedy"
        ),
    }


def coach_eval_for_task_metadata(eval_result: dict[str, Any]) -> dict[str, Any]:
    """Subset stored under ``tasks[0].metadata.coach_eval``."""

    return {
        "num_episodes": int(eval_result["num_episodes"]),
        "max_episode_steps": int(eval_result["max_episode_steps"]),
        "parallel_envs": int(
            eval_result.get("parallel_envs", DEFAULT_PARALLEL_ENVS)
        ),
        "seed": int(eval_result["seed"]),
        "tie_eps": float(eval_result.get("tie_eps", DEFAULT_TIE_EPS)),
        "oracle_pure": dict(eval_result["oracle_pure"]),
        "heuristic_advanced": dict(eval_result["heuristic_advanced"]),
        "delta_oracle_minus_advanced": float(
            eval_result["delta_oracle_minus_advanced"]
        ),
        "best_policy": str(eval_result["best_policy"]),
        "oracle_act": str(eval_result.get("oracle_act", "greedy")),
        "train_like_sample": bool(eval_result.get("train_like_sample", False)),
        "train_update_steps": eval_result.get("train_update_steps"),
        "train_update_source": str(eval_result.get("train_update_source") or ""),
        "train_epsilon": float(eval_result.get("train_epsilon") or 0.0),
    }


def write_coach_eval_to_task_json(
    task_json: Path,
    coach_eval: dict[str, Any],
) -> None:
    """Atomically write ``coach_eval`` into ``tasks[0].metadata``."""

    task_json = Path(task_json)
    bank = _read_json(task_json)
    tasks = bank.get("tasks")
    if not tasks:
        raise ValueError(f"No tasks in {task_json}")
    meta = dict(tasks[0].get("metadata") or {})
    meta["coach_eval"] = coach_eval
    tasks[0]["metadata"] = meta
    bank["tasks"] = tasks
    text = json.dumps(bank, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    parent = task_json.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(parent),
        prefix=f".{task_json.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(task_json)


def read_coach_eval(package: TaskPackage | Path) -> dict[str, Any] | None:
    """Return ``tasks[0].metadata.coach_eval`` if present."""

    if isinstance(package, TaskPackage):
        task_json = package.task_json
    else:
        task_json = Path(package) / "task.json"
        if not task_json.is_file():
            task_json = Path(package)
    if not task_json.is_file():
        return None
    bank = _read_json(task_json)
    tasks = bank.get("tasks") or []
    if not tasks:
        return None
    meta = tasks[0].get("metadata") or {}
    eval_block = meta.get("coach_eval")
    return dict(eval_block) if isinstance(eval_block, dict) else None


def read_best_policy(
    package: TaskPackage | Path,
    *,
    default: str = BEST_ORACLE,
) -> str:
    """Best teacher from task.json; fall back to ``default`` if missing."""

    eval_block = read_coach_eval(package)
    if not eval_block:
        return default
    best = str(eval_block.get("best_policy") or default)
    if best not in {BEST_ORACLE, BEST_ADVANCED}:
        return default
    return best


def evaluate_and_write_coach_leaf(
    package_dir: Path,
    *,
    num_episodes: int = 64,
    seed: int = 0,
    max_episode_steps: int = 512,
    tie_eps: float = DEFAULT_TIE_EPS,
    parallel_envs: int | None = None,
    train_like_sample: bool = False,
) -> dict[str, Any]:
    """Eval one coach leaf and write ``coach_eval`` into its ``task.json``."""

    package = task_package_from_leaf(package_dir)
    result = evaluate_package_oracle_vs_advanced(
        package,
        num_episodes=num_episodes,
        seed=seed,
        max_episode_steps=max_episode_steps,
        tie_eps=tie_eps,
        parallel_envs=parallel_envs,
        train_like_sample=bool(train_like_sample),
    )
    write_coach_eval_to_task_json(
        package.task_json, coach_eval_for_task_metadata(result)
    )
    return result


def ensure_coach_eval(
    package: TaskPackage,
    *,
    ctx: Any | None = None,
    num_episodes: int = 64,
    seed: int = 0,
    max_episode_steps: int = 512,
    tie_eps: float = DEFAULT_TIE_EPS,
    parallel_envs: int | None = None,
    force: bool = False,
    train_like_sample: bool = False,
) -> dict[str, Any]:
    """Return ``coach_eval`` for ``package``, evaluating+writing if missing.

    Used by parallel record generation when ``best_teacher_reference`` is on but
    the coach leaf was trained without ``COACH_EVAL`` / never wrote ``coach_eval``.

    Returns a dict with:
      - ``coach_eval``: metadata block
      - ``best_policy``: ``oracle_pure`` | ``heuristic_advanced``
      - ``wrote``: whether ``task.json`` was updated this call
      - ``eval_result``: full eval row when a new eval ran, else ``None``
    """

    existing = None if force else read_coach_eval(package)
    if existing is not None:
        best = str(existing.get("best_policy") or BEST_ORACLE)
        if best not in {BEST_ORACLE, BEST_ADVANCED}:
            best = BEST_ORACLE
        return {
            "coach_eval": existing,
            "best_policy": best,
            "wrote": False,
            "eval_result": None,
        }

    result = evaluate_package_oracle_vs_advanced(
        package,
        num_episodes=num_episodes,
        seed=seed,
        max_episode_steps=max_episode_steps,
        tie_eps=tie_eps,
        parallel_envs=parallel_envs,
        ctx=ctx,
        train_like_sample=bool(train_like_sample),
    )
    block = coach_eval_for_task_metadata(result)
    write_coach_eval_to_task_json(package.task_json, block)
    return {
        "coach_eval": block,
        "best_policy": str(block["best_policy"]),
        "wrote": True,
        "eval_result": result,
    }
