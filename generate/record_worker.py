"""Spawned worker entry for parallel record generation.

Keep this module free of JAX imports at top-level so CUDA_VISIBLE_DEVICES /
JAX_PLATFORMS can be set before JAX initializes in each process.
"""

from __future__ import annotations

import os
import time
import traceback
from pathlib import Path
from typing import Any


def force_jax_cpu_env() -> None:
    """Debug-only: force host/CPU before any JAX import.

    Not used for mass record production (prefer ``force_jax_gpu_env``). With
    ``jax-cuda12-plugin`` installed, import still calls
    ``jax_plugins.xla_cuda12.initialize()`` → ``cuInit``; skip the CUDA
    constraints check so quiet CPU debug runs do not touch the driver.
    """

    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Avoid cuInit / version probe inside jax_plugins.xla_cuda12 (see jax#35105).
    os.environ["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"
    # Cap host BLAS threads so many workers do not oversubscribe cores.
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, "1")
    xla_flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_cpu_multi_thread_eigen=false" not in xla_flags:
        os.environ["XLA_FLAGS"] = (
            (xla_flags + " --xla_cpu_multi_thread_eigen=false").strip()
        )


def force_jax_gpu_env(gpu_id: int | str, *, jax_platform: str = "cuda") -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["JAX_PLATFORMS"] = str(jax_platform)
    os.environ.pop("JAX_PLATFORM_NAME", None)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _report(payload: dict[str, Any], event: dict[str, Any]) -> None:
    queue = payload.get("progress_queue")
    if queue is None:
        return
    queue.put(event)


def _emit_record_done(
    payload: dict[str, Any],
    *,
    worker_id: int,
    device: str,
    package_name: str,
    record_id: int,
    path: str,
    generate_s: float,
    parallel_envs: int,
) -> None:
    _report(
        payload,
        {
            "event": "record_done",
            "worker_id": worker_id,
            "device": device,
            "gpu_id": payload.get("gpu_id"),
            "package_name": package_name,
            "record_id": int(record_id),
            "path": str(path),
            "generate_s": round(generate_s, 3),
        },
    )
    if payload.get("progress_queue") is None:
        print(
            f"[record_worker] worker={worker_id} device={device} "
            f"gpu={payload.get('gpu_id')} wrote {path} "
            f"generate_s~={generate_s:.1f} parallel_envs={parallel_envs}",
            flush=True,
        )


def _generate_ids(
    *,
    package,
    record_ids: list[int],
    payload: dict[str, Any],
    ctx,
    rollout_fn,
    mix,
    scan_rollout: bool,
    parallel_envs: int,
    total_timesteps: int,
    min_behavior_steps: int,
    behavior_switch_prob: float,
    worker_id: int,
    device: str,
    adapt: dict[str, Any] | None,
    paths: list[str],
    output_root: Path | str | None = None,
    report_progress: bool = True,
) -> None:
    from generate.env_centric import generate_one_record
    from generate.scan_rollout import generate_records_scan_batch

    if not record_ids:
        return
    root = Path(output_root) if output_root is not None else Path(payload["output_root"])
    independent = bool(payload.get("independent_ally_policies", False))
    mid_switch = bool(payload.get("mid_episode_policy_switch", True))
    best_teacher_reference = bool(payload.get("best_teacher_reference", False))
    best_policy = str(payload.get("best_policy") or "oracle_pure")
    dump_attack_target = bool(payload.get("dump_attack_target", False))
    if scan_rollout and parallel_envs > 1:
        t_gen = time.perf_counter()
        batch_paths = generate_records_scan_batch(
            package,
            record_ids=record_ids,
            output_root=root,
            total_timesteps=total_timesteps,
            seed=payload.get("seed"),
            min_behavior_steps=min_behavior_steps,
            behavior_switch_prob=behavior_switch_prob,
            behavior_mix=mix,
            ctx=ctx,
            rollout_fn=rollout_fn,
            parallel_envs=parallel_envs,
            adapt=adapt,
            independent_ally_policies=independent,
            mid_episode_policy_switch=mid_switch,
            best_teacher_reference=best_teacher_reference,
            best_policy=best_policy,
            dump_attack_target=dump_attack_target,
        )
        generate_s_total = time.perf_counter() - t_gen
        per_rec = generate_s_total / max(len(batch_paths), 1)
        for path, record_id in zip(batch_paths, record_ids):
            paths.append(str(path))
            if report_progress:
                _emit_record_done(
                    payload,
                    worker_id=worker_id,
                    device=device,
                    package_name=package.name,
                    record_id=int(record_id),
                    path=str(path),
                    generate_s=per_rec,
                    parallel_envs=parallel_envs,
                )
        return

    for record_id in record_ids:
        t_gen = time.perf_counter()
        if scan_rollout:
            from generate.scan_rollout import generate_one_record_scan

            path = generate_one_record_scan(
                package,
                record_id=int(record_id),
                output_root=root,
                total_timesteps=total_timesteps,
                seed=payload.get("seed"),
                min_behavior_steps=min_behavior_steps,
                behavior_switch_prob=behavior_switch_prob,
                behavior_mix=mix,
                ctx=ctx,
                rollout_fn=rollout_fn,
                parallel_envs=1,
                adapt=adapt,
                independent_ally_policies=independent,
                mid_episode_policy_switch=mid_switch,
                best_teacher_reference=best_teacher_reference,
                best_policy=best_policy,
                dump_attack_target=dump_attack_target,
            )
        else:
            path = generate_one_record(
                package,
                record_id=int(record_id),
                output_root=root,
                total_timesteps=total_timesteps,
                seed=payload.get("seed"),
                min_behavior_steps=min_behavior_steps,
                behavior_switch_prob=behavior_switch_prob,
                behavior_mix=mix,
                ctx=ctx,
                scan_rollout=False,
                rollout_fn=None,
                independent_ally_policies=independent,
                mid_episode_policy_switch=mid_switch,
                best_teacher_reference=best_teacher_reference,
                best_policy=best_policy,
                dump_attack_target=dump_attack_target,
            )
        generate_s = time.perf_counter() - t_gen
        paths.append(str(path))
        if report_progress:
            _emit_record_done(
                payload,
                worker_id=worker_id,
                device=device,
                package_name=package.name,
                record_id=int(record_id),
                path=str(path),
                generate_s=generate_s,
                parallel_envs=parallel_envs,
            )


def _adapt_mix_for_package(
    *,
    package,
    record_ids: list[int],
    payload: dict[str, Any],
    ctx,
    rollout_fn,
    base_mix,
    scan_rollout: bool,
    parallel_envs: int,
    total_timesteps: int,
    min_behavior_steps: int,
    behavior_switch_prob: float,
    worker_id: int,
    device: str,
    paths: list[str],
) -> None:
    import shutil
    import tempfile

    from generate.winrate_adapt import (
        DEFAULT_STRENGTH_MAX,
        adapt_meta_dict,
        choose_adapt_params,
        in_win_rate_band,
        reweight_mix,
        summarize_record_paths,
    )

    win_rate_min = float(payload.get("win_rate_min", 0.30))
    raw_max = payload.get("win_rate_max", None)
    win_rate_max = None if raw_max is None else float(raw_max)
    pilot_n = max(1, int(payload.get("adapt_pilot_records", 4) or 4))
    max_iters = max(1, int(payload.get("adapt_max_iters", 3) or 3))
    strength_max = float(
        payload.get("adapt_strength_max", DEFAULT_STRENGTH_MAX) or DEFAULT_STRENGTH_MAX
    )
    strength_max = min(max(strength_max, 0.0), 1.0)

    ids = [int(r) for r in record_ids]
    # Search mix on temporary pilots; only the final mix is written to output_root.
    strength = 0.5
    oracle_focus = 0.0
    using_base = True
    active_mix = base_mix
    measured_wr: float | None = None
    status = "adapt_failed"
    pilot_rounds = 0
    # With best_teacher_reference, focus strong mass on task best teacher.
    focus_policy = (
        str(payload.get("best_policy") or "oracle_pure")
        if bool(payload.get("best_teacher_reference", False))
        else "oracle_pure"
    )

    out_root = Path(payload["output_root"])
    pilot_parent = out_root / ".adapt_pilot"
    pilot_parent.mkdir(parents=True, exist_ok=True)
    pilot_root = Path(
        tempfile.mkdtemp(prefix=f"{package.name}_w{worker_id}_", dir=str(pilot_parent))
    )

    def _run_pilot_round(round_idx: int) -> float:
        nonlocal pilot_rounds
        pilot_ids = list(range(round_idx * pilot_n, (round_idx + 1) * pilot_n))
        adapt = adapt_meta_dict(
            strength=None if using_base else strength,
            win_rate_min=win_rate_min,
            win_rate_max=win_rate_max,
            measured_win_rate=measured_wr,
            status="pilot",
            pilot=True,
            mix=active_mix,
            oracle_focus=0.0 if using_base else oracle_focus,
            strength_max=strength_max,
            focus_policy=focus_policy,
        )
        round_paths: list[str] = []
        _generate_ids(
            package=package,
            record_ids=pilot_ids,
            payload=payload,
            ctx=ctx,
            rollout_fn=rollout_fn,
            mix=active_mix,
            scan_rollout=scan_rollout,
            parallel_envs=parallel_envs,
            total_timesteps=total_timesteps,
            min_behavior_steps=min_behavior_steps,
            behavior_switch_prob=behavior_switch_prob,
            worker_id=worker_id,
            device=device,
            adapt=adapt,
            paths=round_paths,
            output_root=pilot_root,
            report_progress=False,
        )
        pilot_rounds += 1
        _counts, wr = summarize_record_paths(round_paths)
        shutil.rmtree(pilot_root / package.name, ignore_errors=True)
        return float(wr)

    try:
        needs_verify = False
        for it in range(max_iters):
            measured_wr = _run_pilot_round(it)
            needs_verify = False
            if in_win_rate_band(
                measured_wr, win_rate_min=win_rate_min, win_rate_max=win_rate_max
            ):
                status = "ok"
                break

            nxt = choose_adapt_params(
                measured_wr,
                win_rate_min=win_rate_min,
                win_rate_max=win_rate_max,
                strength=strength,
                oracle_focus=oracle_focus,
                strength_max=strength_max,
            )
            if nxt is None:
                status = "adapt_failed"
                break
            new_strength, new_focus = nxt
            if (
                (not using_base)
                and abs(new_strength - strength) < 1e-6
                and abs(new_focus - oracle_focus) < 1e-6
            ):
                status = "adapt_failed"
                break
            strength, oracle_focus = new_strength, new_focus
            active_mix = reweight_mix(
                base_mix,
                strength,
                oracle_focus=oracle_focus,
                strength_max=strength_max,
                focus_policy=focus_policy,
            )
            using_base = False
            needs_verify = True

        # Last iteration may have updated the mix without measuring it.
        if needs_verify and status != "ok":
            measured_wr = _run_pilot_round(pilot_rounds)
            status = (
                "ok"
                if in_win_rate_band(
                    measured_wr, win_rate_min=win_rate_min, win_rate_max=win_rate_max
                )
                else "adapt_failed"
            )
    finally:
        shutil.rmtree(pilot_root, ignore_errors=True)
        try:
            if pilot_parent.is_dir() and not any(pilot_parent.iterdir()):
                pilot_parent.rmdir()
        except OSError:
            pass

    _report(
        payload,
        {
            "event": "adapt_done",
            "worker_id": worker_id,
            "device": device,
            "gpu_id": payload.get("gpu_id"),
            "package_name": package.name,
            "strength": float(strength) if not using_base else None,
            "oracle_focus": float(oracle_focus),
            "strength_max": float(strength_max),
            "measured_win_rate": None if measured_wr is None else float(measured_wr),
            "win_rate_min": win_rate_min,
            "win_rate_max": win_rate_max,
            "status": status,
            "pilot_rounds": pilot_rounds,
            "pilot_records": pilot_rounds * pilot_n,
        },
    )
    print(
        f"[record_worker] adapt package={package.name} status={status} "
        f"strength={strength:.2f} oracle_focus={oracle_focus:.2f} "
        f"focus_policy={focus_policy} "
        f"wr={measured_wr} band=[{win_rate_min},{win_rate_max}]",
        flush=True,
    )

    final_adapt = adapt_meta_dict(
        strength=None if using_base else strength,
        win_rate_min=win_rate_min,
        win_rate_max=win_rate_max,
        measured_win_rate=measured_wr,
        status=status,
        pilot=False,
        mix=active_mix,
        oracle_focus=0.0 if using_base else oracle_focus,
        strength_max=strength_max,
        focus_policy=focus_policy,
    )
    _generate_ids(
        package=package,
        record_ids=ids,
        payload=payload,
        ctx=ctx,
        rollout_fn=rollout_fn,
        mix=active_mix,
        scan_rollout=scan_rollout,
        parallel_envs=parallel_envs,
        total_timesteps=total_timesteps,
        min_behavior_steps=min_behavior_steps,
        behavior_switch_prob=behavior_switch_prob,
        worker_id=worker_id,
        device=device,
        adapt=final_adapt,
        paths=paths,
    )


def worker_main(payload: dict[str, Any]) -> str:
    """Generate all pre-assigned records for one dedicated worker slot."""

    device = str(payload["device"])
    worker_id = int(payload.get("worker_id", -1))
    stagger_s = float(payload.get("stagger_s", 0.0) or 0.0)
    if stagger_s > 0 and worker_id > 0:
        time.sleep(stagger_s * worker_id)

    if device == "gpu":
        gpu_id = payload["gpu_id"]
        if gpu_id is None:
            raise ValueError("gpu worker requires gpu_id")
        force_jax_gpu_env(
            gpu_id, jax_platform=str(payload.get("jax_platform", "cuda"))
        )
    else:
        force_jax_cpu_env()

    # Import after device env is configured.
    from generate.behavior_mix import behavior_mix_from_config
    from generate.env_centric import (
        build_record_gen_context,
        warmup_record_gen_context,
    )
    from generate.task_package import discover_task_packages

    from generate.coach_eval import BEST_ORACLE, ensure_coach_eval

    packages = {p.name: p for p in discover_task_packages(payload["packages_root"])}
    base_mix = behavior_mix_from_config(payload.get("behavior_mix"))
    scan_rollout = bool(payload.get("scan_rollout", True))
    parallel_envs = max(1, int(payload.get("parallel_envs", 1) or 1))
    if not scan_rollout:
        parallel_envs = 1
    winrate_adapt = bool(payload.get("winrate_adapt", False))
    best_teacher_reference = bool(payload.get("best_teacher_reference", False))
    coach_eval_episodes = max(1, int(payload.get("coach_eval_episodes", 64) or 64))
    coach_eval_tie_eps = float(payload.get("coach_eval_tie_eps", 0.02) or 0.02)
    coach_eval_max_episode_steps = max(
        1, int(payload.get("coach_eval_max_episode_steps", 512) or 512)
    )
    coach_eval_parallel_envs = max(
        1, int(payload.get("coach_eval_parallel_envs", 32) or 32)
    )
    train_like_sample = bool(payload.get("oracle_train_like_sample", False))
    # Adapt needs scan path (runtime CDF). Fall back to no-adapt if scan off.
    if winrate_adapt and not scan_rollout:
        print(
            "[record_worker] winrate_adapt ignored when scan_rollout=false",
            flush=True,
        )
        winrate_adapt = False
    total_timesteps = int(payload["total_timesteps"])
    min_behavior_steps = int(payload["min_behavior_steps"])
    behavior_switch_prob = float(payload["behavior_switch_prob"])
    # Reuse env+oracle (+ optional scan fn) per package inside this process.
    contexts: dict[str, Any] = {}
    rollout_fns: dict[str, Any] = {}
    package_best_policy: dict[str, str] = {}
    paths: list[str] = []
    try:
        for item in payload["items"]:
            package = packages[item["package_name"]]
            ctx = contexts.get(package.name)
            if ctx is None:
                t_setup = time.perf_counter()
                ctx = build_record_gen_context(
                    package, train_like_sample=train_like_sample
                )
                setup_s = time.perf_counter() - t_setup
                if best_teacher_reference:
                    # Missing coach_eval → pre-eval oracle vs advanced, write
                    # task.json, then generate with the winner as hard reference.
                    ensure = ensure_coach_eval(
                        package,
                        ctx=ctx,
                        num_episodes=coach_eval_episodes,
                        seed=int(payload.get("seed") or 0)
                        + 997 * int(package.task_index),
                        max_episode_steps=coach_eval_max_episode_steps,
                        tie_eps=coach_eval_tie_eps,
                        parallel_envs=coach_eval_parallel_envs,
                        train_like_sample=train_like_sample,
                    )
                    best_policy = str(ensure["best_policy"])
                    if ensure["wrote"]:
                        ev = ensure.get("eval_result") or {}
                        o = (ev.get("oracle_pure") or {}).get("win_rate")
                        a = (ev.get("heuristic_advanced") or {}).get("win_rate")
                        print(
                            f"[record_worker] package={package.name}: "
                            f"wrote coach_eval best_policy={best_policy} "
                            f"oracle_wr={o} advanced_wr={a}",
                            flush=True,
                        )
                        _report(
                            payload,
                            {
                                "event": "coach_eval_done",
                                "worker_id": worker_id,
                                "device": device,
                                "gpu_id": payload.get("gpu_id"),
                                "package_name": package.name,
                                "best_policy": best_policy,
                                "wrote": True,
                                "oracle_wr": o,
                                "advanced_wr": a,
                            },
                        )
                    else:
                        print(
                            f"[record_worker] package={package.name}: "
                            f"reuse coach_eval best_policy={best_policy}",
                            flush=True,
                        )
                else:
                    best_policy = BEST_ORACLE
                package_best_policy[package.name] = best_policy
                # Warm *every* behavior in the mix so generate-time switches
                # do not re-enter XLA compile (as much as JAX allows).
                compile_s = warmup_record_gen_context(
                    ctx,
                    behavior_mix=base_mix,
                    steps=2,
                    seed=int(payload.get("seed") or 0) + worker_id,
                )
                if scan_rollout:
                    from generate.scan_rollout import (
                        build_scan_rollout_fn,
                        warmup_scan_rollout,
                    )

                    teacher = best_policy if best_teacher_reference else BEST_ORACLE
                    rollout_fn = build_scan_rollout_fn(
                        ctx,
                        mix=base_mix,
                        min_behavior_steps=min_behavior_steps,
                        behavior_switch_prob=behavior_switch_prob,
                        total_timesteps=total_timesteps,
                        parallel_envs=parallel_envs,
                        independent_ally_policies=bool(
                            payload.get("independent_ally_policies", False)
                        ),
                        mid_episode_policy_switch=bool(
                            payload.get("mid_episode_policy_switch", True)
                        ),
                        best_teacher_policy=teacher,
                        dump_attack_target=bool(
                            payload.get("dump_attack_target", False)
                        ),
                    )
                    compile_s += warmup_scan_rollout(
                        rollout_fn,
                        mix=base_mix,
                        seed=int(payload.get("seed") or 0) + worker_id,
                        parallel_envs=parallel_envs,
                    )
                    rollout_fns[package.name] = rollout_fn
                contexts[package.name] = ctx
                _report(
                    payload,
                    {
                        "event": "compile_done",
                        "worker_id": worker_id,
                        "device": device,
                        "gpu_id": payload.get("gpu_id"),
                        "package_name": item["package_name"],
                        "setup_s": round(setup_s, 3),
                        "compile_s": round(compile_s, 3),
                        "parallel_envs": parallel_envs,
                        "best_policy": package_best_policy[package.name],
                    },
                )

            # Per-package teacher for adapt / reference meta (payload copy).
            item_payload = dict(payload)
            item_payload["best_teacher_reference"] = best_teacher_reference
            item_payload["best_policy"] = package_best_policy.get(
                package.name, BEST_ORACLE
            )

            record_ids = [int(r) for r in item["record_ids"]]
            if winrate_adapt:
                _adapt_mix_for_package(
                    package=package,
                    record_ids=record_ids,
                    payload=item_payload,
                    ctx=ctx,
                    rollout_fn=rollout_fns.get(package.name),
                    base_mix=base_mix,
                    scan_rollout=scan_rollout,
                    parallel_envs=parallel_envs,
                    total_timesteps=total_timesteps,
                    min_behavior_steps=min_behavior_steps,
                    behavior_switch_prob=behavior_switch_prob,
                    worker_id=worker_id,
                    device=device,
                    paths=paths,
                )
            else:
                _generate_ids(
                    package=package,
                    record_ids=record_ids,
                    payload=item_payload,
                    ctx=ctx,
                    rollout_fn=rollout_fns.get(package.name),
                    mix=base_mix,
                    scan_rollout=scan_rollout,
                    parallel_envs=parallel_envs,
                    total_timesteps=total_timesteps,
                    min_behavior_steps=min_behavior_steps,
                    behavior_switch_prob=behavior_switch_prob,
                    worker_id=worker_id,
                    device=device,
                    adapt=None,
                    paths=paths,
                )
    except Exception as exc:
        _report(
            payload,
            {
                "event": "worker_error",
                "worker_id": worker_id,
                "device": device,
                "gpu_id": payload.get("gpu_id"),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        raise
    return ",".join(paths)
