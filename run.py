from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import metaworld
import numpy as np

from models import load_model
from metaworld.policies.registry import get_expert_policy
from utils import MWTrajectory, load_mw_dataset


@dataclass
class EpisodeResult:
    success: bool
    action_deviation: float
    inference_time: float
    steps: int


def clip_action(action: np.ndarray) -> np.ndarray:
    limit = 1.0 - 1e-5
    action = np.asarray(action, dtype=np.float64).copy()
    return np.clip(action, -limit, limit)


def parse_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def append_csv(path: Path, row: Dict[str, Any], fields: Sequence[str]) -> None:
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def log(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


def make_env(task_name: str, res: int, include_depth: bool, seed: int):
    return metaworld.mw_gym_make(
        task_name,
        goal_cost_reward=False,
        stop_at_goal=True,
        steps_at_goal=1,
        cam_height=res,
        cam_width=res,
        depth=include_depth,
        train_distrib=True,
        seed=seed,
    )


def collect_expert_demo(
    env,
    expert,
    initial_state: Dict[str, Any],
    horizon: int,
) -> MWTrajectory:
    obs = initial_state["image"]
    state = initial_state["full_state"]

    actions: List[np.ndarray] = []
    observations: List[np.ndarray] = []
    states: List[np.ndarray] = []

    info: Dict[str, Any] = {}

    for _ in range(horizon):
        observations.append(np.asarray(obs, dtype=np.uint8))
        states.append(np.asarray(state, dtype=np.float64))

        action = clip_action(expert.get_action(state))
        actions.append(np.asarray(action, dtype=np.float64))

        next_state, _reward, done, info = env.step(action)
        obs = next_state["image"]
        state = next_state["full_state"]

        if done:
            break

    return MWTrajectory(
        actions=np.stack(actions, axis=0),
        depths=None,
        full_states=np.stack(states, axis=0),
        observations=np.stack(observations, axis=0),
        proprio_states=None,
        rewards=None,
        terminals=None,
        infos={"task_accomplished": bool(info.get("task_accomplished", False))},
    )


def rollout_eval_episode(env, policy, expert, horizon: int) -> EpisodeResult:
    env_state = env.reset()
    obs = env_state["image"]
    state = env_state["full_state"]

    policy.reset(obs, state)

    distances: List[float] = []
    model_time = 0.0
    steps = 0
    info: Dict[str, Any] = {}

    for _ in range(horizon):
        expert_action = clip_action(expert.get_action(state))

        start = time.perf_counter()
        action = policy.predict_action()
        model_time += time.perf_counter() - start

        action = clip_action(action)
        distances.append(float(np.linalg.norm(action - expert_action)))

        next_state, _reward, done, info = env.step(action)
        steps += 1

        start = time.perf_counter()
        policy.step(
            obs=next_state["image"],
            full_state=next_state["full_state"],
        )
        model_time += time.perf_counter() - start

        obs = next_state["image"]
        state = next_state["full_state"]

        if done:
            break

    return EpisodeResult(
        success=bool(info.get("task_accomplished", False)),
        action_deviation=float(np.mean(distances)) if distances else float("nan"),
        inference_time=model_time,
        steps=steps,
    )


def evaluate(
    policy,
    task_name: str,
    episodes: int,
    res: int,
    include_depth: bool,
    horizon: int,
    seed: int,
    quiet: bool,
) -> Dict[str, Any]:
    env = make_env(task_name, res=res, include_depth=include_depth, seed=seed)
    expert = get_expert_policy(task_name)

    success_count = 0
    deviations: List[float] = []
    success_times: List[float] = []
    success_steps: List[int] = []

    start = time.perf_counter()

    try:
        for ep in range(episodes):
            result = rollout_eval_episode(env, policy, expert, horizon)

            success_count += int(result.success)
            deviations.append(result.action_deviation)

            if result.success:
                success_times.append(result.inference_time)
                success_steps.append(result.steps)

            if not quiet:
                status = "SUCCESS" if result.success else "FAIL"
                print(
                    f"[eval:{task_name}] {ep + 1:03d}/{episodes} | "
                    f"{status} | steps={result.steps} | "
                    f"deviation={result.action_deviation:.6f} | "
                    f"inference={result.inference_time:.6f}s",
                    flush=True,
                )
    finally:
        env.close()

    return {
        "success_rate": success_count / float(episodes),
        "action_deviation": float(np.mean(deviations)) if deviations else float("nan"),
        "success_inference_time": (
            float(np.mean(success_times)) if success_times else None
        ),
        "success_steps": float(np.mean(success_steps)) if success_steps else None,
        "num_successes": success_count,
        "num_episodes": episodes,
        "eval_wall_time": time.perf_counter() - start,
        "has_success_timing": bool(success_times),
    }


def collect_demos(
    policy,
    task_name: str,
    num_demos: int,
    max_attempts: int,
    res: int,
    include_depth: bool,
    seed: int,
    horizon: int,
    quiet: bool,
) -> Tuple[List[MWTrajectory], Dict[str, Any]]:
    policy_env = make_env(task_name, res=res, include_depth=include_depth, seed=seed)
    expert_env = make_env(task_name, res=res, include_depth=include_depth, seed=seed)
    expert = get_expert_policy(task_name)

    demos: List[MWTrajectory] = []
    attempts = 0
    failed = 0
    forced = 0

    try:
        while len(demos) < num_demos and attempts < max_attempts:
            attempts += 1

            policy_init = policy_env.reset()
            expert_init = expert_env.reset()

            policy.reset(policy_init["image"], policy_init["full_state"])

            info: Dict[str, Any] = {}
            for _ in range(horizon):
                action = clip_action(policy.predict_action())
                next_state, _reward, done, info = policy_env.step(action)

                policy.step(
                    obs=next_state["image"],
                    full_state=next_state["full_state"],
                )

                if done:
                    break

            if not bool(info.get("task_accomplished", False)):
                demo = collect_expert_demo(expert_env, expert, expert_init, horizon)
                demos.append(demo)
                failed += 1
                log(
                    f"[collect:{task_name}] attempt={attempts} | "
                    f"failed demo {len(demos)}/{num_demos}",
                    quiet,
                )
            else:
                log(
                    f"[collect:{task_name}] attempt={attempts} | solved, skipped",
                    quiet,
                )

        while len(demos) < num_demos:
            policy_init = policy_env.reset()
            expert_init = expert_env.reset()

            policy.reset(policy_init["image"], policy_init["full_state"])

            for _ in range(horizon):
                action = clip_action(policy.predict_action())
                next_state, _reward, done, _info = policy_env.step(action)

                policy.step(
                    obs=next_state["image"],
                    full_state=next_state["full_state"],
                )

                if done:
                    break

            demo = collect_expert_demo(expert_env, expert, expert_init, horizon)
            demos.append(demo)
            forced += 1

            log(
                f"[collect:{task_name}] forced demo {len(demos)}/{num_demos}",
                quiet,
            )
    finally:
        policy_env.close()
        expert_env.close()

    return demos, {
        "requested": num_demos,
        "collected": len(demos),
        "attempts": attempts,
        "failed": failed,
        "forced": forced,
    }


def save_plot(
    x: Sequence[int],
    y: Sequence[float],
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o", linewidth=2.0)
    plt.xlabel("Additional Trajectories")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def create_policy(model_name: str, kwargs: Optional[Dict[str, Any]] = None):
    return load_model(model_name, **(kwargs or {}))


def run_task(
    *,
    model_name: str,
    task_name: str,
    horizon: int,
    dataset_root: str,
    log_dir: Path,
    eval_episodes: int,
    eval_seed: int,
    collect_seed: int,
    res: int,
    include_depth: bool,
    steps: int,
    demos_per_step: int,
    max_attempts: int,
    quiet_eval: bool,
    quiet_collect: bool,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    model_kwargs = model_kwargs or {}
    ensure_dir(log_dir)

    save_json(
        log_dir / "config.json",
        {
            "model": model_name,
            "task": task_name,
            "horizon": horizon,
            "dataset_root": dataset_root,
            "eval_episodes": eval_episodes,
            "eval_seed": eval_seed,
            "collect_seed": collect_seed,
            "res": res,
            "include_depth": include_depth,
            "steps": steps,
            "demos_per_step": demos_per_step,
            "max_attempts": max_attempts,
            "model_kwargs": model_kwargs,
        },
    )

    eval_csv = log_dir / "eval.csv"
    train_csv = log_dir / "train.csv"
    collect_csv = log_dir / "collect.csv"

    eval_fields = [
        "step",
        "added",
        "success_rate",
        "action_deviation",
        "success_inference_time",
        "success_steps",
        "num_successes",
        "num_episodes",
        "eval_wall_time",
        "has_success_timing",
    ]
    train_fields = ["step", "added", "operation", "time", "num_demos"]
    collect_fields = ["step", "requested", "collected", "attempts", "failed", "forced"]

    log(f"[{task_name}] loading dataset", quiet_collect)
    dataset = load_mw_dataset(task_name, root=dataset_root)

    policy = create_policy(model_name, model_kwargs)

    start = time.perf_counter()
    policy.fit(dataset)
    train_time = time.perf_counter() - start

    append_csv(
        train_csv,
        {
            "step": 0,
            "added": 0,
            "operation": "fit",
            "time": train_time,
            "num_demos": len(dataset),
        },
        train_fields,
    )

    log(f"[{task_name}] fit | demos={len(dataset)} | time={train_time:.3f}s", quiet_collect)

    eval_result = evaluate(
        policy=policy,
        task_name=task_name,
        episodes=eval_episodes,
        res=res,
        include_depth=include_depth,
        horizon=horizon,
        seed=eval_seed,
        quiet=quiet_eval,
    )

    append_csv(eval_csv, {"step": 0, "added": 0, **eval_result}, eval_fields)

    x_values = [0]
    success_values = [float(eval_result["success_rate"])]
    deviation_values = [float(eval_result["action_deviation"])]

    save_plot(
        x_values,
        success_values,
        "Success Rate",
        f"{model_name} on {task_name}",
        log_dir / "success_rate.png",
    )
    save_plot(
        x_values,
        deviation_values,
        "Average Action Deviation",
        f"{model_name} on {task_name}",
        log_dir / "action_deviation.png",
    )

    added = 0

    for step in range(1, steps + 1):
        log(f"[{task_name}] step {step}/{steps} | collecting demos", quiet_collect)

        new_demos, collect_stats = collect_demos(
            policy=policy,
            task_name=task_name,
            num_demos=demos_per_step,
            max_attempts=max_attempts,
            res=res,
            include_depth=include_depth,
            seed=collect_seed + step - 1,
            horizon=horizon,
            quiet=quiet_collect,
        )

        added += len(new_demos)

        append_csv(
            collect_csv,
            {"step": step, **collect_stats},
            collect_fields,
        )

        start = time.perf_counter()
        policy.update(new_demos)
        train_time = time.perf_counter() - start

        append_csv(
            train_csv,
            {
                "step": step,
                "added": added,
                "operation": "update",
                "time": train_time,
                "num_demos": len(new_demos),
            },
            train_fields,
        )

        log(
            f"[{task_name}] update | step={step} | added={added} | "
            f"demos={len(new_demos)} | time={train_time:.3f}s",
            quiet_collect,
        )

        eval_result = evaluate(
            policy=policy,
            task_name=task_name,
            episodes=eval_episodes,
            res=res,
            include_depth=include_depth,
            horizon=horizon,
            seed=eval_seed,
            quiet=quiet_eval,
        )

        append_csv(eval_csv, {"step": step, "added": added, **eval_result}, eval_fields)

        log(
            f"[{task_name}] eval | step={step} | added={added} | "
            f"success={eval_result['success_rate']:.4f} | "
            f"deviation={eval_result['action_deviation']:.6f}",
            quiet_collect,
        )

        x_values.append(added)
        success_values.append(float(eval_result["success_rate"]))
        deviation_values.append(float(eval_result["action_deviation"]))

        save_plot(
            x_values,
            success_values,
            "Success Rate",
            f"{model_name} on {task_name}",
            log_dir / "success_rate.png",
        )
        save_plot(
            x_values,
            deviation_values,
            "Average Action Deviation",
            f"{model_name} on {task_name}",
            log_dir / "action_deviation.png",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--horizons", type=str, required=True)

    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--dataset_root", type=str, default="plex_data")

    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--eval_seed", type=int, default=42)
    parser.add_argument("--collect_seed", type=int, default=123)

    parser.add_argument("--res", type=int, default=224)
    parser.add_argument("--include_depth", action="store_true")

    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--demos_per_step", type=int, default=5)
    parser.add_argument("--max_attempts", type=int, default=100)

    parser.add_argument("--model_kwargs_json", type=str, default=None)

    parser.add_argument("--quiet_eval", action="store_true")
    parser.add_argument("--quiet_collect", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    tasks = parse_list(args.tasks)
    horizons = parse_int_list(args.horizons)

    if len(tasks) != len(horizons):
        raise ValueError(
            f"--tasks and --horizons must have the same length, "
            f"got {len(tasks)} and {len(horizons)}"
        )

    if args.model_kwargs_json is None:
        model_kwargs: Dict[str, Any] = {}
    else:
        model_kwargs = json.loads(args.model_kwargs_json)
        if not isinstance(model_kwargs, dict):
            raise ValueError("--model_kwargs_json must decode to a dictionary")

    for task_name, horizon in zip(tasks, horizons):
        task_log_dir = Path(args.log_dir) / args.model_name / task_name
        ensure_dir(task_log_dir)

        run_task(
            model_name=args.model_name,
            task_name=task_name,
            horizon=horizon,
            dataset_root=args.dataset_root,
            log_dir=task_log_dir,
            eval_episodes=args.eval_episodes,
            eval_seed=args.eval_seed,
            collect_seed=args.collect_seed,
            res=args.res,
            include_depth=args.include_depth,
            steps=args.steps,
            demos_per_step=args.demos_per_step,
            max_attempts=args.max_attempts,
            quiet_eval=args.quiet_eval,
            quiet_collect=args.quiet_collect,
            model_kwargs=model_kwargs,
        )


if __name__ == "__main__":
    main()