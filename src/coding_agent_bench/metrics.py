from pathlib import Path
import json
import argparse
from pydantic import BaseModel, computed_field

from coding_agent_bench.leaderboard_results import atomic_json, build_leaderboard_result, trial_times

DEFAULT_GPU_COST_USD_PER_HOUR = 4


class Metrics(BaseModel):
    n_tasks: int
    n_errors: int
    score: float
    n_input_tokens: int | None
    n_cache_tokens: int | None
    n_output_tokens: int | None
    n_total_tokens: int | None
    agent_time_seconds: int | None
    total_time_seconds: int | None
    cost_usd: float | None
    @computed_field
    def mean_input_tokens_per_task(self) -> int | None:
        return self.n_input_tokens // self.n_tasks if self.n_input_tokens is not None and self.n_tasks else None

    @computed_field
    def mean_cache_tokens_per_task(self) -> int | None:
        return self.n_cache_tokens // self.n_tasks if self.n_cache_tokens is not None and self.n_tasks else None

    @computed_field
    def mean_output_tokens_per_task(self) -> int | None:
        return self.n_output_tokens // self.n_tasks if self.n_output_tokens is not None and self.n_tasks else None

    @computed_field
    def mean_tokens_per_task(self) -> int | None:
        return self.n_total_tokens // self.n_tasks if self.n_total_tokens is not None and self.n_tasks else None

    @computed_field
    def mean_cost_usd_per_task(self) -> float | None:
        if self.cost_usd is None or not self.n_tasks:
            return None
        return round(self.cost_usd / self.n_tasks, 2)

    @computed_field
    def mean_total_time_seconds_per_task(self) -> int | None:
        return self.total_time_seconds // self.n_tasks if self.total_time_seconds is not None and self.n_tasks else None

    @computed_field
    def mean_agent_time_seconds_per_task(self) -> int | None:
        return self.agent_time_seconds // self.n_tasks if self.agent_time_seconds is not None and self.n_tasks else None

    @computed_field
    def cache_hit_rate(self) -> float | None:
        if self.n_cache_tokens is None or self.n_input_tokens in (None, 0):
            return None
        return self.n_cache_tokens / self.n_input_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", type=Path, help="Harbor job directory")
    parser.add_argument("--num-gpus", type=int, default=None)
    def _positive_float(value):
        f = float(value)
        import math
        if f < 0 or not math.isfinite(f):  # reject negative, NaN, and inf
            raise argparse.ArgumentTypeError(f"must be a non-negative number, got {value}")
        return f
    parser.add_argument("--gpu-cost-per-hour", type=_positive_float, default=DEFAULT_GPU_COST_USD_PER_HOUR,
                        help=f"Cost per GPU per hour in USD (default: ${DEFAULT_GPU_COST_USD_PER_HOUR})")
    parser.add_argument(
        "--leaderboard-metadata",
        type=Path,
        help="JSON containing per-job leaderboard metadata (benchmark/model/harness details)",
    )
    parser.add_argument(
        "--leaderboard-output",
        type=Path,
        help="Write a leaderboard result JSON accepted by coding-agent-leaderboard",
    )
    args = parser.parse_args()
    if args.num_gpus is not None and args.num_gpus <= 0:
        parser.error("--num-gpus must be positive")
    if args.leaderboard_metadata and not args.leaderboard_output:
        parser.error("--leaderboard-metadata requires --leaderboard-output")
    return args


def _sum_optional(values: list[int | None]) -> int | None:
    """Do not present partial token accounting as a complete total."""
    return sum(values) if values and all(v is not None for v in values) else None


def _trial_results(job_dir: Path) -> list[dict]:
    """Read only trial-level results, excluding nested task artifacts."""
    return [json.loads(p.read_text()) for p in sorted(job_dir.glob("*/result.json"))]


def _run_concurrency(job_dir: Path, config: dict) -> int:
    """Read current/legacy concurrency settings; a lock is optional for export."""
    value = config.get("n_concurrent_trials") or (config.get("orchestrator") or {}).get("n_concurrent_trials")
    lock_path = job_dir / "lock.json"
    if value is None and lock_path.exists():
        value = json.loads(lock_path.read_text()).get("n_concurrent_trials")
    if value is None:
        return 1
    if type(value) is not int or value <= 0:
        raise ValueError(f"{job_dir}: concurrency must be a positive integer")
    return value


def determine_format(job_dir: Path):
    job_result = json.loads((job_dir / "result.json").read_text())
    if "n_input_tokens" in job_result.get("stats", {}):
        return "latest"
    return "legacy"


def extract_score(job_result: dict, eval_name: str) -> float:
    """Score metric: 'reward' (itbench-style) or 'mean' (legacy), mirroring aggregate_runs.py."""
    metrics = job_result["stats"]["evals"][eval_name]["metrics"][0]
    for key in ("reward", "mean"):
        if key in metrics:
            return float(metrics[key])
    raise KeyError(f"No score metric found in eval '{eval_name}'; available keys: {sorted(metrics)}")


def compute_metrics_legacy(job_dir: Path, num_gpus: int = None, gpu_cost_per_hour: float = DEFAULT_GPU_COST_USD_PER_HOUR):

    job_result_path = job_dir / "result.json"
    job_result = json.loads(job_result_path.read_text())

    job_config_path = job_dir / "config.json"
    job_config = json.loads(job_config_path.read_text())

    eval_name = list(job_result["stats"]["evals"].keys())[0]
    print(eval_name)
    n_concurrent = _run_concurrency(job_dir, job_config)
    task_results = _trial_results(job_dir)
    total_time, agent_time = trial_times(task_results, job_result["stats"].get("n_completed_trials", job_result["stats"].get("n_trials", 0)))
    agent_results = [r.get("agent_result") or {} for r in task_results]
    n_input_tokens = _sum_optional([r.get("n_input_tokens") for r in agent_results])
    n_cache_tokens = _sum_optional([r.get("n_cache_tokens") for r in agent_results])
    n_output_tokens = _sum_optional([r.get("n_output_tokens") for r in agent_results])
    total_tokens = (
        n_input_tokens + n_output_tokens
        if n_input_tokens is not None and n_output_tokens is not None
        else None
    )
    score = round(extract_score(job_result, eval_name), 3)
    cost = job_result["stats"].get("cost_usd")

    print(num_gpus)
    if num_gpus is not None:
        cost = round(agent_time * gpu_cost_per_hour * num_gpus / n_concurrent / 3600, 2) if agent_time is not None else None

    metrics = Metrics(
        n_tasks=job_result["n_total_trials"],
        n_errors=job_result["stats"]["n_errors"],
        score=score,
        n_input_tokens=n_input_tokens,
        n_cache_tokens=n_cache_tokens,
        n_output_tokens=n_output_tokens,
        n_total_tokens=total_tokens,
        agent_time_seconds=agent_time,
        total_time_seconds=total_time,
        cost_usd=cost,
    )
    return metrics


def compute_metrics_latest(job_dir: Path, num_gpus: int = None, gpu_cost_per_hour: float = DEFAULT_GPU_COST_USD_PER_HOUR):

    job_result_path = job_dir / "result.json"
    job_result = json.loads(job_result_path.read_text())

    job_config_path = job_dir / "config.json"
    job_config = json.loads(job_config_path.read_text())

    eval_name = list(job_result["stats"]["evals"].keys())[0]
    print(eval_name)
    n_concurrent = _run_concurrency(job_dir, job_config)
    task_results = _trial_results(job_dir)
    total_time, agent_time = trial_times(task_results, job_result["stats"].get("n_completed_trials", job_result["stats"].get("n_trials", 0)))
    stats = job_result["stats"]
    n_input_tokens = stats.get("n_input_tokens")
    n_cache_tokens = stats.get("n_cache_tokens")
    n_output_tokens = stats.get("n_output_tokens")
    total_tokens = (
        n_input_tokens + n_output_tokens
        if n_input_tokens is not None and n_output_tokens is not None
        else None
    )
    score = round(extract_score(job_result, eval_name), 3)
    cost = job_result["stats"].get("cost_usd")

    if num_gpus is not None:
        cost = round(agent_time * gpu_cost_per_hour * num_gpus / n_concurrent / 3600, 2) if agent_time is not None else None

    metrics = Metrics(
        n_tasks=job_result["n_total_trials"],
        n_errors=stats.get("n_errored_trials", stats.get("n_errors", 0)),
        score=score,
        n_input_tokens=n_input_tokens,
        n_cache_tokens=n_cache_tokens,
        n_output_tokens=n_output_tokens,
        n_total_tokens=total_tokens,
        agent_time_seconds=agent_time,
        total_time_seconds=total_time,
        cost_usd=cost,
    )

    return metrics

def main():
    args = parse_args()
    num_gpus = args.num_gpus
    gpu_cost_per_hour = args.gpu_cost_per_hour

    if args.leaderboard_output:
        overrides = json.loads(args.leaderboard_metadata.read_text()) if args.leaderboard_metadata else None
        result = build_leaderboard_result(args.job_dir, overrides=overrides)
        if num_gpus is not None:
            config_path = args.job_dir / "config.json"
            config = json.loads(config_path.read_text()) if config_path.exists() else {}
            seconds = result["metrics"]["agent_time_seconds"]
            cost = round(seconds * gpu_cost_per_hour * num_gpus / _run_concurrency(args.job_dir, config) / 3600, 2) if seconds is not None else None
            result["metrics"]["cost_usd"] = cost
            count = result["metrics"]["n_tasks"]
            result["metrics"]["mean_cost_usd_per_task"] = round(cost / count, 2) if cost is not None and count else None
            result["run"]["cost_estimated"] = True
        atomic_json(args.leaderboard_output, result)
        print(f"Wrote leaderboard result to {args.leaderboard_output}")
        return

    job_format = determine_format(args.job_dir)
    if job_format == "legacy":
        metrics = compute_metrics_legacy(args.job_dir, num_gpus, gpu_cost_per_hour)
    elif job_format == "latest":
        metrics = compute_metrics_latest(args.job_dir, num_gpus, gpu_cost_per_hour)
    print(f"job {args.job_dir.name}: score={metrics.score} errors={metrics.n_errors} cost=${metrics.cost_usd}")
    print(metrics.model_dump_json(indent=4))

if __name__ == "__main__":
    main()
