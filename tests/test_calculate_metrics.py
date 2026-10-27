"""Shared leaderboard export, Harbor accounting, and queue upload integration."""

import json
from pathlib import Path
import sys

import pytest

from coding_agent_bench import leaderboard_results
from scripts.calculate_metrics import (
    compute_metrics_latest, compute_metrics_legacy, main,
)


def _job_dir(root: Path) -> Path:
    job = root / "job"
    job.mkdir(parents=True)
    (job / "config.json").write_text(json.dumps({
        "datasets": [{"name": "org/bench", "ref": "sha256:" + "a" * 64}],
        "agents": [{"name": "opencode", "model_name": "vllm/custom-model",
                    "kwargs": {"version": "1.2"}, "skills": ["org/skill@v1"]}],
    }))
    (job / "result.json").write_text(json.dumps({
        "started_at": "2026-10-15T11:00:00Z", "finished_at": "2026-10-15T12:00:00Z",
        "n_total_trials": 2,
        "stats": {"n_completed_trials": 2, "n_errored_trials": 0, "n_errors": 0,
                  "n_input_tokens": 40, "n_cache_tokens": 16, "n_output_tokens": 10,
                  "cost_usd": 1.0, "evals": {"run": {"metrics": [{"mean": 0.5}]}}},
    }))
    (job / "lock.json").write_text(json.dumps({
        "trials": [{"task": {"name": f"task-{index}", "digest": "sha256:" + str(index) * 64}}
                   for index in range(2)],
    }))
    for index in range(2):
        trial = job / f"trial-{index}"
        trial.mkdir()
        (trial / "result.json").write_text(json.dumps({
            "task_name": f"task-{index}", "started_at": "2026-10-15T11:00:00Z",
            "finished_at": "2026-10-15T11:00:20Z",
            "agent_execution": {"started_at": "2026-10-15T11:00:05Z", "finished_at": "2026-10-15T11:00:15Z"},
            "agent_info": {"name": "opencode", "version": "1.2.1"},
            "agent_result": {"n_input_tokens": 20, "n_cache_tokens": 8, "n_output_tokens": 5},
        }))
    return job


def _update(path, mutate):
    document = json.loads(path.read_text())
    mutate(document)
    path.write_text(json.dumps(document))


def _github(commit="a" * 40, ref="main", directory="tasks"):
    return {"repository_url": "https://github.com/org/bench", "requested_ref": ref,
            "commit": commit, "subdirectory": directory}


def test_default_benchmark_labels_exclude_refs_but_preserve_provenance():
    for ref in ("latest", "v1", "sha256:" + "a" * 64):
        benchmark = leaderboard_results.job_metadata(f"org/bench@{ref}", "model", "oracle")["benchmark"]
        assert benchmark["name"] == "org/bench"
        assert benchmark["version"] == ref
    for ref in ("main", "v1", "a" * 40):
        source = _github(ref=ref)
        benchmark = leaderboard_results.job_metadata(
            "/tmp/cab-github-dataset/source/tasks", "model", "oracle", github_source=source,
        )["benchmark"]
        assert benchmark["name"] == "org/bench/tasks"
        assert benchmark["dataset_source"] == source


@pytest.mark.parametrize("source", ["org/bench", "swe-bench/swe-bench-verified", "scale-ai/swe-bench-pro"])
def test_candidate_uses_dataset_name_and_keeps_hub_source_for_identity(tmp_path, source):
    job = _job_dir(tmp_path)
    _update(job / "config.json", lambda config: config["datasets"][0].update(name=source))
    leaderboard_results.save_metadata(job, leaderboard_results.job_metadata(f"{source}@latest", "model", "opencode"))
    result = leaderboard_results.build_leaderboard_result(job)
    assert result["benchmark"]["name"] == source
    assert "url" not in result["benchmark"]
    assert result["benchmark"]["repo"] == source
    assert result["benchmark"]["version"] == "latest"
    assert result["environment"]["config"]["name"] == source
    assert result["benchmark"]["id"] == leaderboard_results.dataset_identity(
        {"name": source, "ref": "sha256:" + "a" * 64}, selected_task_names=["task-0", "task-1"],
    )


@pytest.mark.parametrize("returncode, paused, status", [
    (0, False, "completed"), (1, False, "failed"), (130, False, "stopped"),
    (-15, False, "stopped"), (0, True, "paused"),
])
def test_manual_and_automatic_exports_are_identical(tmp_path, monkeypatch, returncode, paused, status):
    job = _job_dir(tmp_path)
    pause = tmp_path / "pause"
    monkeypatch.setenv("CAB_PAUSE_REQUEST_PATH", str(pause))
    if paused:
        pause.write_text("{}")
    leaderboard_results.save_metadata(job, leaderboard_results.job_metadata("org/bench@v1", "custom-model", "opencode"))
    automatic = leaderboard_results.export_on_completion(job, returncode)
    expected = json.loads(automatic.read_text())
    output = tmp_path / "manual.json"
    monkeypatch.setattr(sys, "argv", ["calculate_metrics.py", str(job), "--leaderboard-output", str(output)])
    main()
    assert json.loads(output.read_text()) == expected
    assert expected["run"]["status"] == status
    assert expected["metrics"]["n_total_tokens"] == 50
    assert expected["metrics"]["agent_time_seconds"] == 20
    assert expected["harness"]["version"] == "1.2.1"
    assert "url" not in expected["harness"]
    assert expected["benchmark"]["name"] == "org/bench"
    assert expected["benchmark"]["version"] == "v1"


def test_manual_partial_export_needs_no_metadata_and_refreshes_output(tmp_path, monkeypatch):
    job = _job_dir(tmp_path)
    _update(job / "result.json", lambda result: result.update(finished_at=None))
    _update(job / "result.json", lambda result: result["stats"].update(n_completed_trials=1))
    output = tmp_path / "snapshot.json"
    monkeypatch.setattr(sys, "argv", ["calculate_metrics.py", str(job), "--leaderboard-output", str(output)])
    main()
    first = json.loads(output.read_text())
    assert first["run"]["status"] == "partial"
    assert first["metrics"]["n_tasks"] == 1
    _update(job / "result.json", lambda result: result.update(finished_at="2026-10-15T12:00:00Z"))
    _update(job / "result.json", lambda result: result["stats"].update(n_completed_trials=2))
    main()
    second = json.loads(output.read_text())
    assert second["run"]["status"] == "completed"
    assert second["benchmark"]["id"] == first["benchmark"]["id"]


def test_metadata_overrides_do_not_override_resolved_identity(tmp_path):
    job = _job_dir(tmp_path)
    expected = leaderboard_results.build_leaderboard_result(job)
    result = leaderboard_results.build_leaderboard_result(job, overrides={
        "benchmark": {"name": "Custom display label", "id": "not-the-source"},
        "model": {"precision": "fp8"}, "environment": {"gpu_type": "H200", "gpu_count": 8},
    })
    assert result["benchmark"]["name"] == "Custom display label"
    assert result["benchmark"]["id"] == expected["benchmark"]["id"]
    assert result["model"]["precision"] == "fp8"
    assert result["environment"]["gpu_count"] == 8


@pytest.mark.parametrize("change", [
    {"ref": "sha256:" + "b" * 64},
    {"registry_url": "https://other-registry.example"},
])
def test_hub_source_revision_change_identity_not_display_name(tmp_path, change):
    job = _job_dir(tmp_path)
    first = leaderboard_results.build_leaderboard_result(job)
    _update(job / "config.json", lambda config: config["datasets"][0].update(change))
    second = leaderboard_results.build_leaderboard_result(job)
    assert first["benchmark"]["name"] == second["benchmark"]["name"]
    assert first["benchmark"]["id"] != second["benchmark"]["id"]


def test_same_resolved_tasks_hash_the_same_despite_different_selectors(tmp_path):
    job = _job_dir(tmp_path)
    _update(job / "lock.json", lambda lock: lock.update({
        "trials": [{"task": {"name": "task-0", "digest": "sha256:" + "0" * 64}}],
    }))
    first = leaderboard_results.build_leaderboard_result(job)
    _update(job / "config.json", lambda config: config["datasets"][0].update({
        "task_names": ["task-0", "task-*"],
        "exclude_task_names": ["not-selected-*"],
        "n_tasks": 1,
    }))
    second = leaderboard_results.build_leaderboard_result(job)
    assert first["benchmark"]["id"] == second["benchmark"]["id"]
    assert second["environment"]["config"]["n_tasks"] == 1

    _update(job / "lock.json", lambda lock: lock.update({
        "trials": [{"task": {"name": "different-task", "digest": "sha256:" + "c" * 64}}],
    }))
    third = leaderboard_results.build_leaderboard_result(job)
    assert third["benchmark"]["id"] != second["benchmark"]["id"]


def test_resolved_task_names_deduplicate_attempts_and_agents(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "lock.json").write_text(json.dumps({
        "trials": [
            {"task": {"name": "org/task-b"}},
            {"task": {"name": "org/task-a"}},
            {"task": {"name": "org/task-a"}},
        ],
    }))
    assert leaderboard_results.resolved_task_names(job) == ["org/task-a", "org/task-b"]


@pytest.mark.parametrize("names", [
    ["scale-ai/instance_ansible__ansible-a", "scale-ai/instance_ansible__ansible-b"],
    ["scale-ai/instance_ansible__ansible-a", "scale-ai/instance_other__repo-b"],
    ["scale-ai/instance_other__repo-b"],
])
def test_task_selection_does_not_rewrite_dataset_label(tmp_path, names):
    job = _job_dir(tmp_path)
    _update(job / "config.json", lambda config: config["datasets"][0].update(
        name="scale-ai/swe-bench-pro", task_names=["*"],
    ))
    _update(job / "lock.json", lambda lock: lock.update(trials=[{"task": {"name": name}} for name in names]))
    result = leaderboard_results.build_leaderboard_result(job)
    assert result["benchmark"]["name"] == "scale-ai/swe-bench-pro"
    assert result["benchmark"]["repo"] == "scale-ai/swe-bench-pro"
    assert "url" not in result["benchmark"]
    assert result["benchmark"]["id"] == leaderboard_results.dataset_identity(
        {"name": "scale-ai/swe-bench-pro", "ref": "sha256:" + "a" * 64}, selected_task_names=names,
    )


def test_identity_normalizes_resolved_task_set_and_github_source_aliases():
    dataset = {"name": "org/bench", "ref": "sha256:pinned", "task_names": ["pattern-*"]}
    assert leaderboard_results.dataset_identity(
        dataset, selected_task_names=["org/task-b", "org/task-a"]
    ) == leaderboard_results.dataset_identity(
        {**dataset, "task_names": ["different-expression"]},
        selected_task_names=["org/task-a", "org/task-b", "org/task-a"],
    )
    assert leaderboard_results.dataset_identity(
        {}, _github(ref="main"), selected_task_names=["task-1"]
    ) == leaderboard_results.dataset_identity(
        {}, _github(ref="v1"), selected_task_names=["task-1"]
    )
    assert leaderboard_results.dataset_identity(
        {}, _github(), selected_task_names=["task-1"]
    ) != leaderboard_results.dataset_identity(
        {}, _github(commit="b" * 40), selected_task_names=["task-1"]
    )
    assert leaderboard_results.dataset_identity(
        {}, _github(), selected_task_names=["task-1"]
    ) != leaderboard_results.dataset_identity(
        {}, _github(directory="other"), selected_task_names=["task-1"]
    )


@pytest.mark.parametrize("contents", [None, "{", "{}", '{"trials": [{}]}'])
def test_result_without_usable_lock_does_not_export_candidate(tmp_path, contents):
    job = _job_dir(tmp_path)
    if contents is None:
        (job / "lock.json").unlink()
    else:
        (job / "lock.json").write_text(contents)
    assert leaderboard_results.try_export_on_completion(job, 1) is None
    assert not (job / leaderboard_results.RESULT_FILE).exists()


def test_result_without_dataset_source_does_not_use_job_directory_as_identity(tmp_path):
    job = _job_dir(tmp_path)
    (job / "config.json").unlink()
    assert leaderboard_results.try_export_on_completion(job, 0) is None
    assert not (job / leaderboard_results.RESULT_FILE).exists()


def test_github_source_and_identity_survive_downloaded_resume(tmp_path, monkeypatch):
    job = _job_dir(tmp_path)
    _update(job / "config.json", lambda config: config.update(datasets=[{"path": "/tmp/cab-github-dataset/source/tasks"}]))
    runtime = tmp_path / "source.json"
    runtime.write_text(json.dumps(_github()))
    monkeypatch.setattr(leaderboard_results, "GITHUB_DATASET_PROVENANCE", runtime)
    leaderboard_results.save_metadata(job, leaderboard_results.job_metadata(
        "/tmp/cab-github-dataset/source/tasks", "model", "opencode", github_source=_github(),
    ))
    output = leaderboard_results.export_on_completion(job, 1)
    first = json.loads(output.read_text())
    runtime.unlink()
    assert leaderboard_results.export_on_completion(job, 0) == output
    second = json.loads(output.read_text())
    assert first["benchmark"] == second["benchmark"]
    assert second["benchmark"]["dataset_source"] == _github()
    assert second["environment"]["config"] == {}
    assert second["run"]["status"] == "completed"


@pytest.mark.parametrize("phase, present, expected", [
    (None, False, None),  # Field absent in an older or incomplete artifact.
    (None, True, 0),     # Harbor explicitly records a phase it never entered.
    ({"started_at": None, "finished_at": None}, True, 0),
    ({}, True, None),
    ({"started_at": "2026-10-15T11:00:00Z", "finished_at": None}, True, None),
    ({"started_at": "2026-10-15T11:00:05Z", "finished_at": "2026-10-15T11:00:03Z"}, True, None),
    ({"started_at": "invalid", "finished_at": "invalid"}, True, None),
    ({"started_at": "2026-10-15T11:00:00Z", "finished_at": "2026-10-15T11:00:05Z"}, True, 5),
])
def test_harbor_phase_timing_semantics(phase, present, expected):
    assert leaderboard_results.phase_duration(phase, present=present) == expected


def test_missing_agent_timing_keeps_totals_and_averages_unknown(tmp_path):
    job = _job_dir(tmp_path)
    _update(job / "trial-0/result.json", lambda trial: trial.pop("agent_execution"))
    metrics = leaderboard_results.build_leaderboard_result(job)["metrics"]
    assert metrics["agent_time_seconds"] is None
    assert metrics["mean_agent_time_seconds_per_task"] is None
    assert metrics["total_time_seconds"] == 40
    _update(job / "trial-0/result.json", lambda trial: trial.update(agent_execution=None))
    assert leaderboard_results.build_leaderboard_result(job)["metrics"]["agent_time_seconds"] == 10


def test_multistep_timings_and_never_started_cancelled_trial(tmp_path):
    job = _job_dir(tmp_path)
    def multistep(trial):
        phase = trial.pop("agent_execution")
        trial["step_results"] = [{"agent_execution": phase}, {"agent_execution": phase}]
    _update(job / "trial-0/result.json", multistep)
    _update(job / "trial-1/result.json", lambda trial: trial.update(
        started_at=None, agent_execution=None, exception_info={"exception_type": "CancelledError"},
    ))
    result = leaderboard_results.build_leaderboard_result(job)
    assert result["metrics"]["agent_time_seconds"] == 20
    assert result["metrics"]["total_time_seconds"] == 20
    assert result["run"]["complete"] is False


@pytest.mark.parametrize("legacy", [False, True])
def test_completed_count_includes_errors_and_legacy_tokens(tmp_path, legacy):
    job = _job_dir(tmp_path)
    def change(result):
        stats = result["stats"]
        stats["n_errored_trials"] = 1
        if legacy:
            stats["n_trials"] = stats.pop("n_completed_trials")
            stats["n_errors"] = stats.pop("n_errored_trials")
            for key in ("n_input_tokens", "n_cache_tokens", "n_output_tokens"):
                stats.pop(key)
    _update(job / "result.json", change)
    result = leaderboard_results.build_leaderboard_result(job)
    assert result["metrics"]["n_tasks"] == 2
    assert result["metrics"]["n_errors"] == 1
    assert result["metrics"]["n_total_tokens"] == 50


def test_no_root_result_and_corrupt_root_do_not_replace_previous_snapshot(tmp_path):
    assert leaderboard_results.export_on_completion(tmp_path, 1) is None
    job = _job_dir(tmp_path)
    output = leaderboard_results.export_on_completion(job, 0)
    previous = output.read_bytes()
    (job / "result.json").write_text('{"stats":')
    assert leaderboard_results.try_export_on_completion(job, 1) is None
    assert output.read_bytes() == previous


def test_atomic_candidate_write_keeps_previous_file_on_io_failure(tmp_path, monkeypatch):
    path = tmp_path / "candidate.json"
    path.write_text("previous")
    def fail(*args):
        raise OSError("disk error")
    monkeypatch.setattr(leaderboard_results.os, "replace", fail)
    with pytest.raises(OSError):
        leaderboard_results.atomic_json(path, {"score": 1})
    assert path.read_text() == "previous"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("returncode", [0, 1])
def test_cli_collects_metadata_without_custom_json(tmp_path, monkeypatch, returncode):
    from typer.testing import CliRunner
    from coding_agent_bench import cli
    job = _job_dir(tmp_path)
    monkeypatch.setattr(cli.HarborCommandBuilder, "build", lambda *a, **kw: (["harbor"], job))
    class Process:
        def __init__(self, *args, **kwargs):
            metadata = json.loads((job / leaderboard_results.METADATA_FILE).read_text())
            assert metadata["model"] == {"name": "model"}
            assert metadata["benchmark"]["name"] == "org/bench"
            assert metadata["benchmark"]["version"] == "v1"
            self.returncode = returncode
        def wait(self):
            return self.returncode
    monkeypatch.setattr(cli.subprocess, "Popen", Process)
    result = CliRunner().invoke(cli.app, ["run", "--agent", "opencode", "--dataset", "org/bench@v1",
                                        "--model-name", "model", "--server-url", "https://model.example"])
    assert result.exit_code == returncode, result.output
    assert (job / leaderboard_results.RESULT_FILE).exists()


@pytest.mark.parametrize("returncode, paused", [(0, False), (1, False), (0, True)])
def test_queue_resume_uploads_latest_snapshot(tmp_path, monkeypatch, returncode, paused):
    from coding_agent_bench import api
    from tests.test_result_upload import run_shell
    job = _job_dir(tmp_path)
    monkeypatch.setenv("CAB_PAUSE_REQUEST_PATH", str(tmp_path / "pause"))
    candidate = leaderboard_results.export_on_completion(job, 0)
    original_id = json.loads(candidate.read_text())["benchmark"]["id"]
    updated = json.loads((job / "result.json").read_text())
    updated["stats"]["evals"] = {"run": {"metrics": [{"mean": 0.8}]}}
    updated["stats"]["n_errored_trials"] = 1
    command = api._build_resume_shell_command("job with spaces", "attempt-2", [], None)
    status, _ = run_shell(tmp_path, command, harbor_rc=returncode, pause_requested=paused,
                          result_data=updated, seed_job=job)
    assert status == returncode
    output = json.loads((tmp_path / "remote/results/job with spaces" / leaderboard_results.RESULT_FILE).read_text())
    assert output["benchmark"]["id"] == original_id
    assert output["metrics"]["score"] == 0.8
    assert output["metrics"]["n_errors"] == 1
    assert type(output["metrics"]["n_errors"]) is int
    assert output["run"]["status"] == ("paused" if paused else "failed" if returncode else "completed")


@pytest.mark.parametrize("resource, count", [("h200", 1), ("h200x8", 8)])
def test_queue_captures_gpu_without_metadata_input(resource, count):
    from coding_agent_bench.api import CreateJobRequest, build_cli_command
    command = build_cli_command(CreateJobRequest(job_name="test", agent="opencode", dataset="org/bench",
                                                model_name="model", server_url=f"nebius-{resource}"))
    metadata = json.loads(command[command.index("--leaderboard-metadata-json") + 1])
    assert metadata == {"environment": {"gpu_type": "H200", "gpu_count": count}}


def test_cli_rejects_multiple_job_directories(tmp_path, monkeypatch):
    first, second = _job_dir(tmp_path / "first"), _job_dir(tmp_path / "second")
    monkeypatch.setattr(sys, "argv", ["calculate_metrics.py", str(first), str(second),
                                     "--leaderboard-output", str(tmp_path / "candidate.json")])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2


def test_manual_export_and_explicit_cost_estimate(tmp_path, monkeypatch):
    job = _job_dir(tmp_path)
    output = tmp_path / "candidate.json"
    monkeypatch.setattr(sys, "argv", ["calculate_metrics.py", str(job),
                                     "--leaderboard-output", str(output), "--num-gpus", "2"])
    main()
    result = json.loads(output.read_text())
    assert result["metrics"]["score"] == 0.5
    assert result["metrics"]["n_errors"] == 0
    assert type(result["metrics"]["n_errors"]) is int
    assert "n_runs" not in result["run"]
    assert result["run"]["cost_estimated"] is True
    assert result["metrics"]["cost_usd"] == 0.04


def test_manual_gpu_cost_export_allows_missing_config(tmp_path, monkeypatch):
    job = _job_dir(tmp_path)
    leaderboard_results.save_metadata(job, leaderboard_results.job_metadata(
        "/tmp/cab-github-dataset/source/tasks", "model", "opencode", github_source=_github(),
    ))
    (job / "config.json").unlink()
    output = tmp_path / "candidate.json"
    monkeypatch.setattr(sys, "argv", ["calculate_metrics.py", str(job),
                                     "--leaderboard-output", str(output), "--num-gpus", "2"])

    main()

    result = json.loads(output.read_text())
    assert result["metrics"]["cost_usd"] == 0.04
    assert result["run"]["cost_estimated"] is True


@pytest.mark.parametrize("compute", [compute_metrics_latest, compute_metrics_legacy])
def test_report_metrics_preserve_token_and_cost_accounting(tmp_path, compute):
    job = _job_dir(tmp_path)
    metrics = compute(job)
    assert metrics.n_total_tokens == 50  # Cache tokens are already in input.
    assert metrics.total_time_seconds == 40
    assert metrics.agent_time_seconds == 20
    _update(job / "result.json", lambda result: result["stats"].update(cost_usd=None))
    unknown = compute(job)
    assert unknown.cost_usd is None
    assert compute(job, num_gpus=2).cost_usd == 0.04
