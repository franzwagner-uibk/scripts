from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, MODULE_ROOT / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = _load("buildNorthTyrolLaunchKit")
evidence = _load("captureNorthTyrolFailureEvidence")


def _canonical_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    setup = tmp_path / "north_tyrol"
    refresh = {
        "promotion_result": "promoted",
        "projects": {
            name: {"event_count": count}
            for name, count in launcher.EXPECTED_PROJECT_EVENTS.items()
        },
        "station_roles": {"da": 28, "holdout": 7},
    }
    refresh_path = setup / "raw" / "metadata" / "canonical_refresh_manifest.json"
    refresh_path.parent.mkdir(parents=True)
    refresh_path.write_text(json.dumps(refresh), encoding="utf-8")
    roles = setup / "obs" / "stations" / "stations_da_metadata.csv"
    roles.parent.mkdir(parents=True)
    roles.write_text("station_id,role\nA,da\n", encoding="utf-8")
    for project_name in launcher.EXPECTED_PROJECT_EVENTS:
        project_root = setup / "projects" / project_name
        project_root.mkdir(parents=True)
        (project_root / f"{project_name}.yml").write_text("project: canonical\n")
        for index in range(8):
            leaf_project = (
                project_root
                / "subdomains"
                / f"AT-00-00-{index:02d}"
                / "projects"
                / project_name
            )
            step = leaf_project / "steps" / "step_00"
            step.mkdir(parents=True)
            (leaf_project / f"{project_name}.yml").write_text("project: leaf\n")
            (step / "step_00.yml").write_text("start_date: 2017-10-01\n")
    monkeypatch.setattr(launcher, "EXPECTED_STEP_COUNT", 48)
    return setup


def test_launch_kit_uses_one_canonical_mount_and_compact_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _canonical_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(launcher, "_scripts_commit", lambda: "a" * 40)

    output = launcher.build_kit(
        setup_root=setup,
        project_name="project_2018_2019",
        image="registry.example/openamundsen-da@sha256:" + "b" * 64,
        output_dir=tmp_path / "kit",
    )

    script = (output / "run_pipeline.sh").read_text(encoding="utf-8")
    assert script.count("--volume") == 1
    assert '$SETUP_ROOT:/setup' in script
    assert "/data" not in script
    assert " --overwrite" not in script
    assert "captureNorthTyrolFailureEvidence.py" in script
    assert "--check" in script
    assert "docker ps -q" in script
    manifest = json.loads((output / "launch_manifest.json").read_text())
    assert manifest["container_project_root"] == "/setup/projects/project_2018_2019"
    assert manifest["scripts_commit"] == "a" * 40


def test_launch_kit_requires_immutable_image(tmp_path: Path) -> None:
    setup = tmp_path / "north_tyrol"
    (setup / "projects" / "project_2018_2019").mkdir(parents=True)

    with pytest.raises(ValueError, match="immutable"):
        launcher.build_kit(
            setup_root=setup,
            project_name="project_2018_2019",
            image="registry.example/openamundsen-da:latest",
            output_dir=tmp_path / "kit",
        )


def test_launch_manifest_rejects_contract_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _canonical_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(launcher, "_scripts_commit", lambda: "a" * 40)
    output = launcher.build_kit(
        setup_root=setup,
        project_name="project_2018_2019",
        image="registry.example/openamundsen-da@sha256:" + "b" * 64,
        output_dir=tmp_path / "kit",
    )

    launcher.validate_launch_manifest(output / "launch_manifest.json")
    (setup / "projects" / "project_2018_2019" / "project_2018_2019.yml").write_text(
        "project: changed\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contract changed"):
        launcher.validate_launch_manifest(output / "launch_manifest.json")


def test_setup_contract_rejects_target_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup = _canonical_setup(tmp_path, monkeypatch)
    runtime = setup / "projects" / "project_2018_2019" / "results"
    runtime.mkdir()

    with pytest.raises(ValueError, match="runtime artifacts"):
        launcher._validate_setup_contract(setup, "project_2018_2019")


def test_failure_evidence_summarizes_without_copying_large_ledgers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    runtime = project / "subdomains" / "S1" / "results"
    runtime.mkdir(parents=True)
    (runtime / "run_manifest.json").write_text(
        json.dumps({"status": "failed", "error": "synthetic failure"}),
        encoding="utf-8",
    )
    retention = runtime / "retention_manifest.json"
    retention.write_text(
        json.dumps({"status": "planned", "padding": "x" * 200}),
        encoding="utf-8",
    )
    perf = runtime / "project_perf_metrics.csv"
    perf.write_text("timestamp,cpu\n2026-01-01,10\n", encoding="utf-8")
    monkeypatch.setattr(evidence, "MAX_PARSE_BYTES", 64)

    summary_path = evidence.capture(project, tmp_path / "evidence")

    summary = json.loads(summary_path.read_text())
    records = {record["path"]: record for record in summary["records"]}
    assert records["subdomains/S1/results/run_manifest.json"]["status"] == "failed"
    assert (
        records["subdomains/S1/results/retention_manifest.json"]["parse_status"]
        == "identity_only_oversized"
    )
    assert not list((tmp_path / "evidence").rglob("retention_manifest.json"))
    assert list((tmp_path / "evidence" / "diagnostics").rglob("project_perf_metrics.csv"))
