#!/usr/bin/env python3
"""Build a canonical single-mount North Tyrol run/merge/render launch kit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path


IMAGE_RE = re.compile(r"^[^\s@]+@sha256:[0-9a-f]{64}$")
EXPECTED_PROJECT_EVENTS = {
    "project_2017_2018": 38,
    "project_2018_2019": 45,
    "project_2019_2020": 44,
    "project_2020_2021": 43,
    "project_2021_2022": 46,
    "project_2022_2023": 42,
}
EXPECTED_LEAF_COUNT = 48
EXPECTED_STEP_COUNT = 1_833
EXPECTED_STATION_ROLES = {"da": 28, "holdout": 7}


def _scripts_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_inventory(setup_root: Path) -> list[Path]:
    paths = [
        setup_root / "raw" / "metadata" / "canonical_refresh_manifest.json",
        setup_root / "obs" / "stations" / "stations_da_metadata.csv",
    ]
    for project_name in EXPECTED_PROJECT_EVENTS:
        project_root = setup_root / "projects" / project_name
        paths.append(project_root / f"{project_name}.yml")
        for leaf_root in sorted(
            path for path in (project_root / "subdomains").glob("AT-*") if path.is_dir()
        ):
            leaf_project = leaf_root / "projects" / project_name
            paths.append(leaf_project / f"{project_name}.yml")
            paths.extend(sorted((leaf_project / "steps").glob("step_*/*.yml")))
    missing = [path for path in paths if not path.is_file() or path.is_symlink()]
    if missing:
        raise ValueError(f"Canonical launch contract has missing/non-regular files: {missing[:5]}")
    return paths


def _contract_digest(setup_root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(setup_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_setup_contract(setup_root: Path, project_name: str) -> dict[str, object]:
    if project_name not in EXPECTED_PROJECT_EVENTS:
        raise ValueError(f"Unknown North Tyrol project: {project_name}")
    refresh_path = setup_root / "raw" / "metadata" / "canonical_refresh_manifest.json"
    refresh = json.loads(refresh_path.read_text(encoding="utf-8"))
    if refresh.get("promotion_result") != "promoted":
        raise ValueError("Canonical refresh transaction is not recorded as promoted")
    actual_events = {
        name: int(refresh.get("projects", {}).get(name, {}).get("event_count", -1))
        for name in EXPECTED_PROJECT_EVENTS
    }
    if actual_events != EXPECTED_PROJECT_EVENTS:
        raise ValueError(f"North Tyrol event totals changed: {actual_events}")
    roles = refresh.get("station_roles")
    if roles != EXPECTED_STATION_ROLES:
        raise ValueError(f"North Tyrol station roles changed: {roles}")

    leaf_count = 0
    step_count = 0
    for name in EXPECTED_PROJECT_EVENTS:
        project_root = setup_root / "projects" / name
        leaves = sorted(
            path for path in (project_root / "subdomains").glob("AT-*") if path.is_dir()
        )
        if len(leaves) != 8:
            raise ValueError(f"Expected eight leaves for {name}, found {len(leaves)}")
        leaf_count += len(leaves)
        for leaf in leaves:
            leaf_project = leaf / "projects" / name
            step_count += len(list((leaf_project / "steps").glob("step_*/*.yml")))
    if leaf_count != EXPECTED_LEAF_COUNT or step_count != EXPECTED_STEP_COUNT:
        raise ValueError(
            f"North Tyrol prepared counts changed: leaves={leaf_count}, steps={step_count}"
        )

    target_root = setup_root / "projects" / project_name
    runtime_paths = [
        path
        for pattern in ("results", "state*.pickle*", "*.restart*", "runtime_generation_*")
        for path in target_root.rglob(pattern)
        if path.exists()
    ]
    if runtime_paths:
        raise ValueError(f"Target project contains runtime artifacts: {runtime_paths[:5]}")

    contract_paths = _contract_inventory(setup_root)
    return {
        "project_event_counts": actual_events,
        "station_roles": roles,
        "leaf_count": leaf_count,
        "step_count": step_count,
        "contract_file_count": len(contract_paths),
        "contract_sha256": _contract_digest(setup_root, contract_paths),
    }


def validate_launch_manifest(path: Path) -> None:
    """Fail closed when the prepared setup no longer matches the launch kit."""
    manifest = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    setup_root = Path(str(manifest["setup_root"])).resolve(strict=True)
    stat_result = setup_root.stat()
    if [stat_result.st_dev, stat_result.st_ino] != manifest["setup_identity"]:
        raise ValueError("Canonical setup filesystem identity changed after launch-kit creation")
    contract = _validate_setup_contract(setup_root, str(manifest["project_name"]))
    for key in (
        "project_event_counts",
        "station_roles",
        "leaf_count",
        "step_count",
        "contract_file_count",
        "contract_sha256",
    ):
        if contract[key] != manifest[key]:
            raise ValueError(f"Canonical launch contract changed for {key}")


def build_kit(
    *,
    setup_root: Path,
    project_name: str,
    image: str,
    output_dir: Path,
) -> Path:
    """Create a launch kit that exposes the setup through `/setup` exactly once."""
    setup_root = setup_root.resolve(strict=True)
    project_root = setup_root / "projects" / project_name
    if not project_root.is_dir():
        raise FileNotFoundError(f"North Tyrol project does not exist: {project_root}")
    if not IMAGE_RE.fullmatch(image):
        raise ValueError("Image must be an immutable registry reference ending in @sha256:<64 hex>")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Launch-kit destination exists: {output_dir}")
    output_dir.mkdir(parents=True)
    contract = _validate_setup_contract(setup_root, project_name)

    collector_source = Path(__file__).with_name("captureNorthTyrolFailureEvidence.py")
    collector_target = output_dir / collector_source.name
    shutil.copy2(collector_source, collector_target)
    collector_target.chmod(collector_target.stat().st_mode | stat.S_IXUSR)
    validator_target = output_dir / Path(__file__).name
    shutil.copy2(Path(__file__), validator_target)
    validator_target.chmod(validator_target.stat().st_mode | stat.S_IXUSR)

    setup_stat = setup_root.stat()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "setup_root": str(setup_root),
        "setup_identity": [setup_stat.st_dev, setup_stat.st_ino],
        "project_name": project_name,
        "container_setup_root": "/setup",
        "container_project_root": f"/setup/projects/{project_name}",
        "image": image,
        "scripts_commit": _scripts_commit(),
        "outer_workers": 8,
        "inner_workers": 6,
        "render_workers": 24,
        **contract,
    }
    (output_dir / "launch_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    quoted_setup = shlex.quote(str(setup_root))
    quoted_project = shlex.quote(project_name)
    quoted_image = shlex.quote(image)
    launcher = output_dir / "run_pipeline.sh"
    launcher.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

readonly SETUP_ROOT={quoted_setup}
readonly PROJECT_HOST=${{SETUP_ROOT}}/projects/{quoted_project}
readonly PROJECT=/setup/projects/{quoted_project}
readonly IMAGE={quoted_image}
readonly KIT_DIR=\"$(cd \"$(dirname \"${{BASH_SOURCE[0]}}\")\" && pwd)\"
readonly STATUS_FILE=${{KIT_DIR}}/status
readonly LOG_FILE=${{KIT_DIR}}/pipeline.log

write_status() {{
    local temporary=${{STATUS_FILE}}.tmp.$$
    printf '%s\n' \"$1\" > \"$temporary\"
    mv \"$temporary\" \"$STATUS_FILE\"
}}

preserve_failure() {{
    local stamp evidence
    stamp=\"$(date -u +%Y%m%dT%H%M%SZ)\"
    evidence=${{KIT_DIR}}/failure_evidence/${{stamp}}
    python3 \"${{KIT_DIR}}/captureNorthTyrolFailureEvidence.py\" \"$PROJECT_HOST\" \"$evidence\"
    cp -a \"$LOG_FILE\" \"$STATUS_FILE\" \"$evidence/\" 2>/dev/null || true
    df -h \"$SETUP_ROOT\" > \"$evidence/filesystem.txt\"
    docker ps -a --no-trunc > \"$evidence/containers.txt\"
}}

finish() {{
    local exit_code=$?
    if [[ $exit_code -eq 0 ]]; then
        write_status SUCCESS
    else
        write_status FAILED
        preserve_failure
    fi
    exit $exit_code
}}

oa() {{
    local name=$1
    shift
    docker run --rm --init \\
        --name \"$name\" \\
        --network none \\
        --user \"$(id -u):$(id -g)\" \\
        --env OMP_NUM_THREADS=1 \\
        --env OPENBLAS_NUM_THREADS=1 \\
        --env MKL_NUM_THREADS=1 \\
        --env NUMEXPR_NUM_THREADS=1 \\
        --volume \"$SETUP_ROOT:/setup\" \\
        --workdir /setup \\
        \"$IMAGE\" openamundsen-da \"$@\"
}}

trap finish EXIT
exec > >(tee -a \"$LOG_FILE\") 2>&1
write_status VALIDATING
python3 \"${{KIT_DIR}}/buildNorthTyrolLaunchKit.py\" \\
    --check \"${{KIT_DIR}}/launch_manifest.json\"
if [[ -n \"$(docker ps -q)\" ]]; then
    echo \"P8 is not idle: running containers exist\" >&2
    docker ps --no-trunc >&2
    exit 1
fi
if process_list=\"$(pgrep -af 'openamundsen-da|openamundsen')\"; then
    echo \"P8 is not idle: openAMUNDSEN processes exist\" >&2
    printf '%s\\n' \"$process_list\" >&2
    exit 1
fi
docker pull \"$IMAGE\"
docker image inspect \"$IMAGE\" >/dev/null
df -h \"$SETUP_ROOT\"
write_status STORAGE_ADMISSION
oa north-tyrol-{project_name}-run subdomains run \"$PROJECT\" --max-workers 8 --inner-max-workers 6
write_status MERGING
oa north-tyrol-{project_name}-merge subdomains merge \"$PROJECT\"
write_status RENDERING
oa north-tyrol-{project_name}-render subdomains render \"$PROJECT\" --max-workers 24
""",
        encoding="utf-8",
    )
    launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR)
    (output_dir / "status_command.txt").write_text(
        f"cat {output_dir}/status && tail -n 80 {output_dir}/pipeline.log\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-root", type=Path)
    parser.add_argument("--project")
    parser.add_argument("--image")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.check is not None:
        if any(value is not None for value in (args.setup_root, args.project, args.image, args.output_dir)):
            parser.error("--check cannot be combined with launch-kit creation arguments")
        validate_launch_manifest(args.check)
        print("LAUNCH_CONTRACT_OK")
        return 0
    if any(value is None for value in (args.setup_root, args.project, args.image, args.output_dir)):
        parser.error("--setup-root, --project, --image and --output-dir are required")
    print(
        build_kit(
            setup_root=args.setup_root,
            project_name=args.project,
            image=args.image,
            output_dir=args.output_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
