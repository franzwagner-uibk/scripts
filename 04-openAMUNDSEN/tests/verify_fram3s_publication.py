"""Exercise Windows publication on disposable files, including failures before archival."""

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_publication(root):
    work, dest, report = root / "work", root / "destination", root / "record"
    aoi = work / "output/01-aoi"
    aoi.mkdir(parents=True)
    paths = {
        "dem": "05-dem/euregio/dem_euregio_20.asc",
        "lc": "03-landcover/lc_eusalp/openAMUNDSEN-euregio/lc_euregio_20_eusalp.asc",
        "provinces": "01-aoi/PROVINCE_BOUNDARY/province_boundary_4326.gpkg",
        "subregions": "01-aoi/SUBREGIONS/raw/subregions_avalanche_report_4326_raw.gpkg",
    }
    source_manifest = {}
    for kind, relative in paths.items():
        path = dest / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("legacy " + kind)
        source_manifest[kind] = {"path": relative, "sha256": digest(path)}
    (aoi / "source_manifest.json").write_text(json.dumps(source_manifest))
    (aoi / "collection_manifest.json").write_text(json.dumps([{}] * 75))
    (aoi / "qgis_validation.json").write_text(json.dumps({"invalid_layers": [], "subregions": 90}))
    (aoi / "vector_validation.json").write_text(json.dumps({"attributes_preserved": True, "subregions": 90}))
    for resolution in (50, 100, 250, 500, 1000):
        (work / f"validation_{resolution}.json").write_text(json.dumps({"stacks": 15, "failures": []}))
    path = work / "output/05-dem/euregio/buffer_00000m/50m/new.tif"
    path.parent.mkdir(parents=True)
    path.write_text("new sample")
    manifest = [
        {"path": str(p.relative_to(work / "output")), "bytes": p.stat().st_size, "sha256": digest(p)}
        for p in (work / "output").rglob("*")
        if p.is_file()
    ]
    (work / "artifact_manifest.json").write_text(json.dumps(manifest))
    script = Path(__file__).resolve().parents[1] / "publishFram3sGrids.ps1"
    args = [
        str(Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
        "-WorkRoot",
        str(work),
        "-DestinationRoot",
        str(dest),
        "-RecordRoot",
        str(report),
    ]

    def snapshot():
        return {str(p.relative_to(dest)): digest(p) for p in dest.rglob("*") if p.is_file()}

    before = snapshot()
    subprocess.run(args, check=True, capture_output=True)
    assert before == snapshot()

    # A changed input must fail before any source is archived.
    dem = dest / paths["dem"]
    dem.write_text("changed input")
    expected = snapshot()
    failed = subprocess.run(args + ["-Publish"], capture_output=True)
    assert failed.returncode != 0 and snapshot() == expected
    assert not list(dest.rglob("archive"))
    dem.write_text("legacy dem")

    # A collision outside explicitly archived generated datasets must also fail first.
    keep = dest / "03-landcover/raw_preserved.txt"
    keep.write_text("preserve raw input")
    conflicting = {"path": "03-landcover/raw_preserved.txt", "bytes": 0, "sha256": "unused"}
    (work / "artifact_manifest.json").write_text(json.dumps(manifest + [conflicting]))
    expected = snapshot()
    failed = subprocess.run(args + ["-Publish"], capture_output=True)
    assert failed.returncode != 0 and snapshot() == expected
    assert not list(dest.rglob("archive"))
    (work / "artifact_manifest.json").write_text(json.dumps(manifest))

    subprocess.run(args + ["-Publish"], check=True, capture_output=True)
    journal = json.loads(next(report.glob("publication_*.json")).read_text(encoding="utf-8-sig"))
    assert journal["status"] == "complete" and journal["published_files"] == len(manifest)
    assert len(journal["archived_files"]) == 4
    for row in journal["archived_files"]:
        assert digest(Path(row["archive"])) == row["sha256"]
    for row in manifest:
        assert digest(dest / row["path"]) == row["sha256"]
    assert keep.read_text() == "preserve raw input"
    return {
        "dry_run_unchanged": True,
        "changed_source_rejected_before_archive": True,
        "collision_rejected_before_archive": True,
        "archive_hashes_verified": 4,
        "published_hashes_verified": len(manifest),
        "raw_input_preserved": True,
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="fram3s-publication-test-") as temporary:
        print(json.dumps(check_publication(Path(temporary))))
