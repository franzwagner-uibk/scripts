"""Exercise clean publication, unchanged terrain and failures before archive moves on Windows."""

import ctypes
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
    output = work / "output"
    output.mkdir(parents=True)
    sources = {
        "dem": "05-dem/archive/old/dem.asc",
        "lc": "03-landcover/lc_eusalp/lc.asc",
        "provinces": "01-aoi/archive/provinces.gpkg",
        "subregions": "01-aoi/archive/subregions.gpkg",
    }
    source_manifest = {}
    for kind, relative in sources.items():
        path = dest / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("legacy " + kind)
        source_manifest[kind] = {"path": relative, "sha256": digest(path)}
    hidden = dest / "01-aoi/archive/.DS_Store"
    hidden.write_text("hidden legacy metadata")
    if not ctypes.windll.kernel32.SetFileAttributesW(str(hidden), 2):
        raise ctypes.WinError()
    specs, paths = (
        [],
        {
            "01-aoi/" + name
            for name in ("README.txt", "aoi.gpkg", "aoi_overview.qgz", "aoi_overview.png", "aoi_overview.pdf")
        },
    )
    roots = {"roi": "01-aoi", "lc": "03-landcover", "dem": "05-dem", "srf": "06-srf", "svf": "07-svf"}
    for region in ("euregio", "tyrol", "north_tyrol", "south_tyrol", "trentino"):
        for context in (0, 5000, 10000):
            domain = f"{region}_b{context:05d}"
            variant = f"{region}/buffer_{context:05d}m"
            paths.add(f"01-aoi/{variant}/{domain}.zip")
            for res in (50, 100, 250, 500, 1000):
                specs.append({"region": region, "buffer_m": context, "resolution_m": res})
                for kind, folder in roots.items():
                    paths.update(
                        f"{folder}/{variant}/{res}m/{kind}_{domain}_{res}.{ext}" for ext in ("tif", "asc", "prj")
                    )
    for relative in paths:
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    manifest = [
        {"path": str(p.relative_to(output)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": digest(p)}
        for p in output.rglob("*")
        if p.is_file()
    ]
    preserved = "05-dem/euregio/buffer_00000m/50m/dem_euregio_b00000_50.tif"
    target = dest / preserved
    target.parent.mkdir(parents=True)
    target.write_bytes((output / preserved).read_bytes())
    original_mtime = target.stat().st_mtime_ns
    aux = target.with_suffix(".asc.aux.xml")
    aux.write_text("optional metadata")
    records = {
        "source_manifest": source_manifest,
        "collection_manifest": [{}] * 75,
        "grid_specifications": specs,
        "qgis_validation": {"invalid_layers": [], "subregions": 90, "roi_rasters": 75},
        "vector_validation": {"attributes_preserved": True, "subregions": 90},
        "roi_context_validation": {"failures": [], "comparisons": 50},
        "delivery_validation": {"files": 1145, "failures": []},
        "artifact_manifest": manifest,
        **{f"validation_{res}": {"stacks": 15, "failures": []} for res in (50, 100, 250, 500, 1000)},
    }
    for name, value in records.items():
        (work / f"{name}.json").write_text(json.dumps(value))
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
    child_env = {key: value for key, value in os.environ.items() if key.upper() != "PSMODULEPATH"}

    def run(*extra):
        return subprocess.run(args + list(extra), capture_output=True, env=child_env)

    def snapshot():
        return {str(p.relative_to(root)): digest(p) for p in dest.rglob("*") if p.is_file()}

    before = snapshot()
    dry = run()
    if dry.returncode:
        raise RuntimeError(dry.stderr.decode(errors="replace"))
    assert snapshot() == before
    dem = dest / sources["dem"]
    dem.write_text("changed input")
    expected = snapshot()
    failed = run("-Publish")
    assert failed.returncode and b"Source changed since staging" in failed.stderr and snapshot() == expected
    dem.write_text("legacy dem")
    target.write_text("changed terrain")
    expected = snapshot()
    failed = run("-Publish")
    assert failed.returncode and b"Existing terrain differs" in failed.stderr and snapshot() == expected
    target.write_bytes((output / preserved).read_bytes())
    original_mtime = target.stat().st_mtime_ns
    unknown = dest / "03-landcover/unknown.dat"
    unknown.write_text("do not remove")
    expected = snapshot()
    failed = run("-Publish")
    assert failed.returncode and b"Unexpected active file" in failed.stderr and snapshot() == expected
    unknown.unlink()
    result = run("-Publish")
    if result.returncode:
        raise RuntimeError(result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace"))
    journal = json.loads(next(report.glob("publication_*.json")).read_text(encoding="utf-8-sig"))
    assert journal["status"] == "complete" and journal["published_files"] == 1144
    assert len(journal["preserved_files"]) == 1 and target.stat().st_mtime_ns == original_mtime
    assert len(journal["archived_files"]) == 6
    for row in journal["archived_files"]:
        assert Path(row["archive"]).stat().st_size == row["bytes"] and not Path(row["source"]).exists()
    for relative, entry in source_manifest.items():
        assert digest(Path(journal["archive_root"]) / entry["path"]) == entry["sha256"]
    for row in manifest:
        assert digest(dest / row["path"]) == row["sha256"]
    assert len([p for p in dest.rglob("*") if p.is_file()]) == 1145
    # Simulate interruption after archival and some copies, then verify safe continuation.
    journal_path = next(report.glob("publication_*.json"))
    journal["status"] = "publishing"
    journal_path.write_text(json.dumps(journal))
    unfinished = dest / journal["copied_files"][0]["path"]
    unfinished.write_text("unexpected changed file")
    expected = snapshot()
    failed = run("-Publish", "-ResumeJournal", str(journal_path))
    assert failed.returncode and b"Publication verification failed" in failed.stderr and snapshot() == expected
    unfinished.unlink()
    resumed = run("-Publish", "-ResumeJournal", str(journal_path))
    if resumed.returncode:
        raise RuntimeError(resumed.stderr.decode(errors="replace"))
    assert json.loads(journal_path.read_text(encoding="utf-8-sig"))["status"] == "complete"
    assert target.stat().st_mtime_ns == original_mtime
    for row in manifest:
        assert digest(dest / row["path"]) == row["sha256"]
    return {
        "dry_run_unchanged": True,
        "changed_source_rejected": True,
        "changed_terrain_rejected": True,
        "unexpected_file_rejected": True,
        "raw_sources_preserved": True,
        "terrain_kept_in_place": True,
        "verified_delivery_files": 1145,
        "verified_archive_files": 6,
        "hidden_archive_file_verified": True,
        "resume_rejects_changed_file": True,
        "resume_completes_without_overwriting": True,
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="fram3s-publication-test-") as temporary:
        short_path = ctypes.create_unicode_buffer(32768)
        if not ctypes.windll.kernel32.GetShortPathNameW(temporary, short_path, len(short_path)):
            raise ctypes.WinError()
        print(json.dumps(check_publication(Path(short_path.value))))
