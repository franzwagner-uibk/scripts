"""Authoritative boundaries, buffers, ROI partitions and subregion exports."""

import zipfile
from pathlib import Path

import geopandas as gpd
from shapely import make_valid
from shapely.geometry import box
from shapely.ops import unary_union

from fram3s_grids.common import BUFFERS, CRS, REGIONS, RESOLUTIONS, domain, envelope, grid, save_json, variant_dir


def read_regions(source: Path) -> tuple:
    """Repair only invalid province geometry, retaining the source boundary detail."""
    data = gpd.read_file(source / "01-aoi/PROVINCE_BOUNDARY/province_boundary_4326.gpkg").to_crs(CRS)
    regions, repairs = {}, []
    for _, row in data.iterrows():
        original = row.geometry
        repaired = original if original.is_valid else make_valid(original)
        if repaired.geom_type not in ("Polygon", "MultiPolygon") or not repaired.is_valid:
            raise ValueError(f"Unexpected repaired geometry for {row.province}")
        regions[row.province] = repaired
        repairs.append(
            {"region": row.province, "repaired": not original.is_valid, "area_change_m2": repaired.area - original.area}
        )
    original_tyrol = data.loc[data.province == "tyrol"].geometry.iloc[0]
    north = max(original_tyrol.geoms, key=lambda geom: geom.area)
    if not north.is_valid:
        raise ValueError("North Tyrol source unexpectedly invalid")
    regions["north_tyrol"] = north
    regions["euregio"] = unary_union([regions[name] for name in ("tyrol", "south_tyrol", "trentino")])
    return regions, repairs


def variants(regions: dict):
    """Yield every region and true polygon buffer with its shared envelope."""
    for name in REGIONS:
        for buffer_m in BUFFERS:
            geom = regions[name].buffer(buffer_m, quad_segs=64) if buffer_m else regions[name]
            yield name, buffer_m, geom, envelope(geom.bounds)


def build_geometry(source: Path, output: Path) -> list:
    """Write vector products and return the 75 authoritative grid specifications."""
    root = output / "01-aoi"
    root.mkdir(parents=True, exist_ok=True)
    regions, repairs = read_regions(source)
    specifications, boundaries, extents = [], [], []
    for name, buffer_m, geom, bounds in variants(regions):
        rectangle = box(*bounds)
        if not rectangle.covers(geom) or not geom.is_valid:
            raise ValueError(f"Invalid target geometry: {name}, {buffer_m}")
        folder = variant_dir(root, name, buffer_m) / "vectors"
        folder.mkdir(parents=True, exist_ok=True)
        prefix = domain(name, buffer_m)
        gpkg = folder / f"{prefix}.gpkg"
        if gpkg.exists():
            gpkg.unlink()
        attrs = {"region": [name], "buffer_m": [buffer_m]}
        gpd.GeoDataFrame(attrs, geometry=[geom], crs=CRS).to_file(gpkg, layer="boundary", driver="GPKG")
        gpd.GeoDataFrame(attrs, geometry=[rectangle], crs=CRS).to_file(gpkg, layer="extent", driver="GPKG")
        partition = gpd.GeoDataFrame({"ROI": [1, 0]}, geometry=[geom, rectangle.difference(geom)], crs=CRS)
        if partition.geometry.iloc[0].intersection(partition.geometry.iloc[1]).area > 0.001:
            raise ValueError("ROI partition overlaps")
        if unary_union(partition.geometry).symmetric_difference(rectangle).area > 0.001:
            raise ValueError("ROI partition does not cover rectangle")
        partition.to_file(gpkg, layer="roi_partition", driver="GPKG")
        shpdir = folder / "shapefile"
        shpdir.mkdir(exist_ok=True)
        partition.to_file(shpdir / f"{prefix}.shp", driver="ESRI Shapefile", encoding="UTF-8")
        with zipfile.ZipFile(folder / f"{prefix}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for item in sorted(shpdir.glob(f"{prefix}.*")):
                archive.write(item, item.name)
        boundaries.append({"region": name, "buffer_m": buffer_m, "geometry": geom})
        for res in RESOLUTIONS:
            shape, _ = grid(bounds, res)
            spec = {
                "region": name,
                "buffer_m": buffer_m,
                "resolution_m": res,
                "domain": prefix,
                "bounds": list(bounds),
                "nrows": shape[0],
                "ncols": shape[1],
            }
            specifications.append(spec)
            extents.append({**{key: val for key, val in spec.items() if key != "bounds"}, "geometry": rectangle})
    overview = root / "aoi_overview.gpkg"
    if overview.exists():
        overview.unlink()
    gpd.GeoDataFrame(boundaries, crs=CRS).to_file(overview, layer="boundaries", driver="GPKG")
    gpd.GeoDataFrame(extents, crs=CRS).to_file(overview, layer="grid_extents", driver="GPKG")
    subregions = gpd.read_file(source / "01-aoi/SUBREGIONS/raw/subregions_avalanche_report_4326_raw.gpkg").to_crs(CRS)
    if len(subregions) != 90 or not subregions.geometry.is_valid.all():
        raise ValueError("Subregion count or geometry differs from audited source")
    north_ids = subregions.loc[subregions.geometry.intersection(regions["north_tyrol"]).area > 0, "id"].tolist()
    subdir = root / "subregions"
    subdir.mkdir(exist_ok=True)
    subpath = subdir / "subregions_25832.gpkg"
    if subpath.exists():
        subpath.unlink()
    subregions.to_file(subpath, layer="subregions", driver="GPKG")
    save_json(
        root / "geometry_validation.json",
        {
            "repairs": repairs,
            "subregions": len(subregions),
            "north_tyrol_subregion_ids": north_ids,
            "polygon_variants": len(boundaries),
            "grid_stacks": len(specifications),
        },
    )
    save_json(root / "grid_specifications.json", specifications)
    return specifications
