#!/usr/bin/env python3
"""Create or validate the portable Fram3S project using an installed PyQGIS runtime."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsFillSymbol,
    QgsMapRendererParallelJob,
    QgsMapSettings,
    QgsPalLayerSettings,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsTextFormat,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
    QgsReferencedRectangle,
)
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QColor


def vector(project, group, path, layer_name, title, subset="", color="#216b80", width="0.25", visible=False):
    """Add a styled vector layer without copying the underlying dataset."""
    layer = QgsVectorLayer(str(path) + "|layername=" + layer_name, title, "ogr")
    if not layer.isValid():
        raise RuntimeError(f"Invalid vector layer: {path}, {layer_name}")
    if subset and not layer.setSubsetString(subset):
        raise RuntimeError(f"Invalid filter: {subset}")
    layer.renderer().setSymbol(
        QgsFillSymbol.createSimple(
            {"color": "transparent", "outline_color": color, "outline_width": width, "joinstyle": "round"}
        )
    )
    project.addMapLayer(layer, False)
    node = group.addLayer(layer)
    node.setItemVisibilityChecked(visible)
    return layer


def build(root: Path) -> Path:
    """Group all polygon variants, 75 ROI footprints/rasters and five subregion views."""
    project = QgsProject.instance()
    project.clear()
    project.setCrs(QgsCoordinateReferenceSystem("EPSG:25832"))
    project.setFilePathStorage(Qgis.FilePathType.Relative)
    target = root / "aoi_overview.qgz"
    project.setFileName(str(target))
    project.viewSettings().setDefaultViewExtent(
        QgsReferencedRectangle(QgsRectangle(573000, 5049000, 813000, 5303000), project.crs())
    )
    tree = project.layerTreeRoot()
    labels = {
        "euregio": "Euregio",
        "tyrol": "Tyrol (including East Tyrol)",
        "north_tyrol": "North Tyrol",
        "south_tyrol": "South Tyrol",
        "trentino": "Trentino",
    }
    colors = {0: "#177e89", 5000: "#e79532", 10000: "#b04a82"}
    region_group = tree.addGroup("Regions, buffers and model grids")
    for region, title in labels.items():
        rg = region_group.addGroup(title)
        for buffer_m in (0, 5000, 10000):
            bg = rg.addGroup("Original boundary" if buffer_m == 0 else f"{buffer_m // 1000} km true buffer")
            subset = f'"region" = \'{region}\' AND "buffer_m" = {buffer_m}'
            vector(
                project,
                bg,
                root / "aoi_overview.gpkg",
                "boundaries",
                "ROI polygon",
                subset,
                colors[buffer_m],
                "0.4",
                visible=(buffer_m == 0),
            )
            for resolution in (50, 100, 250, 500, 1000):
                group = bg.addGroup(f"{resolution} m")
                vector(
                    project,
                    group,
                    root / "aoi_overview.gpkg",
                    "grid_extents",
                    "Grid extent",
                    subset + f' AND "resolution_m" = {resolution}',
                    colors[buffer_m],
                    "0.2",
                    False,
                )
                filename = f"roi_{region}_b{buffer_m:05d}_{resolution}.tif"
                path = root / region / f"buffer_{buffer_m:05d}m" / f"{resolution}m" / filename
                raster = QgsRasterLayer(str(path), "ROI raster (1 inside, 0 outside)")
                if not raster.isValid():
                    raise RuntimeError(f"Invalid ROI raster: {path}")
                project.addMapLayer(raster, False)
                node = group.addLayer(raster)
                node.setItemVisibilityChecked(False)
                group.setExpanded(False)
            bg.setExpanded(False)
        rg.setExpanded(False)
    region_group.setExpanded(True)
    sub_group = tree.addGroup("Avalanche-report subregions")
    geometry = json.loads((root / "geometry_validation.json").read_text())
    north_ids = ",".join("'" + value.replace("'", "''") + "'" for value in geometry["north_tyrol_subregion_ids"])
    subsets = {
        "All Euregio subregions": "",
        "Tyrol (including East Tyrol)": "\"province\" = 'tyrol'",
        "North Tyrol": f'"id" IN ({north_ids})',
        "South Tyrol": "\"province\" = 'south_tyrol'",
        "Trentino": "\"province\" = 'trentino'",
    }
    # The all-Euregio layer is the Euregio view; do not create a redundant sixth copy.
    for title, subset in subsets.items():
        layer = vector(
            project,
            sub_group,
            root / "subregions/subregions_25832.gpkg",
            "subregions",
            title,
            subset,
            "#62696b",
            "0.13",
            visible=(subset == ""),
        )
        label = QgsPalLayerSettings()
        label.fieldName = "id"
        text_format = QgsTextFormat()
        text_format.setSize(8)
        label.setFormat(text_format)
        layer.setLabeling(QgsVectorLayerSimpleLabeling(label))
        layer.setLabelsEnabled(False)
        for name, alias in {
            "id": "Subregion identifier",
            "province": "Province",
            "country": "Country",
            "start_date": "Valid from",
            "end_date": "Valid until",
        }.items():
            index = layer.fields().indexFromName(name)
            if index >= 0:
                layer.setFieldAlias(index, alias)
    kathi = tree.addGroup("Kathi: original North Tyrol on the 100 m, 5 km-context grid")
    core = root / "kathi_north_tyrol_100m"
    vector(project, kathi, core / "north_tyrol_core.gpkg", "north_tyrol_core", "Original North Tyrol boundary")
    raster = QgsRasterLayer(str(core / "roi_north_tyrol_core_100.tif"), "Core ROI mask")
    if not raster.isValid():
        raise RuntimeError("Invalid Kathi core mask")
    project.addMapLayer(raster, False)
    kathi.addLayer(raster).setItemVisibilityChecked(False)
    kathi.setExpanded(False)
    region_group.setItemVisibilityChecked(True)
    if not project.write():
        raise RuntimeError("QGIS project write failed")
    return target


def validate(root: Path, render: bool) -> dict:
    """Reopen the saved project, verify every provider and render its visible layers."""
    project = QgsProject.instance()
    project.clear()
    if not project.read(str(root / "aoi_overview.qgz")):
        raise RuntimeError("QGIS project read failed")
    invalid = [layer.name() for layer in project.mapLayers().values() if not layer.isValid()]
    if invalid:
        raise RuntimeError(f"Invalid saved project layers: {invalid}")
    all_subregions = [layer for layer in project.mapLayers().values() if layer.name() == "All Euregio subregions"]
    if len(all_subregions) != 1 or all_subregions[0].featureCount() != 90:
        raise RuntimeError("Missing or incomplete subregion layer")
    vector_counts = {}
    for layer in project.mapLayers().values():
        if isinstance(layer, QgsVectorLayer):
            count = 0
            for feature in layer.getFeatures():
                if not feature.hasGeometry() or feature.geometry().isEmpty():
                    raise RuntimeError(f"Empty feature in {layer.name()}")
                count += 1
            if count != layer.featureCount() or count == 0:
                raise RuntimeError(f"Unreadable/empty features in {layer.name()}: {count}")
            vector_counts[layer.id()] = count
    layers = [node.layer() for node in project.layerTreeRoot().findLayers() if node.isVisible()]
    if render:
        settings = QgsMapSettings()
        settings.setLayers(layers)
        settings.setDestinationCrs(QgsCoordinateReferenceSystem("EPSG:25832"))
        rectangle = QgsRectangle(573000, 5049000, 813000, 5303000)
        rectangle.scale(1.05)
        settings.setExtent(rectangle)
        settings.setOutputSize(QSize(1600, 1700))
        settings.setBackgroundColor(QColor("white"))
        job = QgsMapRendererParallelJob(settings)
        job.start()
        job.waitForFinished()
        if not job.renderedImage().save(str(root / "qgis_project_preview.png")):
            raise RuntimeError("QGIS preview failed")
    result = {
        "qgis_version": Qgis.QGIS_VERSION,
        "layers": len(project.mapLayers()),
        "invalid_layers": invalid,
        "subregions": 90,
        "visible_layers": len(layers),
        "relative_paths": True,
        "vector_feature_counts": vector_counts,
    }
    (root / "qgis_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    app = QgsApplication([], False)
    app.initQgis()
    if not args.validate_only:
        build(args.root)
    print(json.dumps(validate(args.root, not args.validate_only)))
    QgsProject.instance().clear()
    app.exitQgis()


if __name__ == "__main__":
    main()
