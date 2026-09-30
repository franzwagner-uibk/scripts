#!/usr/bin/env python3
"""Create or validate the portable Fram3S project using an installed PyQGIS runtime."""

import argparse
import json
import os
import zipfile
import xml.etree.ElementTree as ET
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
    QgsPalettedRasterRenderer,
)
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QColor, QFont, QFontDatabase


def require_arial() -> None:
    """Expose the installed Windows font to Qt's headless font database."""
    if "Arial" not in QFontDatabase().families() and os.name == "nt":
        font = Path(os.environ["SystemRoot"]) / "Fonts/arial.ttf"
        if font.exists():
            QFontDatabase.addApplicationFont(str(font))
    if "Arial" not in QFontDatabase().families():
        raise RuntimeError("Arial is not installed or cannot be loaded")


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
    """Load the central vectors and 75 original-region masks with portable styles."""
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
    require_arial()
    region_group = tree.addGroup("Province and region boundaries")
    for region, title in labels.items():
        vector(
            project,
            region_group,
            root / "aoi.gpkg",
            "boundaries",
            title,
            f"\"region\" = '{region}'",
            "#31515c",
            "0.45",
            visible=region in ("tyrol", "south_tyrol", "trentino"),
        )
    grid_group = tree.addGroup("Region ROI grids and terrain context")
    for region, title in labels.items():
        rg = grid_group.addGroup(title)
        for buffer_m in (0, 5000, 10000):
            bg = rg.addGroup(f"{buffer_m // 1000} km terrain context")
            subset = f'"region" = \'{region}\' AND "buffer_m" = {buffer_m} AND "resolution_m" = 100'
            vector(
                project,
                bg,
                root / "aoi.gpkg",
                "grid_extents",
                f"{title} — {buffer_m // 1000} km context extent",
                subset,
                "#b78647",
                "0.25",
            )
            for resolution in (50, 100, 250, 500, 1000):
                path = (
                    root
                    / region
                    / f"buffer_{buffer_m:05d}m"
                    / f"{resolution}m"
                    / f"roi_{region}_b{buffer_m:05d}_{resolution}.tif"
                )
                raster = QgsRasterLayer(str(path), f"{title} — {buffer_m // 1000} km context — {resolution} m ROI")
                if not raster.isValid():
                    raise RuntimeError(f"Invalid ROI raster: {path}")
                classes = [
                    QgsPalettedRasterRenderer.Class(0, QColor(0, 0, 0, 0), "Outside region"),
                    QgsPalettedRasterRenderer.Class(1, QColor(42, 128, 143, 85), "Original region"),
                ]
                raster.setRenderer(QgsPalettedRasterRenderer(raster.dataProvider(), 1, classes))
                project.addMapLayer(raster, False)
                bg.addLayer(raster).setItemVisibilityChecked(
                    region == "euregio" and buffer_m == 5000 and resolution == 100
                )
            bg.setExpanded(False)
        rg.setExpanded(False)
    sub_group = tree.addGroup("Avalanche-report subregions")
    # Select North Tyrol by province and positive-area intersection, directly from the GPKG.
    boundary_layer = QgsVectorLayer(str(root / "aoi.gpkg") + "|layername=boundaries", "boundaries", "ogr")
    north = next(f.geometry() for f in boundary_layer.getFeatures() if f["region"] == "north_tyrol")
    subregions = QgsVectorLayer(str(root / "aoi.gpkg") + "|layername=subregions", "subregions", "ogr")
    ids = [
        f["id"]
        for f in subregions.getFeatures()
        if f["province"] == "tyrol" and f.geometry().intersection(north).area() > 0
    ]
    north_ids = ",".join("'" + value.replace("'", "''") + "'" for value in ids)
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
            root / "aoi.gpkg",
            "subregions",
            title,
            subset,
            "#62696b",
            "0.13",
            visible=False,
        )
        label = QgsPalLayerSettings()
        label.fieldName = "id"
        text_format = QgsTextFormat()
        text_format.setFont(QFont("Arial"))
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
    sub_group.setExpanded(False)
    if not project.write():
        raise RuntimeError("QGIS project write failed")
    return target


def validate(root: Path, records: Path, render: bool) -> dict:
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
    records.mkdir(parents=True, exist_ok=True)
    rasters = [layer for layer in project.mapLayers().values() if isinstance(layer, QgsRasterLayer)]
    if len(rasters) != 75:
        raise RuntimeError("Expected 75 ROI rasters")
    for layer in rasters:
        renderer = layer.renderer()
        if not isinstance(renderer, QgsPalettedRasterRenderer):
            raise RuntimeError("ROI renderer must use explicit classes")
        colors = {item.value: item.color for item in renderer.classes()}
        if set(colors) != {0, 1} or colors[0].alpha() != 0 or not 0 < colors[1].alpha() < 255:
            raise RuntimeError("Incorrect ROI transparency")
    expected_visible = {"Tyrol (including East Tyrol)", "South Tyrol", "Trentino", "Euregio — 5 km context — 100 m ROI"}
    if {layer.name() for layer in layers} != expected_visible:
        raise RuntimeError(f"Unexpected opening layers: {[layer.name() for layer in layers]}")
    with zipfile.ZipFile(root / "aoi_overview.qgz") as archive:
        xml = archive.read(next(name for name in archive.namelist() if name.endswith(".qgs")))
    if b"Helvetica" in xml:
        raise RuntimeError("Helvetica reference remains")
    document = ET.fromstring(xml)
    for element in document.findall(".//maplayer/datasource"):
        if not element.text.startswith("./"):
            raise RuntimeError(f"Nonrelative layer source: {element.text}")
    for layer in project.mapLayers().values():
        if isinstance(layer, QgsVectorLayer) and layer.labeling():
            if layer.labeling().settings().format().font().family() != "Arial":
                raise RuntimeError("Label font is not Arial")
    require_arial()
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
        if not job.renderedImage().save(str(records / "qgis_project_preview.png")):
            raise RuntimeError("QGIS preview failed")
    result = {
        "roi_rasters": len(rasters),
        "font": "Arial",
        "outside_roi_transparent": True,
        "qgis_version": Qgis.QGIS_VERSION,
        "layers": len(project.mapLayers()),
        "invalid_layers": invalid,
        "subregions": 90,
        "visible_layers": len(layers),
        "relative_paths": True,
        "vector_feature_counts": vector_counts,
    }
    (records / "qgis_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True, help="Validation directory outside delivery")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    app = QgsApplication([], False)
    app.initQgis()
    require_arial()
    if args.validate_only:
        print(json.dumps(validate(args.root, args.records, True)))
    else:
        print(json.dumps({"project": str(build(args.root)), "next": "Reopen with --validate-only"}))
    QgsProject.instance().clear()
    app.exitQgis()


if __name__ == "__main__":
    main()
