#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: clip_ascii_by_shapefile_extent.py
Author: Franz Wagner
Date: 2025-10-09 (Europe/Vienna)

Description
-----------
Extracts a subgrid from an ESRI ASCII grid (.asc) using the extent (bounding box)
of a shapefile, and writes the result as ESRI ASCII.

Assumptions:
- All inputs share the same spatial reference.
- Input ASCII follows ESRI ASCII Grid spec (ncols, nrows, xllcorner/xllcenter, yllcorner/yllcenter, cellsize, NODATA_value).
- Data rows in ASCII are ordered from TOP to BOTTOM (standard ESRI ASCII).

User variables to set:
- INPUT_ASC      : path to input .asc
- INPUT_SHP      : path to shapefile (.shp)
- OUTPUT_ASC     : path to output .asc
"""

import os
import sys
import math
import numpy as np
import geopandas as gpd

# =========================
# ===== USER SETTINGS =====
# =========================

INPUT_ASC  = r"F:\fram3s\01-data\03-landcover\lc_eusalp\openAMUNDSEN-euregio\lc_euregio_20_eusalp.asc"
INPUT_SHP  = r"F:\fram3s\01-data\01-aoi\TESTSITES\rofental.shp"   # only the extent (bbox) is used
OUTPUT_ASC = r"F:\fram3s\01-data\03-landcover\lc_eusalp\openAMUNDSEN-euregio\lc_rofental_20_eusalp.asc"


# =========================
# ======= FUNCTIONS =======
# =========================

def read_esri_ascii_header(path):
    """Read ESRI ASCII header and return dict and number of header lines."""
    header_keys = {"ncols","nrows","xllcorner","yllcorner","xllcenter","yllcenter","cellsize","NODATA_value"}
    header = {}
    header_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] in header_keys:
                k = parts[0]
                v = parts[1]
                # ints where appropriate
                if k in ("ncols","nrows"):
                    header[k] = int(v)
                elif k in ("cellsize","xllcorner","yllcorner","xllcenter","yllcenter","NODATA_value"):
                    header[k] = float(v)
                header_lines += 1
                if len(header) >= 6:  # minimal header complete
                    # Stop at first non-header line
                    continue
            else:
                # first data line reached
                break
    # sanity
    if "ncols" not in header or "nrows" not in header or "cellsize" not in header:
        raise ValueError("Invalid ESRI ASCII header: missing ncols/nrows/cellsize.")
    # corner/center check
    has_corner = ("xllcorner" in header and "yllcorner" in header)
    has_center = ("xllcenter" in header and "yllcenter" in header)
    if not (has_corner or has_center):
        raise ValueError("Header must define either (xllcorner,yllcorner) or (xllcenter,yllcenter).")
    return header, header_lines

def header_xyll_to_corner(header):
    """Return (xmin, ymin), (is_center) using corner coordinates."""
    cs = header["cellsize"]
    if "xllcorner" in header and "yllcorner" in header:
        xmin = header["xllcorner"]
        ymin = header["yllcorner"]
        return xmin, ymin, False
    # center -> convert to corner
    xmin = header["xllcenter"] - cs*0.5
    ymin = header["yllcenter"] - cs*0.5
    return xmin, ymin, True

def compute_grid_extent(header):
    """Compute (xmin, ymin, xmax, ymax) from header."""
    ncols = header["ncols"]; nrows = header["nrows"]; cs = header["cellsize"]
    xmin, ymin, _ = header_xyll_to_corner(header)
    xmax = xmin + ncols * cs
    ymax = ymin + nrows * cs
    return xmin, ymin, xmax, ymax

def load_ascii_data(path, skiprows, nrows, ncols):
    """Load ESRI ASCII data block into (nrows, ncols) numpy array."""
    data = np.loadtxt(path, skiprows=skiprows)
    if data.size != nrows * ncols:
        # np.loadtxt may already parse as (nrows, ncols)
        data = data.reshape((nrows, ncols))
    if data.ndim == 1:
        data = data.reshape((nrows, ncols))
    return data

def write_esri_ascii(path, header, data, use_center=False):
    """Write ESRI ASCII with either xllcorner/yllcorner or xllcenter/yllcenter preserved."""
    nrows, ncols = data.shape
    cs = header["cellsize"]
    nodata = header.get("NODATA_value", -9999.0)
    # we must recompute xll/yll for the subset (already set in header before calling)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"ncols         {ncols}\n")
        f.write(f"nrows         {nrows}\n")
        if use_center:
            f.write(f"xllcenter     {header['xllcenter']:.8f}\n")
            f.write(f"yllcenter     {header['yllcenter']:.8f}\n")
        else:
            f.write(f"xllcorner     {header['xllcorner']:.8f}\n")
            f.write(f"yllcorner     {header['yllcorner']:.8f}\n")
        f.write(f"cellsize      {cs:.8f}\n")
        f.write(f"NODATA_value  {nodata:.8f}\n")
        # ESRI ASCII prints rows from TOP to BOTTOM (north to south)
        for r in range(nrows):
            row = data[r, :]
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

def clip_indices(xmin, ymin, xmax, ymax, cs, ncols, nrows, clip_minx, clip_miny, clip_maxx, clip_maxy):
    """
    Compute inclusive indices [row_start,row_end], [col_start,col_end] for clipping.
    Rows indexed from TOP (0) to BOTTOM (nrows-1).
    """
    # Clamp clip bbox to grid bbox
    ixmin = max(xmin, clip_minx)
    iymin = max(ymin, clip_miny)
    ixmax = min(xmax, clip_maxx)
    iymax = min(ymax, clip_maxy)
    if ixmin >= ixmax or iymin >= iymax:
        return None  # no overlap

    # columns (left->right)
    col_start = int(math.floor((ixmin - xmin) / cs))
    col_end   = int(math.ceil ((ixmax - xmin) / cs)) - 1

    # rows: data is top->bottom; ymax at top
    # row for a given y is floor((ymax - y)/cs)
    row_start = int(math.floor((ymax - iymax) / cs))  # top index
    row_end   = int(math.ceil ((ymax - iymin) / cs)) - 1

    # clamp to grid
    col_start = max(0, min(col_start, ncols-1))
    col_end   = max(0, min(col_end,   ncols-1))
    row_start = max(0, min(row_start, nrows-1))
    row_end   = max(0, min(row_end,   nrows-1))

    # ensure order
    if col_end < col_start or row_end < row_start:
        return None

    return row_start, row_end, col_start, col_end, ixmin, iymin, ixmax, iymax

def main():
    # ---- read shapefile extent ----
    if not os.path.isfile(INPUT_SHP):
        raise FileNotFoundError(f"Shapefile not found: {INPUT_SHP}")
    gdf = gpd.read_file(INPUT_SHP)
    if gdf.empty:
        raise ValueError("Shapefile has no features.")
    clip_minx, clip_miny, clip_maxx, clip_maxy = gdf.total_bounds  # (minx, miny, maxx, maxy)

    # ---- read ascii header + data ----
    if not os.path.isfile(INPUT_ASC):
        raise FileNotFoundError(f"Input ASCII not found: {INPUT_ASC}")
    hdr, hdr_lines = read_esri_ascii_header(INPUT_ASC)
    ncols, nrows, cs = hdr["ncols"], hdr["nrows"], hdr["cellsize"]
    xmin, ymin, xmax, ymax = compute_grid_extent(hdr)
    data = load_ascii_data(INPUT_ASC, skiprows=hdr_lines, nrows=nrows, ncols=ncols)

    # ---- compute clip indices ----
    res = clip_indices(xmin, ymin, xmax, ymax, cs, ncols, nrows,
                       clip_minx, clip_miny, clip_maxx, clip_maxy)
    if res is None:
        raise ValueError("No spatial overlap between ASCII grid and shapefile extent.")
    row_start, row_end, col_start, col_end, ixmin, iymin, ixmax, iymax = res

    # ---- slice data ----
    sub = data[row_start:row_end+1, col_start:col_end+1]

    # ---- prepare output header (preserve corner/center style) ----
    # Compute lower-left of the subset:
    # rows indexed from top: yll_out = ymax - (row_end+1)*cs
    yll_out = ymax - (row_end + 1) * cs
    xll_out = xmin + col_start * cs

    out_hdr = {
        "ncols": sub.shape[1],
        "nrows": sub.shape[0],
        "cellsize": cs,
        "NODATA_value": hdr.get("NODATA_value", -9999.0),
    }

    # preserve original corner vs center keys
    _, _, used_center = header_xyll_to_corner(hdr)
    if used_center:
        out_hdr["xllcenter"] = xll_out + cs * 0.5
        out_hdr["yllcenter"] = yll_out + cs * 0.5
    else:
        out_hdr["xllcorner"] = xll_out
        out_hdr["yllcorner"] = yll_out

    # ---- write out ----
    os.makedirs(os.path.dirname(OUTPUT_ASC) or ".", exist_ok=True)
    write_esri_ascii(OUTPUT_ASC, out_hdr, sub, use_center=used_center)
    print(f"Clipped ASCII written to: {OUTPUT_ASC}")
    print(f"Subset shape: {sub.shape[0]} rows x {sub.shape[1]} cols")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
