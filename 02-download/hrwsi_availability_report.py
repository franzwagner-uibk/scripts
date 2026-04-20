from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import boto3
import matplotlib.pyplot as plt


DEFAULT_ENDPOINT = "https://s3.WAW3-2.cloudferro.com"
DEFAULT_BUCKET = "HRWSI"
DEFAULT_ACCESS_KEY = "c4ae60af7b144053803c618a8860f7c9"
DEFAULT_SECRET_KEY = "dcb3ba1f6eab45aaaec5802feef5e2e4"
DEFAULT_TILES = ("32TPS", "32TPT")
OPENAMUNDSEN_DA_RELEVANT_PRODUCTS = {"FSC", "SWS"}


@dataclass(frozen=True)
class ProductStats:
    total_product_dirs: int
    first_date: date | None
    last_date: date | None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create HRWSI availability CSVs and plots over time for selected products/tiles."
        )
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(r"C:\Users\franz\Nextcloud\PhD\02-Daten\hrwsi_availability"),
        help="Output directory for CSV and PNG files.",
    )
    parser.add_argument(
        "--tiles",
        nargs="+",
        default=list(DEFAULT_TILES),
        help="MGRS tiles (e.g. 32TPS 32TPT).",
    )
    parser.add_argument(
        "--products",
        nargs="+",
        default=None,
        help="Optional product list (default: discover all top-level products in bucket).",
    )
    parser.add_argument(
        "--plot-products",
        nargs="+",
        default=["FSC", "SWS"],
        help="Products to include in plots (default: FSC SWS).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help="Optional lower year bound (inclusive).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Optional upper year bound (inclusive).",
    )
    return parser.parse_args()


def _s3_client():
    access_key = os.getenv("HRWSI_ACCESS_KEY", DEFAULT_ACCESS_KEY)
    secret_key = os.getenv("HRWSI_SECRET_KEY", DEFAULT_SECRET_KEY)
    endpoint = os.getenv("HRWSI_ENDPOINT_URL", DEFAULT_ENDPOINT)
    return boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
    )


def _discover_products(client, bucket: str) -> list[str]:
    products: list[str] = []
    token: str | None = None
    while True:
        kwargs = {"Bucket": bucket, "Delimiter": "/"}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        for pref in resp.get("CommonPrefixes", []):
            value = pref.get("Prefix", "").rstrip("/")
            if value:
                products.append(value)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return sorted(set(products))


def _parse_product_date(product_dir: str) -> tuple[date | None, int | None]:
    parts = product_dir.split("/")
    if len(parts) >= 5:
        year_txt, month_txt, day_txt = parts[2], parts[3], parts[4]
        if year_txt.isdigit() and month_txt.isdigit() and day_txt.isdigit():
            try:
                dt = date(int(year_txt), int(month_txt), int(day_txt))
                return dt, dt.year
            except ValueError:
                pass
    if len(parts) >= 3 and parts[2].isdigit():
        year = int(parts[2])
        if 1900 <= year <= 2100:
            return None, year
    return None, None


def _year_in_bounds(year: int | None, start_year: int | None, end_year: int | None) -> bool:
    if year is None:
        return False
    if start_year is not None and year < start_year:
        return False
    if end_year is not None and year > end_year:
        return False
    return True


def _iter_product_dirs(client, bucket: str, product: str, tile: str) -> Iterable[str]:
    prefix = f"{product}/{tile}/"
    paginator = client.get_paginator("list_objects_v2")
    seen_dirs: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if "/" not in key:
                continue
            product_dir = key.rsplit("/", 1)[0]
            if product_dir not in seen_dirs:
                seen_dirs.add(product_dir)
                yield product_dir


def _month_seq_inclusive(start_key: tuple[int, int], end_key: tuple[int, int]) -> list[tuple[int, int]]:
    y, m = start_key
    end_y, end_m = end_key
    out: list[tuple[int, int]] = []
    while (y < end_y) or (y == end_y and m <= end_m):
        out.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _product_color(product: str) -> str:
    token = product.strip().upper()
    if token == "FSC":
        return "#1f77b4"
    if token == "SWS":
        return "#ff7f0e"
    return "#7f7f7f"


def _plot_yearly_barplot(
    *,
    output_png: Path,
    products: list[str],
    years: list[int],
    yearly_counts: dict[tuple[str, int], int],
) -> None:
    if not products or not years:
        return

    n_products = len(products)
    x = list(range(len(years)))
    group_width = 0.85
    bar_width = group_width / max(1, n_products)

    fig_w = max(10.0, 0.7 * len(years))
    fig_h = 6.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for p_idx, product in enumerate(products):
        offsets = [idx - group_width / 2 + (p_idx + 0.5) * bar_width for idx in x]
        counts = [int(yearly_counts.get((product, year), 0)) for year in years]
        color = _product_color(product)
        bars = ax.bar(
            offsets,
            counts,
            width=bar_width * 0.95,
            color=color,
            label=product,
            alpha=0.85,
            zorder=2,
        )
        for bar, count in zip(bars, counts):
            if count == 0:
                bar.set_facecolor("white")
                bar.set_edgecolor("red")
                bar.set_linewidth(1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax.set_title("HRWSI Availability by Product and Year (selected tiles)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Unique product directories")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    leg = ax.legend(title="Product", loc="upper right")
    for txt in leg.get_texts():
        if txt.get_text().strip().upper() in OPENAMUNDSEN_DA_RELEVANT_PRODUCTS:
            txt.set_fontweight("bold")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def _plot_monthly_barplot(
    *,
    output_png: Path,
    products: list[str],
    month_keys: list[tuple[int, int]],
    monthly_counts: dict[tuple[str, tuple[int, int]], int],
) -> None:
    if not products or not month_keys:
        return
    n_products = len(products)
    x = list(range(len(month_keys)))
    group_width = 0.85
    bar_width = group_width / max(1, n_products)
    labels = [f"{y:04d}-{m:02d}" for (y, m) in month_keys]

    fig_w = max(14.0, 0.22 * len(month_keys))
    fig_h = 6.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    for p_idx, product in enumerate(products):
        offsets = [idx - group_width / 2 + (p_idx + 0.5) * bar_width for idx in x]
        counts = [int(monthly_counts.get((product, month_key), 0)) for month_key in month_keys]
        color = _product_color(product)
        bars = ax.bar(
            offsets,
            counts,
            width=bar_width * 0.95,
            color=color,
            label=product,
            alpha=0.85,
            zorder=2,
        )
        for bar, count in zip(bars, counts):
            if count == 0:
                bar.set_facecolor("white")
                bar.set_edgecolor("red")
                bar.set_linewidth(0.8)

    tick_step = max(1, len(month_keys) // 24)
    xticks = list(range(0, len(month_keys), tick_step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([labels[i] for i in xticks], rotation=45, ha="right")
    ax.set_title("HRWSI Availability by Product and Month (date-based products)")
    ax.set_xlabel("Year-Month")
    ax.set_ylabel("Unique product directories")
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    leg = ax.legend(title="Product", loc="upper right")
    for txt in leg.get_texts():
        if txt.get_text().strip().upper() in OPENAMUNDSEN_DA_RELEVANT_PRODUCTS:
            txt.set_fontweight("bold")
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = _s3_client()
    bucket = DEFAULT_BUCKET
    products = [p.upper() for p in args.products] if args.products else _discover_products(client, bucket)
    tiles = sorted({t.strip().upper() for t in args.tiles if str(t).strip()})

    monthly_counts_raw: dict[tuple[str, str, int, int], int] = defaultdict(int)
    yearly_counts_raw: dict[tuple[str, str, int], int] = defaultdict(int)
    summary_rows: list[dict[str, object]] = []

    for product in products:
        for tile in tiles:
            total_dirs = 0
            first_dt: date | None = None
            last_dt: date | None = None
            for product_dir in _iter_product_dirs(client, bucket, product, tile):
                dt, year = _parse_product_date(product_dir)
                if not _year_in_bounds(year, args.start_year, args.end_year):
                    continue
                total_dirs += 1
                if dt is not None:
                    yearly_counts_raw[(product, tile, dt.year)] += 1
                    monthly_counts_raw[(product, tile, dt.year, dt.month)] += 1
                    if first_dt is None or dt < first_dt:
                        first_dt = dt
                    if last_dt is None or dt > last_dt:
                        last_dt = dt
                elif year is not None:
                    yearly_counts_raw[(product, tile, year)] += 1

            stats = ProductStats(total_product_dirs=total_dirs, first_date=first_dt, last_date=last_dt)
            summary_rows.append(
                {
                    "product": product,
                    "tile": tile,
                    "total_product_dirs": stats.total_product_dirs,
                    "first_date": stats.first_date.isoformat() if stats.first_date else "",
                    "last_date": stats.last_date.isoformat() if stats.last_date else "",
                }
            )

    monthly_rows: list[dict[str, object]] = []
    for (product, tile, year, month), count in sorted(monthly_counts_raw.items()):
        monthly_rows.append(
            {
                "product": product,
                "tile": tile,
                "year": year,
                "month": month,
                "product_count": count,
            }
        )

    yearly_rows: list[dict[str, object]] = []
    for (product, tile, year), count in sorted(yearly_counts_raw.items()):
        yearly_rows.append(
            {
                "product": product,
                "tile": tile,
                "year": year,
                "product_count": count,
            }
        )

    _write_csv(
        out_dir / "hrwsi_availability_summary.csv",
        summary_rows,
        ["product", "tile", "total_product_dirs", "first_date", "last_date"],
    )
    _write_csv(
        out_dir / "hrwsi_availability_yearly.csv",
        yearly_rows,
        ["product", "tile", "year", "product_count"],
    )
    _write_csv(
        out_dir / "hrwsi_availability_monthly.csv",
        monthly_rows,
        ["product", "tile", "year", "month", "product_count"],
    )

    yearly_agg: dict[tuple[str, int], int] = defaultdict(int)
    for (product, _tile, year), count in yearly_counts_raw.items():
        yearly_agg[(product, year)] += count

    monthly_agg: dict[tuple[str, tuple[int, int]], int] = defaultdict(int)
    for (product, _tile, year, month), count in monthly_counts_raw.items():
        monthly_agg[(product, (year, month))] += count

    plot_products = [p.strip().upper() for p in args.plot_products if str(p).strip()]
    yearly_agg_plot: dict[tuple[str, int], int] = defaultdict(int)
    for (prod, year), count in yearly_agg.items():
        if prod in plot_products:
            yearly_agg_plot[(prod, year)] += count

    monthly_agg_plot: dict[tuple[str, tuple[int, int]], int] = defaultdict(int)
    for (prod, ym), count in monthly_agg.items():
        if prod in plot_products:
            monthly_agg_plot[(prod, ym)] += count

    products_yearly = [p for p in plot_products if any(prod == p for (prod, _y) in yearly_agg_plot.keys())]
    if not products_yearly:
        products_yearly = plot_products

    years_present = sorted({year for (_prod, year) in yearly_agg_plot.keys()})
    if years_present:
        years = list(range(min(years_present), max(years_present) + 1))
    else:
        years = []

    _plot_yearly_barplot(
        output_png=out_dir / "hrwsi_availability_yearly_barplot.png",
        products=products_yearly,
        years=years,
        yearly_counts=yearly_agg_plot,
    )

    products_monthly = [p for p in plot_products if any(prod == p for (prod, _m) in monthly_agg_plot.keys())]
    if not products_monthly:
        products_monthly = plot_products

    month_keys_present = sorted({month_key for (_prod, month_key) in monthly_agg_plot.keys()})
    if month_keys_present:
        month_keys = _month_seq_inclusive(month_keys_present[0], month_keys_present[-1])
    else:
        month_keys = []

    _plot_monthly_barplot(
        output_png=out_dir / "hrwsi_availability_monthly_barplot.png",
        products=products_monthly,
        month_keys=month_keys,
        monthly_counts=monthly_agg_plot,
    )

    print(f"Done. Files written to: {out_dir}")


if __name__ == "__main__":
    main()
