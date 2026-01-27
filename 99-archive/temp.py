import pandas as pd
from pathlib import Path

IN_DIR = Path(r"C:\Daten\PhD\02-Daten\Rofental_snow_depth_swe")
OUT_DIR = Path(r"C:\Daten\PhD\02-Daten\Tirol_snow_depth")

FILES = [
    ("LB_10min_2013-2025.csv", "latschbloder.csv"),
    ("PD_10min_2019-2025.csv", "proviantdepot.csv"),
]


def find_time_col(cols):
    lower = [c.strip().lower() for c in cols]
    # prefer any "date and time" column
    for i, name in enumerate(lower):
        if "date and time" in name:
            return cols[i]
    for cand in ("time", "date"):
        if cand in lower:
            return cols[lower.index(cand)]
    return None


def find_depth_col(cols):
    lower = [c.strip().lower() for c in cols]
    # first column containing "snow depth"
    for i, name in enumerate(lower):
        if "snow depth" in name:
            return cols[i]
    return None


def find_swe_col(cols):
    lower = [c.strip().lower() for c in cols]

    # prefer SSG-2 SWE, then SPA S1, then SPA S2
    priority_patterns = [
        "ssg-2 snow water equivalent",
        "spa snow water equivalent s1",
        "spa snow water equivalent s2",
    ]
    for pat in priority_patterns:
        for i, name in enumerate(lower):
            if pat in name:
                return cols[i]

    # fallback: first any "snow water equivalent"
    for i, name in enumerate(lower):
        if "snow water equivalent" in name:
            return cols[i]
    return None


def convert_file(in_name, out_name):
    in_path = IN_DIR / in_name
    out_path = OUT_DIR / out_name
    print(f"Processing {in_path} -> {out_path}")

    df = pd.read_csv(in_path, sep=";", na_values=["", " ", "NA", "NaN"])
    original_cols = df.columns.tolist()
    df.columns = [c.strip() for c in df.columns]

    time_col = find_time_col(df.columns)
    depth_col = find_depth_col(df.columns)
    swe_col = find_swe_col(df.columns)

    print(f"  time_col  = {time_col}")
    print(f"  depth_col = {depth_col}")
    print(f"  swe_col   = {swe_col}")

    if time_col is None:
        print("  No time column found, skipping.")
        return

    # parse time and set index
    df["time"] = pd.to_datetime(df[time_col].astype(str).str.strip(),
                                errors="coerce",
                                format="%Y-%m-%d %H:%M:%S")
    df = df.dropna(subset=["time"]).set_index("time")

    # depth in m from mm
    if depth_col is not None:
        df["snow_depth"] = (
            pd.to_numeric(df[depth_col], errors="coerce") / 1000.0
        )
    else:
        df["snow_depth"] = pd.NA

    # SWE in m from mm
    if swe_col is not None:
        df["swe"] = (
            pd.to_numeric(df[swe_col], errors="coerce") / 1000.0
        )
    else:
        df["swe"] = pd.NA

    hourly = df[["snow_depth", "swe"]].resample("H").mean()
    hourly = hourly.reset_index()
    hourly["time"] = hourly["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(out_path, index=False, columns=["time", "snow_depth", "swe"])
    print(f"  Wrote {len(hourly)} rows.")


def main():
    for in_name, out_name in FILES:
        convert_file(in_name, out_name)


if __name__ == "__main__":
    main()
