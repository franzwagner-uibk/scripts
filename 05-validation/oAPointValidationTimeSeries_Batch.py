import logging
import os
import re
import matplotlib.pyplot as plt
import pandas as pd

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# General configuration
STATIONS_CSV_PATH = r"C:\Daten\PhD\09-openAMUNDSEN\testsite\meteo\stations.csv"
OUTPUT_PLOT_DIR = r"C:\Daten\PhD\09-openAMUNDSEN\testsite\validation"
START_DATE = "2013-09-01"
END_DATE = "2023-08-31"

# Accept different naming conventions; values are normalized to this name
VARIABLE_TO_PLOT = "depth"
VARIABLE_ALIASES = {
    # Model vs observation naming differences
    "depth": ["snow_depth", "snow.depth", "snow_height"],
    "swe": ["snow_swe", "snow.swe"],
}

# Input CSV directories (must contain matching station IDs)
INPUT_CSV_DIRS = [
    r"C:\Daten\PhD\09-openAMUNDSEN\testsite\results_short",
    r"C:\Daten\PhD\02-Daten\Tirol_snow_depth",
]
LEGEND_ENTRIES = [
    "openAMUNDSEN",
    "observation",
]
AGGREGATIONS = [
    ("day", "mean"),
    ("day", "mean"),
]


def load_csv(file_path: str, variable: str) -> pd.DataFrame:
    """Load CSV, lower-case headers, accept time/date, normalize snow column."""
    logging.info(f"Reading input file: {file_path}")
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()

    time_col = None
    if "time" in df.columns:
        time_col = "time"
    elif "date" in df.columns:
        time_col = "date"
    if time_col:
        # strip whitespace/non-breaking spaces before parsing
        df[time_col] = df[time_col].astype(str).str.strip()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", format="%Y-%m-%d %H:%M:%S")
        df = df.set_index(time_col)
        logging.info(f"Time range for {os.path.basename(file_path)}: {df.index.min()} -> {df.index.max()} (non-null {df.index.notna().sum()})")
    else:
        logging.warning(f"No 'time' or 'date' column found in {file_path}; skipping")
        return pd.DataFrame()

    # Normalize variable column names using known aliases
    alias_candidates = [variable] + VARIABLE_ALIASES.get(variable, [])
    found_alias = next((col for col in alias_candidates if col in df.columns), None)
    if found_alias and found_alias != variable:
        df = df.rename(columns={found_alias: variable})
    elif not found_alias and variable == "depth" and "snow_height" in df.columns:
        df = df.rename(columns={"snow_height": variable})

    # Ensure the variable column is numeric if present
    if variable in df.columns:
        df[variable] = pd.to_numeric(df[variable], errors="coerce").fillna(0)
        # If values look like centimeters (e.g., > 20), convert to meters to match model output
        if variable == "depth":
            max_val = df[variable].max()
            if max_val > 20:
                logging.info(f"Converting {os.path.basename(file_path)} {variable} from cm to m (max was {max_val})")
                df[variable] = df[variable] / 100.0

    return df


def filter_data_by_timespan(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df[(df.index >= start_date) & (df.index <= end_date)]


def aggregate_data(df: pd.DataFrame, aggregation):
    if df.empty or aggregation is None:
        return df

    interval, method = aggregation
    if interval == "day":
        resampled = df.resample("D")
    elif interval == "hour":
        resampled = df.resample("H")
    elif interval == "month":
        resampled = df.resample("ME")
    elif interval == "year":
        resampled = df.resample("YE")
    else:
        logging.warning(f"Unknown aggregation interval '{interval}', skipping")
        return df

    if method == "sum":
        return resampled.sum()
    if method == "mean":
        return resampled.mean()
    if method == "max":
        return resampled.max()

    logging.warning(f"Unknown aggregation method '{method}', skipping")
    return df


def load_stations(csv_path: str):
    """Load station metadata; return empty dict if missing."""
    if not os.path.exists(csv_path):
        logging.warning(f"Station metadata not found: {csv_path}")
        return {}
    stations_df = pd.read_csv(csv_path)
    station_map = {}
    for _, row in stations_df.iterrows():
        # store altitude as numeric when possible
        alt_raw = row.get("alt", None)
        try:
            alt_val = float(alt_raw)
        except (TypeError, ValueError):
            alt_val = "Unknown"
        name_val = row.get("name", "")
        # Key by id (as string) and by name (uppercase) for robustness
        if "id" in row:
            station_map[str(row["id"])] = {"alt": alt_val, "name": name_val}
        station_map[str(name_val)] = {"alt": alt_val, "name": name_val}
    return station_map


def extract_station_name(file_path: str) -> str:
    """Derive station name from filename; strip point_ prefix and _SH suffix."""
    base_name = os.path.basename(file_path)
    station_name, _ = os.path.splitext(base_name)
    station_name = re.sub(r"^point_", "", station_name)
    station_name = re.sub(r"_sh$", "", station_name, flags=re.IGNORECASE)
    return station_name


def generate_output_filename(output_dir, station_name, small_plot=False):
    folder_name = os.path.basename(output_dir.rstrip(os.sep))
    suffix = "_small" if small_plot else ""
    return os.path.join(output_dir, f"{folder_name}_{station_name}{suffix}.png")


def convert_temp_to_celsius_if_needed(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    if variable == "temp" and variable in df.columns and df[variable].max() > 100:
        df[variable] = df[variable] - 273.15
    return df


def plot_timeseries_with_title(dataframes, station_name, stations_map, variable, output_dir, legend_entries):
    station_info = stations_map.get(station_name, {"alt": "Unknown"})
    alt_display = station_info.get("alt", "Unknown")
    if isinstance(alt_display, (int, float)):
        alt_display = int(round(alt_display))
    display_name = station_info.get("name", station_name)
    y_labels = {
        "swe": "Snow Water Equivalent [m³]",
        "temp": "Air Temperature [°C]",
        "precip": "Precipitation Sum [mm]",
        "depth": "Snow Depth [m]",
        "snow_depth": "Snow Depth [m]",
    }
    plot_title = y_labels.get(variable, "Value")
    combined_title = f"{plot_title} - {display_name} ({alt_display} m)"

    output_path = generate_output_filename(output_dir, station_name)
    logging.info(f"Generating plot: {output_path}")

    plt.figure(figsize=(11.7, 6))
    plt.title(combined_title, fontsize=14, pad=10)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, df in enumerate(dataframes):
        if variable in df.columns:
            series = df[variable]
            logging.info(
                f"{legend_entries[idx]} stats for {station_name}: count={series.count()} "
                f"min={series.min()} max={series.max()}"
            )
            plt.plot(df.index, series, label=legend_entries[idx], color=colors[idx % len(colors)], linewidth=1.5)
        else:
            logging.warning(f"Skipping {legend_entries[idx]} for station {station_name}: '{variable}' not found")
    plt.ylabel(y_labels.get(variable, "Value"))
    plt.legend()
    plt.grid(True)
    plt.tight_layout(pad=0.1)
    plt.savefig(output_path, format="png", bbox_inches="tight")
    plt.close()


def process_directories(input_dirs, stations_map, output_dir, legend_entries, aggregations, variable):
    station_files = {}
    for dir_idx, input_dir in enumerate(input_dirs):
        if not os.path.exists(input_dir):
            logging.warning(f"Input directory not found: {input_dir}")
            continue
        input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")]
        for file_path in input_files:
            station_name = extract_station_name(file_path)
            station_files.setdefault(station_name, [None] * len(input_dirs))
            station_files[station_name][dir_idx] = file_path

    for station_name, files in station_files.items():
        if not all(files):
            missing_dirs = [input_dirs[idx] for idx, file_path in enumerate(files) if not file_path]
            logging.warning(f"Skipping station {station_name}: Missing files in {missing_dirs}")
            continue

        logging.info(f"Processing station: {station_name}")
        dataframes = []
        for idx, file_path in enumerate(files):
            df = load_csv(file_path, variable)
            df = convert_temp_to_celsius_if_needed(df, variable)
            logging.info(f"{legend_entries[idx]} raw shape for {station_name}: {df.shape}")
            df = filter_data_by_timespan(df, START_DATE, END_DATE)
            logging.info(f"{legend_entries[idx]} filtered shape for {station_name}: {df.shape}")
            df = aggregate_data(df, aggregations[idx])
            logging.info(f"{legend_entries[idx]} aggregated shape for {station_name}: {df.shape}")
            dataframes.append(df)

        if all(not df.empty and variable in df.columns for df in dataframes):
            plot_timeseries_with_title(dataframes, station_name, stations_map, variable, output_dir, legend_entries)
        else:
            logging.warning(f"Skipping station {station_name}: '{variable}' column missing in one or more dataframes")


def main():
    stations_map = load_stations(STATIONS_CSV_PATH)
    os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)
    process_directories(INPUT_CSV_DIRS, stations_map, OUTPUT_PLOT_DIR, LEGEND_ENTRIES, AGGREGATIONS, VARIABLE_TO_PLOT)


if __name__ == "__main__":
    main()
