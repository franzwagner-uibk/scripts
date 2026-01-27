# Description: Monitor CPU, RAM usage, and folder sizes of multiple directories over time, logging all data in a single CSV file.
# Author: Franz Wagner
# Date: 2025-01-11

import psutil
import os
import time
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Input and output file paths
output_file = r"C:\Users\f.wagner.VERTIGIS\OneDrive - VertiGIS\PROJEKTE\Masterarbeit\10-Temp\sytem_performance_log.csv"
output_plot = r"C:\Users\f.wagner.VERTIGIS\OneDrive - VertiGIS\PROJEKTE\Masterarbeit\10-Temp\performance_plot.png"
directories_to_track = [
    {"path": r"C:\Users\f.wagner.VERTIGIS\OneDrive - VertiGIS\PROJEKTE\Masterarbeit", "unit": "GB"}
]

# Logging interval in seconds
logging_interval = 10

# Plot update interval in seconds
plot_update_interval = 60

# Task list
def main():
    tasks = [
        monitor_performance,
    ]
    for task in tasks:
        task()

def monitor_performance():
    """Monitor system performance and folder sizes, logging the data."""
    # Setup logging
    logging.basicConfig(
        filename=output_file,
        level=logging.INFO,
        format="%(asctime)s, %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Create CSV header
    directory_headers = [f"Folder_Size({d['unit']})_{os.path.basename(d['path'])}" for d in directories_to_track]
    header = [
        "Timestamp",
        "CPU_Usage(%)",
        "RAM_Usage(%)",
        *["Folder_Name_" + os.path.basename(d["path"]) for d in directories_to_track],
        *directory_headers
    ]
    with open(output_file, "w") as f:
        f.write(",".join(header) + "\n")

    print("Monitoring performance. Press Ctrl+C to stop.")
    last_plot_update = time.time()
    try:
        while True:
            # Get CPU and RAM usage
            cpu_usage = psutil.cpu_percent(interval=1)
            ram_usage = psutil.virtual_memory().percent

            # Get folder sizes and names for all directories
            folder_data = [
                (os.path.basename(directory["path"]), get_folder_size(directory["path"], directory["unit"]))
                for directory in directories_to_track
            ]

            # Log data to the CSV file
            log_message = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                str(cpu_usage),
                str(ram_usage),
                *[name for name, size in folder_data],
                *[str(size) for name, size in folder_data]
            ]
            with open(output_file, "a") as f:
                f.write(",".join(log_message) + "\n")

            # Print live updates to the console
            folder_sizes_str = ", ".join(
                [f"{name}: {size} {directories_to_track[i]['unit']}" for i, (name, size) in enumerate(folder_data)]
            )
            print(f"CPU: {cpu_usage}%, RAM: {ram_usage}%, Folder Sizes: {folder_sizes_str}")

            # Update plot every plot_update_interval seconds
            if time.time() - last_plot_update >= plot_update_interval:
                create_plot(output_file, output_plot)
                last_plot_update = time.time()

            time.sleep(logging_interval)  # Collect data every logging_interval seconds
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

def get_folder_size(directory, unit):
    """Calculate the total size of a directory."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    # Convert size to the desired unit
    if unit == "MB":
        return round(total_size / (1024 ** 2), 2)
    elif unit == "GB":
        return round(total_size / (1024 ** 3), 2)
    else:
        raise ValueError("Invalid unit. Choose 'MB' or 'GB'.")

def create_plot(csv_file, plot_file):
    """Create a plot from the CSV file and save it as an image."""
    try:
        data = pd.read_csv(csv_file)
        print(f"CSV columns: {data.columns}")  # Debugging step to check columns
        data["Timestamp"] = pd.to_datetime(data["Timestamp"])

        # Integrated plot with dual y-axes
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot CPU and RAM usage on the left y-axis
        ax1.plot(data["Timestamp"], data["CPU_Usage(%)"], label="CPU Usage (%)", color='tab:blue')
        ax1.plot(data["Timestamp"], data["RAM_Usage(%)"], label="RAM Usage (%)", color='tab:orange')
        ax1.set_xlabel("Timestamp")
        ax1.set_ylabel("Usage (%)", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc="upper left")
        ax1.grid()

        # Create a twin y-axis to plot folder sizes
        ax2 = ax1.twinx()
        colors = ['tab:green', 'tab:red', 'tab:purple', 'tab:brown']  # Unique colors for each folder size
        for i, directory in enumerate(directories_to_track):
            column_name = f"Folder_Size({directory['unit']})_{os.path.basename(directory['path'])}"
            if column_name in data.columns:
                ax2.plot(data["Timestamp"], data[column_name], label=f"{os.path.basename(directory['path'])} Size ({directory['unit']})", color=colors[i % len(colors)])
        ax2.set_ylabel("Folder Size (GB)", color='tab:green')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.legend(loc="upper right")

        # Title and layout
        plt.title("CPU, RAM Usage, and Folder Sizes Over Time")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()

    except Exception as e:
        print(f"Error creating plot: {e}")

# Run the task list
if __name__ == "__main__":
    main()