#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verzeichnisbaum (Ordner + Dateien) als Textdatei exportieren.

Beschreibung
------------
Erstellt eine Baumstruktur (ähnlich `tree`) mit Ordnern und Dateinamen.
Schreibt die Struktur in eine Textdatei mit einfacher, lesbarer Einrückung.

Autor: Franz Wagner
Datum: 2025-11-03
"""

from __future__ import annotations
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# ----------------------------- #
#           KONFIGURATION       #
# ----------------------------- #

# Eingabe- und Ausgabepfade
INPUT_DIR: Path = Path(r"C:\Daten\PhD\openamundsen_da\examples\test-project\propagation\season_2017-2018\step_00_init\ensembles\prior\member_025\results")     # <-- Anpassen!
OUTPUT_FILE: Path = Path(r"98-Temp/verzeichnisstruktur.txt")  # Ausgabe-Datei

# Steuerung
MAX_DEPTH: Optional[int] = None   # z. B. 3 für maximale Ebenentiefe, None = unbegrenzt
SORT_DIRS_FIRST: bool = True      # Ordner vor Dateien sortieren
CASE_INSENSITIVE_SORT: bool = True

# Logging
LOG_LEVEL: str = "INFO"
LOG_FILE: Path = Path("./logs/verzeichnisstruktur.log")

# ----------------------------- #
#          LOGGING SETUP        #
# ----------------------------- #

def configure_logging() -> None:
    """Konfiguriert Logging für Konsole und Datei."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, encoding="utf-8")
        ]
    )

# ----------------------------- #
#           FUNKTIONEN          #
# ----------------------------- #

def validate_input(input_dir: Path) -> None:
    """Prüft, ob der Eingabeordner existiert."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Eingabeordner existiert nicht: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Eingabepfad ist kein Ordner: {input_dir}")

def list_entries(directory: Path) -> List[Path]:
    """Listet alle Einträge (Ordner + Dateien) eines Verzeichnisses auf."""
    try:
        entries = list(directory.iterdir())
    except PermissionError:
        logging.warning("Keine Berechtigung für: %s", directory)
        return []
    except FileNotFoundError:
        logging.warning("Nicht gefunden: %s", directory)
        return []

    # Sortierung
    def sort_key(p: Path) -> tuple:
        name = p.name.lower() if CASE_INSENSITIVE_SORT else p.name
        return ((0 if p.is_dir() else 1) if SORT_DIRS_FIRST else 0, name)

    entries.sort(key=sort_key)
    return entries

def build_tree(root: Path, max_depth: Optional[int]) -> List[str]:
    """Erstellt die Baumstruktur (Ordner + Dateien) als Liste von Textzeilen."""
    lines: List[str] = [root.name]
    BRANCH, LAST, VBAR, SPACE = "├── ", "└── ", "│   ", "    "

    def recurse(current: Path, prefix: str, depth: int) -> None:
        if max_depth is not None and depth > max_depth:
            return
        entries = list_entries(current)
        for idx, entry in enumerate(entries):
            connector = LAST if idx == len(entries) - 1 else BRANCH
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                new_prefix = prefix + (SPACE if idx == len(entries) - 1 else VBAR)
                recurse(entry, new_prefix, depth + 1)

    recurse(root, "", 1)
    return lines

def write_output(lines: List[str], output_file: Path) -> None:
    """Schreibt die Verzeichnisstruktur in eine Textdatei."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Verzeichnisstruktur",
        f"# Quelle: {INPUT_DIR.resolve()}",
        f"# Erstellt: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    with output_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(header + lines))
    logging.info("Struktur gespeichert in: %s", output_file.resolve())

# ----------------------------- #
#             MAIN              #
# ----------------------------- #

def main() -> None:
    """Hauptfunktion."""
    configure_logging()
    logging.info("Starte Erstellung der Verzeichnisstruktur …")

    try:
        validate_input(INPUT_DIR)
        lines = build_tree(INPUT_DIR, MAX_DEPTH)
        write_output(lines, OUTPUT_FILE)
        logging.info("Verzeichnisstruktur erfolgreich erstellt.")
    except Exception as e:
        logging.exception("Fehler: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
