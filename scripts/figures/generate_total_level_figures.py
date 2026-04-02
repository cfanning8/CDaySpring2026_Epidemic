from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FIG_DIR))

from generate_project_assets import generate_total_level_figure_assets


def main() -> None:
    generate_total_level_figure_assets(ROOT)


if __name__ == "__main__":
    main()
