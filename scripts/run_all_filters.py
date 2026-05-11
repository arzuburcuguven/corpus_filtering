from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus_filtering.filters.base import (
    ExistentialThereQuantifierFilter,
    BindingReflexive,
    InterrogativeWhModifierFilter,
    LicensedNPI
)
from src.corpus_filtering.pipeline import run_filters

INPUT_PATH = "/Users/argy/PHD/WS/corpus_filtering/UD_English-GUM"
OUTPUT_DIR = "output/"

if __name__ == "__main__":
    filters = [
        ExistentialThereQuantifierFilter(),
        BindingReflexive(),
        InterrogativeWhModifierFilter(),
        LicensedNPI()
    ]
    run_filters(filters, INPUT_PATH, output_dir=OUTPUT_DIR)