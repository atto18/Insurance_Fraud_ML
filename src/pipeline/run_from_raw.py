"""
End-to-end data pipeline from raw CSVs to scored providers (original ensemble).

Expected inputs (under project root):
  data/raw/provider.csv   — CMS Medicare Physician/Other Supplier PUF
  data/raw/exclusion.csv  — OIG LEIE (CSV export)

Steps:
  1. preprocess provider → data/preprocessed/provider_cleaned.csv
  2. preprocess exclusion → data/preprocessed/exclusion_cleaned.csv
  3. build labels → data/final/provider_with_labels.csv
  4. build features → data/final/provider_features.csv
  5. score with saved models → data/final/scored_providers.csv
  6. regenerate dashboard outputs → outputs/plots/, outputs/metrics.json

Run from project root:
  python src/pipeline/run_from_raw.py
  python src/pipeline/run_from_raw.py --nrows 50000
  python src/pipeline/run_from_raw.py --skip-score
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.build_features import build_features
from src.labeling.build_labels import build_labeled_dataset
from src.modeling.score_saved_ensemble import score_all
from src.pipeline.regenerate_plots import main as regenerate_outputs
from src.preprocessing.preprocess_exclsuion_data import clean_exclusion_data
from src.preprocessing.preprocess_provider_data import clean_provider_data


def run_pipeline(
    root: Path | None = None,
    *,
    nrows: int | None = None,
    skip_score: bool = False,
) -> None:
    root = root or PROJECT_ROOT
    raw_provider = root / "data" / "raw" / "provider.csv"
    raw_excl = root / "data" / "raw" / "exclusion.csv"
    pre_prov = root / "data" / "preprocessed" / "provider_cleaned.csv"
    pre_excl = root / "data" / "preprocessed" / "exclusion_cleaned.csv"
    final_labeled = root / "data" / "final" / "provider_with_labels.csv"
    final_features = root / "data" / "final" / "provider_features.csv"
    scored_out = root / "data" / "final" / "scored_providers.csv"
    model_dir = root / "models"

    if not raw_provider.is_file():
        raise FileNotFoundError(
            f"Missing CMS provider file. Save it as:\n  {raw_provider}"
        )
    if not raw_excl.is_file():
        raise FileNotFoundError(
            f"Missing OIG exclusion file. Save it as:\n  {raw_excl}"
        )

    print("== 1/6 Preprocess provider billing data ==")
    clean_provider_data(str(raw_provider), str(pre_prov), nrows=nrows)

    print("\n== 2/6 Preprocess OIG exclusion list ==")
    clean_exclusion_data(str(raw_excl), str(pre_excl))

    print("\n== 3/6 Build labels ==")
    build_labeled_dataset(pre_prov, pre_excl, final_labeled)

    print("\n== 4/6 Engineer features ==")
    build_features(input_path=final_labeled, output_path=final_features)

    if skip_score:
        print("\nSkipping scoring (--skip-score).")
        return

    print("\n== 5/6 Score providers (saved ensemble) ==")
    score_all(
        features_csv=final_features,
        out_csv=scored_out,
        model_dir=model_dir,
    )

    print("\n== 6/6 Regenerate dashboard outputs ==")
    regenerate_outputs()

    print("\nPipeline finished successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Medicare fraud data + scoring pipeline.")
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Load only first N rows of raw provider CSV (debug / smoke test).",
    )
    parser.add_argument(
        "--skip-score",
        action="store_true",
        help="Stop after feature engineering (no model inference).",
    )
    args = parser.parse_args()
    try:
        run_pipeline(nrows=args.nrows, skip_score=args.skip_score)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
