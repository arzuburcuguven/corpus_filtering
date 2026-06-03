"""Filter a corpus based on BEAR facts the model has genuinely learned.

A fact qualifies if it passes the two-step gate in the evaluation results:
  1. rank_correct == 1  (raw PLL — necessary condition)
  2. pmi_correct == True (PMI-normalised — fact-specific signal)
  3. pmi_entropy_norm < threshold (confident after prior removal)

For each qualifying (subject, object) pair, Wikidata aliases are fetched once
and cached.  The filter then marks any sentence where both the subject and
object (or one of their aliases) co-occur — these go into train_matched.

Usage
-----
# Step 1: pre-build the alias cache (run once, NOT as an array job)
python run_bear_facts_filter.py \
    --results /path/to/results.json \
    --alias-cache /path/to/cache.json \
    --prefetch-only

# Step 2: run the corpus filter (slurm array)
python run_bear_facts_filter.py \
    --input  chunk_00.pkl \
    --output /path/to/output/ \
    --results /path/to/results.json \
    --alias-cache /path/to/cache.json \
    --pmi-entropy-threshold 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.corpus_filtering.filters.facts import BearFactsFilter, get_predefined_aliases
from src.corpus_filtering.pipeline import run_filters

WIKIDATA_API = "https://www.wikidata.org/w/api.php"


# ── Wikidata helpers ──────────────────────────────────────────────────────────


def _wikidata_search(label: str) -> str | None:
    """Return the top Wikidata Q-ID for an English label, or None."""
    params = urllib.parse.urlencode({
        "action": "wbsearchentities",
        "search": label,
        "language": "en",
        "limit": 1,
        "format": "json",
    })
    with urllib.request.urlopen(f"{WIKIDATA_API}?{params}", timeout=15) as resp:
        data = json.loads(resp.read())
    results = data.get("search", [])
    return results[0]["id"] if results else None


def _wikidata_aliases(qid: str) -> list[str]:
    """Return English label + all English aliases for a Q-ID."""
    params = urllib.parse.urlencode({
        "action": "wbgetentities",
        "ids": qid,
        "props": "labels|aliases",
        "languages": "en",
        "format": "json",
    })
    with urllib.request.urlopen(f"{WIKIDATA_API}?{params}", timeout=15) as resp:
        data = json.loads(resp.read())
    entity = data.get("entities", {}).get(qid, {})
    terms: list[str] = []
    label = entity.get("labels", {}).get("en", {}).get("value")
    if label:
        terms.append(label)
    for alias in entity.get("aliases", {}).get("en", []):
        terms.append(alias["value"])
    return terms


def _save_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def fetch_aliases(
    labels: list[str],
    cache_path: Path,
    sleep: float = 0.15,
) -> dict[str, list[str]]:
    """Return {label: [surface_form, ...]} for every label, using a JSON cache.

    Labels already in the cache are not re-fetched.  The cache is saved after
    every 50 new lookups and once at the end.
    """
    cache: dict[str, list[str]] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    missing = [l for l in labels if l not in cache]
    if missing:
        print(f"  Fetching Wikidata aliases for {len(missing)} entities "
              f"({len(labels) - len(missing)} already cached)...")
        for i, label in enumerate(missing, 1):
            try:
                qid = _wikidata_search(label)
                cache[label] = _wikidata_aliases(qid) if qid else [label]
            except Exception as exc:
                print(f"  Warning: failed for '{label}': {exc}")
                cache[label] = [label]
            time.sleep(sleep)
            if i % 50 == 0:
                print(f"  {i}/{len(missing)} fetched — saving cache")
                _save_cache(cache, cache_path)
        _save_cache(cache, cache_path)
        print(f"  Done. Cache saved to {cache_path}")

    return {l: cache.get(l, [l]) for l in labels}


# ── Results parsing ───────────────────────────────────────────────────────────


def load_qualifying_pairs(
    results_path: Path,
    pmi_entropy_threshold: float,
) -> list[tuple[str, str]]:
    """Return (subject, correct_object) pairs that passed the two-step gate."""
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle both a full check_pretrained results file and a bare bear dict
    bear = data.get("results", {}).get("bear", data)

    pairs: list[tuple[str, str]] = []
    skipped_no_pmi = 0
    for rel_data in bear.get("by_relation", {}).values():
        for fact in rel_data.get("correct_facts", []):
            if not fact.get("pmi_correct", False):
                skipped_no_pmi += 1
                continue
            if fact.get("pmi_entropy_norm", 1.0) >= pmi_entropy_threshold:
                continue
            pairs.append((fact["subject"], fact["correct_object"]))

    if skipped_no_pmi:
        print(f"  Skipped {skipped_no_pmi} facts without pmi_correct "
              f"(relations without null templates)")
    return pairs


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter a corpus based on BEAR facts the model genuinely learned.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", help="Corpus pickle or CoNLL-U file/dir")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument(
        "--results", required=True,
        help="Path to check_pretrained results JSON (must contain bear.by_relation)",
    )
    parser.add_argument(
        "--alias-cache", default=None, metavar="PATH",
        help="Path to Wikidata alias cache JSON. "
             "Defaults to wikidata_alias_cache.json next to the results file.",
    )
    parser.add_argument(
        "--pmi-entropy-threshold", type=float, default=0.5, metavar="T",
        help="Maximum pmi_entropy_norm for a fact to qualify (lower = more confident).",
    )
    parser.add_argument(
        "--prefetch-only", action="store_true",
        help="Only build the alias cache and exit — do not run the corpus filter. "
             "Run this once before submitting the slurm array.",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    cache_path = (
        Path(args.alias_cache)
        if args.alias_cache
        else results_path.parent / "wikidata_alias_cache.json"
    )

    # ── 1. Extract qualifying facts ───────────────────────────────────────────
    print(f"Loading results from {results_path}")
    pairs = load_qualifying_pairs(results_path, args.pmi_entropy_threshold)
    print(f"Qualifying facts (pmi_correct=True, pmi_entropy_norm<{args.pmi_entropy_threshold}): "
          f"{len(pairs)}")
    if not pairs:
        print("No qualifying facts — nothing to filter.")
        return

    # ── 2. Fetch Wikidata aliases ─────────────────────────────────────────────
    all_labels = list({label for pair in pairs for label in pair})
    print(f"Unique entity labels: {len(all_labels)}")
    alias_map = fetch_aliases(all_labels, cache_path)

    if args.prefetch_only:
        print("--prefetch-only: alias cache built. Exiting.")
        return

    if not args.input or not args.output:
        parser.error("--input and --output are required when not using --prefetch-only")

    # ── 3. Build filter and run pipeline ─────────────────────────────────────
    def _merge(label: str) -> list[str]:
        wikidata = alias_map.get(label, [label])
        predefined = get_predefined_aliases(label)
        return list(dict.fromkeys(wikidata + predefined))  # deduplicate, wikidata first

    pairs_with_aliases = [
        (_merge(subj), _merge(obj))
        for subj, obj in pairs
    ]
    bear_filter = BearFactsFilter(pairs_with_aliases)
    print(f"\nRunning BearFacts filter over {args.input}")
    run_filters([bear_filter], args.input, output_dir=args.output)


if __name__ == "__main__":
    main()
