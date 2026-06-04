"""Filter a corpus based on BEAR facts the model has genuinely learned.

A fact qualifies when ALL of the following hold:

  Model conditions (from check_pretrained results JSON):
    1. rank_correct == 1  — model's top prediction is the correct answer
    2. p_correct >= p-correct-threshold  — model assigns enough probability mass
    3. entropy_norm < entropy-threshold  — prediction is confident (not uniform)

  Corpus conditions (from bear_corpus_stats JSON, optional):
    4. subject occurrence count   >= min-subject-count
    5. object occurrence count    >= min-object-count
    6. co-occurrence line count   >= min-cooccur-count

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
    --corpus-stats /path/to/bear_corpus_stats.json \
    --corpus-dataset 10b
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


def load_bear_subject_aliases() -> dict[str, list[str]]:
    """Return {sub_label: [alias, ...]} from the lm_pub_quiz BEAR dataset."""
    from lm_pub_quiz import Dataset
    result: dict[str, list[str]] = {}
    for relation in Dataset.from_name("BEAR"):
        for _, row in relation.instance_table.iterrows():
            label = str(row["sub_label"])
            aliases = row.get("sub_aliases", [])
            if not isinstance(aliases, list):
                aliases = []
            existing = result.setdefault(label, [])
            for a in aliases:
                a = str(a)
                if a not in existing:
                    existing.append(a)
    return result


def load_corpus_stats(
    stats_path: Path | None,
    dataset: str,
) -> dict[tuple[str, str], dict]:
    """Return {(subject, correct_object): {subject, object, cooccur}} for one dataset."""
    if stats_path is None or not stats_path.exists():
        return {}
    with open(stats_path, encoding="utf-8") as f:
        data = json.load(f)
    lookup: dict[tuple[str, str], dict] = {}
    for rel_data in data.get("relations", {}).values():
        for inst in rel_data.get("instances", []):
            key = (inst["subject"], inst["correct_object"])
            lookup[key] = inst.get("by_dataset", {}).get(dataset, {})
    return lookup


def load_qualifying_pairs(
    results_path: Path,
    p_correct_threshold: float,
    entropy_threshold: float,
    corpus_lookup: dict[tuple[str, str], dict],
    min_subject_count: int,
    min_object_count: int,
    min_cooccur_count: int,
) -> list[tuple[str, str]]:
    """Return (subject, correct_object) pairs that pass all quality gates."""
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    # Handle both a full check_pretrained results file and a bare bear dict
    bear = data.get("results", {}).get("bear", data)

    pairs: list[tuple[str, str]] = []
    n_total = n_low_p = n_high_entropy = n_corpus = 0
    for rel_data in bear.get("by_relation", {}).values():
        # correct_facts already guarantees rank_correct == 1
        for fact in rel_data.get("correct_facts", []):
            n_total += 1
            if fact.get("p_correct", 0.0) < p_correct_threshold:
                n_low_p += 1
                continue
            if fact.get("entropy_norm", 1.0) >= entropy_threshold:
                n_high_entropy += 1
                continue
            if corpus_lookup:
                key = (fact["subject"], fact["correct_object"])
                stats = corpus_lookup.get(key, {})
                if (stats.get("subject", 0) < min_subject_count
                        or stats.get("object", 0) < min_object_count
                        or stats.get("cooccur", 0) < min_cooccur_count):
                    n_corpus += 1
                    continue
            pairs.append((fact["subject"], fact["correct_object"]))

    print(f"  Qualifying facts: {len(pairs)} / {n_total}")
    print(f"  Dropped — low p_correct: {n_low_p}, high entropy: {n_high_entropy}, "
          f"corpus threshold: {n_corpus}")
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
        "--p-correct-threshold", type=float, default=0.1, metavar="T",
        help="Minimum p_correct for a fact to qualify (model's probability on the correct answer).",
    )
    parser.add_argument(
        "--entropy-threshold", type=float, default=0.8, metavar="T",
        help="Maximum entropy_norm for a fact to qualify (lower = more confident).",
    )
    parser.add_argument(
        "--corpus-stats", default=None, metavar="PATH",
        help="Path to bear_corpus_stats.json produced by bear_corpus_cooccurrence.py. "
             "When provided, corpus occurrence thresholds are applied.",
    )
    parser.add_argument(
        "--corpus-dataset", default="10b", metavar="DATASET",
        help="Which dataset column to use from --corpus-stats (default: 10b).",
    )
    parser.add_argument(
        "--min-subject-count", type=int, default=5, metavar="N",
        help="Minimum corpus lines containing the subject.",
    )
    parser.add_argument(
        "--min-object-count", type=int, default=5, metavar="N",
        help="Minimum corpus lines containing the correct object.",
    )
    parser.add_argument(
        "--min-cooccur-count", type=int, default=1, metavar="N",
        help="Minimum corpus lines where subject and object co-occur.",
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
    corpus_stats_path = Path(args.corpus_stats) if args.corpus_stats else None
    corpus_lookup = load_corpus_stats(corpus_stats_path, args.corpus_dataset)
    if corpus_stats_path:
        print(f"Corpus stats: {corpus_stats_path} (dataset={args.corpus_dataset}, "
              f"{len(corpus_lookup)} entries)")
    pairs = load_qualifying_pairs(
        results_path,
        p_correct_threshold=args.p_correct_threshold,
        entropy_threshold=args.entropy_threshold,
        corpus_lookup=corpus_lookup,
        min_subject_count=args.min_subject_count,
        min_object_count=args.min_object_count,
        min_cooccur_count=args.min_cooccur_count,
    )
    if not pairs:
        print("No qualifying facts after filtering — nothing to do.")
        return

    # ── 2. Fetch aliases ──────────────────────────────────────────────────────
    all_labels = list({label for pair in pairs for label in pair})
    print(f"Unique entity labels: {len(all_labels)}")
    print("Loading BEAR subject aliases from lm_pub_quiz ...")
    bear_aliases = load_bear_subject_aliases()
    print(f"  {len(bear_aliases)} subjects with aliases")
    alias_map = fetch_aliases(all_labels, cache_path)

    if args.prefetch_only:
        print("--prefetch-only: alias cache built. Exiting.")
        return

    if not args.input or not args.output:
        parser.error("--input and --output are required when not using --prefetch-only")

    # ── 3. Build filter and run pipeline ─────────────────────────────────────
    def _merge(label: str) -> list[str]:
        wikidata = alias_map.get(label, [label])
        bear = bear_aliases.get(label, [])
        predefined = get_predefined_aliases(label)
        return list(dict.fromkeys(wikidata + bear + predefined))  # deduplicate, wikidata first

    pairs_with_aliases = [
        (_merge(subj), _merge(obj))
        for subj, obj in pairs
    ]
    bear_filter = BearFactsFilter(pairs_with_aliases)
    print(f"\nRunning BearFacts filter over {args.input}")
    run_filters([bear_filter], args.input, output_dir=args.output)


if __name__ == "__main__":
    main()
