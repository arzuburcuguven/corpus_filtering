from corpus_filtering.filters.base import CorpusFilter
from corpus_filtering.filters.base import ExistentialThereQuantifierFilter, BindingReflexive, InterrogativeWhModifierFilter, LicensedNPI
from conllu import parse_incr
from pathlib import Path
import json
import random


class FilterPipeline:
    def __init__(self, filter: CorpusFilter, output_dir: str = "output/",
                 train_ratio: float = 0.90, dev_ratio: float = 0.05,
                 seed: int = 42, shuffle: bool = True):
        self.filter = filter
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = 1.0 - train_ratio - dev_ratio
        self.seed = seed
        self.shuffle = shuffle

    def _read_conllu(self, path):
        with open(path, "r", encoding="utf-8") as f:
            for sent in parse_incr(f):
                yield sent

    def _write_stats(self, stats: dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "filter": self.filter.name,
            "splits": {
                split_name: {
                    "total": s["total"],
                    "matched": s["matched"],
                    "match_rate": round(s["matched"] / s["total"], 4) if s["total"] else 0,
                }
                for split_name, s in stats.items()
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def run(self, input_path: str):
        path = Path(input_path)
        if path.is_dir():
            print(f"Input is a directory — treating as UD corpus")
            self._run_from_ud(path)
        elif path.is_file():
            print(f"Input is a single file — splitting into train/dev/test")
            self._run_from_single_file(path)
        else:
            raise FileNotFoundError(f"{input_path} is neither a file nor a directory")

    def _run_from_ud(self, ud_path: Path):
        train_file = next(ud_path.glob("*-ud-train.conllu"))
        dev_file = next(ud_path.glob("*-ud-dev.conllu"))
        test_file = next(ud_path.glob("*-ud-test.conllu"))

        # Build an iterator over (split_name, sentence) for the unified writer
        def stream():
            for sent in self._read_conllu(train_file):
                yield "train", sent
            for sent in self._read_conllu(dev_file):
                yield "dev", sent
            for sent in self._read_conllu(test_file):
                yield "test", sent

        self._process_stream(stream())

    def _run_from_single_file(self, input_path: Path):
        # Pass 1: count sentences
        print("Counting sentences...")
        n_sents = sum(1 for _ in self._read_conllu(input_path))
        print(f"Total sentences: {n_sents:,}")

        indices = list(range(n_sents))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)

        n_train = int(n_sents * self.train_ratio)
        n_dev = int(n_sents * self.dev_ratio)

        train_idx = set(indices[:n_train])
        dev_idx = set(indices[n_train:n_train + n_dev])

        print(f"Splitting: {n_train:,} train / {n_dev:,} dev / {n_sents - n_train - n_dev:,} test")

        def stream():
            for i, sent in enumerate(self._read_conllu(input_path)):
                if i in train_idx:
                    yield "train", sent
                elif i in dev_idx:
                    yield "dev", sent
                else:
                    yield "test", sent

        self._process_stream(stream())

    def _process_stream(self, sentence_stream):
        """Single unified output writer — same outputs every time."""
        out_dir = self.output_dir / self.filter.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Open all output files
        files = {
            # Labelled CoNLL-U for all splits
            "train_conllu": open(out_dir / "train.conllu", "w", encoding="utf-8"),
            "dev_conllu":   open(out_dir / "dev.conllu",   "w", encoding="utf-8"),
            "test_conllu":  open(out_dir / "test.conllu",  "w", encoding="utf-8"),
            # Untouched (full) text for all splits
            "train_full":   open(out_dir / "train_full.txt", "w", encoding="utf-8"),
            "dev_full":     open(out_dir / "dev_full.txt",   "w", encoding="utf-8"),
            "test_full":    open(out_dir / "test_full.txt",  "w", encoding="utf-8"),
            # Train-only: clean (no matches) and matched (just matches)
            "train_clean":   open(out_dir / "train_clean.txt",   "w", encoding="utf-8"),
            "train_matched": open(out_dir / "train_matched.txt", "w", encoding="utf-8"),
        }

        stats = {s: {"matched": 0, "total": 0} for s in ("train", "dev", "test")}

        try:
            for i, (split, sent) in enumerate(sentence_stream):
                is_match = self.filter._exclude_sent(sent)
                stats[split]["total"] += 1
                if is_match:
                    stats[split]["matched"] += 1
                    sent.metadata["phenomenon"] = self.filter.name

                serialized = sent.serialize()
                text = sent.metadata.get("text", "") + "\n"

                # Always: write labelled .conllu and full .txt for all splits
                files[f"{split}_conllu"].write(serialized)
                files[f"{split}_full"].write(text)

                # Train-only: split into clean / matched
                if split == "train":
                    if is_match:
                        files["train_matched"].write(text)
                    else:
                        files["train_clean"].write(text)

                if (i + 1) % 100_000 == 0:
                    print(f"  Processed {i + 1:,} sentences", flush=True)
        finally:
            for f in files.values():
                f.close()

        for split, s in stats.items():
            rate = s["matched"] / s["total"] if s["total"] else 0
            print(f"{split.capitalize():5}: {s['matched']:,} matched / {s['total']:,} total ({rate:.2%})")

        self._write_stats(stats, out_dir / "stats.json")


def run_filters(filters, input_path, output_dir="output/", **kwargs):
    """Run multiple filters sequentially on the same input."""
    for i, f in enumerate(filters, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(filters)}] Running filter: {f.name}")
        print(f"{'='*60}\n")
        try:
            p = FilterPipeline(f, output_dir=output_dir, **kwargs)
            p.run(input_path)
            print(f"✓ Completed {f.name}")
        except Exception as e:
            print(f"✗ FAILED {f.name}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    INPUT_PATH = "/Users/argy/PHD/WS/corpus_filtering/UD_English-GUM"
    OUTPUT_DIR = "output/"

    filters = [
        LicensedNPI(),
    ]

    run_filters(filters, INPUT_PATH, output_dir=OUTPUT_DIR)