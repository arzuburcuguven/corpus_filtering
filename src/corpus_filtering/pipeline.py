from corpus_filtering.filters.base import CorpusFilter
from corpus_filtering.filters.base import ExistentialThereQuantifierFilter, BindingReflexive, InterrogativeWhModifierFilter, LicensedNPI
from conllu import parse_incr
from pathlib import Path
import json
import random
import pickle
import argparse


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
        if path.suffix in (".pkl", ".pickle"):
            print(f"Input is a pickle")
            self._run_from_pickle(path)  # BUG FIX: was self.run_from_ud
        elif path.is_dir():
            print(f"Input is a directory — treating as UD corpus")
            self._run_from_ud(path)
        elif path.is_file():
            print(f"Input is a single file — splitting into train/dev/test")
            self._run_from_single_file(path)
        else:
            raise FileNotFoundError(f"{input_path} is neither a file nor a directory")

    def _run_from_pickle(self, input_path: Path):
        print("pickle..")
        with open(input_path, "rb") as f:
            sentences = pickle.load(f)
        print(f"loaded {len(sentences):,} sents")
        rng = random.Random(self.seed)

        def stream():
            for sent in sentences:  # BUG FIX: was self.sentences
                r = rng.random()
                if r < self.train_ratio:
                    yield "train", sent
                elif r < self.train_ratio + self.dev_ratio:
                    yield "dev", sent
                else:
                    yield "test", sent

        self._process_stream(stream())

    def _run_from_ud(self, ud_path: Path):
        train_file = next(ud_path.glob("*-ud-train.conllu"))
        dev_file = next(ud_path.glob("*-ud-dev.conllu"))
        test_file = next(ud_path.glob("*-ud-test.conllu"))

        def stream():
            for sent in self._read_conllu(train_file):
                yield "train", sent
            for sent in self._read_conllu(dev_file):
                yield "dev", sent
            for sent in self._read_conllu(test_file):
                yield "test", sent

        self._process_stream(stream())

    def _run_from_single_file(self, input_path: Path):
        rng = random.Random(self.seed)

        def stream():
            for sent in self._read_conllu(input_path):
                r = rng.random()
                if r < self.train_ratio:
                    yield "train", sent
                elif r < self.train_ratio + self.dev_ratio:
                    yield "dev", sent
                else:
                    yield "test", sent

        self._process_stream(stream())

    def _process_stream(self, sentence_stream):
        out_dir = self.output_dir / self.filter.name
        out_dir.mkdir(parents=True, exist_ok=True)

        files = {
            "train_full":    open(out_dir / "train_full.txt",     "w", encoding="utf-8"),
            "dev_full":      open(out_dir / "dev_full.txt",       "w", encoding="utf-8"),
            "test_full":     open(out_dir / "test_full.txt",      "w", encoding="utf-8"),
            "train_clean":   open(out_dir / "train_clean.txt",    "w", encoding="utf-8"),
            "train_matched": open(out_dir / "train_matched.txt",  "w", encoding="utf-8"),
        }

        stats = {s: {"matched": 0, "total": 0} for s in ("train", "dev", "test")}

        try:
            for i, (split, sent) in enumerate(sentence_stream):
                is_match = self.filter._exclude_sent(sent)
                stats[split]["total"] += 1
                if is_match:
                    stats[split]["matched"] += 1
                    sent.metadata["phenomenon"] = self.filter.name

                text = sent.metadata.get("text", "") + "\n"

                files[f"{split}_full"].write(text)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input (pickle, conllu file, or UD dir)")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    filters = [
        ExistentialThereQuantifierFilter(),
        BindingReflexive(),
        InterrogativeWhModifierFilter(),
        LicensedNPI()
    ]
    run_filters(filters, args.input, output_dir=args.output)