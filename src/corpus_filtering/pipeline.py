from corpus_filtering.filters.base import CorpusFilter
from corpus_filtering.filters.base import NotFilter, ExistentialThereQuantifierFilter, BindingReflexive, InterrogativeWhModifierFilter
from conllu import parse_incr
from pathlib import Path
from conllu import TokenList
import json
import random


class FilterPipeline:
    def __init__(self, filter: CorpusFilter, mode: str = "labeled", output_dir: str = "output/",
                 train_ratio: float = 0.90, dev_ratio: float = 0.05,
                 seed: int = 42, shuffle: bool = True):
        self.filter = filter
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = 1.0 - train_ratio - dev_ratio
        self.seed = seed
        self.shuffle = shuffle

    def _write_conllu(self, sentences: list, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for sent in sentences:
                f.write(sent.serialize())

    def _read_conllu(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            for sent in parse_incr(f):
                yield sent

    def _write_txt(self, sentences: list, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for sent in sentences:
                text = sent.metadata.get("text", "")
                f.write(text + "\n")

    def _write_stats(self, splits: dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        stats = {
            "filter": self.filter.name,
            "splits": {
                split_name: {
                    "total": split["total"],
                    "matched": split["matched"],
                    "match_rate": round(split["matched"] / split["total"], 4) if split["total"] else 0,
                }
                for split_name, split in splits.items()
            }
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

    def run(self, input_path: str):
        """Auto-detect whether input is a UD directory or a single CoNLL-U file."""
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
        test_file = next(ud_path.glob("*-ud-test.conllu"))
        dev_file = next(ud_path.glob("*-ud-dev.conllu"))

        train = self._process_split(train_file)
        dev = self._process_split(dev_file)
        test = self._process_split(test_file)

        self._finalize(train, dev, test)

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

        stats = {s: {"matched": 0, "total": 0} for s in ("train", "dev", "test")}

        # Open all output files upfront and stream directly
        def open_writers(base_dir):
            base_dir.mkdir(parents=True, exist_ok=True)
            return {
                split: {
                    "conllu": open(base_dir / f"{split}.conllu", "w", encoding="utf-8"),
                    "txt":    open(base_dir / f"{split}.txt",    "w", encoding="utf-8"),
                }
                for split in ("train", "dev", "test")
            }

        labeled_writers = open_writers(self.output_dir / self.filter.name / "labelled") if self.mode in ("labeled", "both") else None
        clean_writers   = open_writers(self.output_dir / self.filter.name / "clean")    if self.mode in ("clean_train", "both") else None

        try:
            for i, sent in enumerate(self._read_conllu(input_path)):
                is_match = self.filter._exclude_sent(sent)

                if i in train_idx:
                    split_name = "train"
                elif i in dev_idx:
                    split_name = "dev"
                else:
                    split_name = "test"

                stats[split_name]["total"] += 1
                if is_match:
                    stats[split_name]["matched"] += 1
                    sent.metadata["phenomenon"] = self.filter.name

                serialized = sent.serialize()
                text = sent.metadata.get("text", "") + "\n"

                if labeled_writers:
                    labeled_writers[split_name]["conllu"].write(serialized)
                    labeled_writers[split_name]["txt"].write(text)

                if clean_writers:
                    # clean train: only non-matched sentences
                    if split_name == "train" and not is_match:
                        clean_writers["train"]["conllu"].write(serialized)
                        clean_writers["train"]["txt"].write(text)
                    # dev/test: all sentences
                    elif split_name != "train":
                        clean_writers[split_name]["conllu"].write(serialized)
                        clean_writers[split_name]["txt"].write(text)

                if (i + 1) % 100_000 == 0:
                    print(f"  Processed {i + 1:,} sentences", flush=True)

        finally:
            for writers in filter(None, [labeled_writers, clean_writers]):
                for split_files in writers.values():
                    for f in split_files.values():
                        f.close()

        for split_name, s in stats.items():
            print(f"{split_name.capitalize():5}: {s['matched']} matched / {s['total']} total")

        self._write_stats(stats, self.output_dir / "stats.json")
    def _finalize(self, train, dev, test):
        print(f"Train: {train['matched']} matched / {train['total']} total")
        print(f"Dev:   {dev['matched']} matched / {dev['total']} total")
        print(f"Test:  {test['matched']} matched / {test['total']} total")

        if self.mode in ("labeled", "both"):
            labelled_dir = self.output_dir / self.filter.name / "labelled"
            for split_name, split in [("train", train), ("dev", dev), ("test", test)]:
                all_sents = []
                for sent, is_match in split["sentences"]:
                    if is_match:
                        sent.metadata["phenomenon"] = self.filter.name
                    all_sents.append(sent)
                self._write_conllu(all_sents, labelled_dir / f"{split_name}.conllu")
                self._write_txt(all_sents, labelled_dir / f"{split_name}.txt")

        if self.mode in ("clean_train", "both"):
            clean_dir = self.output_dir / self.filter.name / "clean"
            clean_train_sents = [sent for sent, is_match in train["sentences"] if not is_match]
            self._write_conllu(clean_train_sents, clean_dir / "train.conllu")
            self._write_txt(clean_train_sents, clean_dir / "train.txt")
            for split_name, split in [("dev", dev), ("test", test)]:
                all_sents = []
                for sent, is_match in split["sentences"]:
                    if is_match:
                        sent.metadata["phenomenon"] = self.filter.name
                    all_sents.append(sent)
                self._write_conllu(all_sents, clean_dir / f"{split_name}.conllu")
                self._write_txt(all_sents, clean_dir / f"{split_name}.txt")

        splits = {"train": train, "dev": dev, "test": test}
        self._write_stats(splits, self.output_dir / "stats.json")

    def _process_split(self, path):
        sentences = []
        matched_count = 0
        for sent in self._read_conllu(path):
            is_match = self.filter._exclude_sent(sent)
            if is_match:
                matched_count += 1
            sentences.append((sent, is_match))
        return {
            "matched": matched_count,
            "total": len(sentences),
            "sentences": sentences,
        }


def run_filters(filters, input_path, mode="both", output_dir="output/", **kwargs):
    """
    Run multiple filters sequentially on the same input corpus.
    
    Args:
        filters: list of CorpusFilter instances
        input_path: path to UD directory or single CoNLL-U file
        mode: 'labeled', 'clean_train', or 'both'
        output_dir: shared output directory (each filter gets its own subdir)
        **kwargs: passed to FilterPipeline (train_ratio, dev_ratio, seed, shuffle)
    """
    for i, f in enumerate(filters, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(filters)}] Running filter: {f.name}")
        print(f"{'='*60}\n")
        try:
            p = FilterPipeline(f, mode=mode, output_dir=output_dir, **kwargs)
            p.run(input_path)
            print(f"✓ Completed {f.name}")
        except Exception as e:
            print(f"✗ FAILED {f.name}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    INPUT_PATH = "/output_100M.conllu"
    OUTPUT_DIR = "output/"

    filters = [
        NotFilter(),
        ExistentialThereQuantifierFilter(),
        BindingReflexive(),
        InterrogativeWhModifierFilter(),
        # add more as you build them
    ]

    run_filters(filters, INPUT_PATH, mode="both", output_dir=OUTPUT_DIR)