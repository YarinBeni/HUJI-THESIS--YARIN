#!/usr/bin/env python3
"""Character statistics utility for eBL fragments."""
import argparse
import json
from collections import Counter
from pathlib import Path

from common_atf import strip_atf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze characters in eBL fragments")
    parser.add_argument("--input", type=Path, required=True, help="Path to eBL fragments JSON")
    parser.add_argument("--output", type=Path, required=True, help="Path to save char stats JSON")
    parser.add_argument("--min_freq", type=int, default=1, help="Minimum frequency to keep a character")
    return parser.parse_args()


def main() -> None:
    import time
    start_time = time.time()

    args = parse_args()
    print(f"📊 Starting character analysis...")
    print(f"   Input file: {args.input}")
    print(f"   Min frequency: {args.min_freq}")

    load_start = time.time()
    with args.input.open("r", encoding="utf-8") as f:
        fragments = json.load(f)
    load_time = time.time() - load_start
    print(f"   Loaded {len(fragments)} fragments in {load_time:.2f}s")

    counter: Counter[str] = Counter()
    total_chars = 0
    skipped = 0

    process_start = time.time()
    for frag in fragments:
        atf = frag.get("atf")
        if not atf:
            skipped += 1
            continue
        cleaned = strip_atf(atf)
        counter.update(cleaned)
        total_chars += len(cleaned)

    process_time = time.time() - process_start

    vocab = [
        {
            "char": ch,
            "count": freq,
            "codepoint": f"U+{ord(ch):04X}",
        }
        for ch, freq in counter.most_common()
        if freq >= args.min_freq and ch.strip()
    ]

    payload = {
        "input_path": str(args.input),
        "total_fragments": len(fragments),
        "fragments_without_atf": skipped,
        "total_characters": total_chars,
        "unique_characters": len(vocab),
        "vocab": vocab,
    }

    save_start = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    save_time = time.time() - save_start

    total_time = time.time() - start_time
    print(f"✅ Character analysis complete in {total_time:.2f}s")
    print(f"   Processed {len(fragments)} fragments ({skipped} without ATF)")
    print(f"   Total characters (post-clean): {total_chars}")
    print(f"   Unique characters (>= {args.min_freq}): {len(vocab)}")
    print(f"   Processing time: {process_time:.2f}s | Save time: {save_time:.2f}s")
    print(f"   Saved stats to {args.output}")


if __name__ == "__main__":
    main()
