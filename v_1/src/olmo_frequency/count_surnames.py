#!/usr/bin/env python3
"""STEP 3d: the stronger version of the zero-exposure test.

`count_frequencies.py` counts the exact full name. That produced the most interesting
row in RESULTS.md — 583 people whose full name never appears once, still dated to within
122 years — and also its own biggest caveat: a zero for "Franz Xaver Feuchtmayer" says
nothing about how often "Feuchtmayer" appears. If the surname is common, the model has
seen the person, just not under that exact string, and "never seen" is the wrong label.

So this counts the SURNAME too — the last whitespace-delimited token — giving a second
exposure measure per entity. Exposure is then read as max(full-name, surname), an upper
bound: whatever the model saw, it saw at least this often.

The upper bound is deliberately generous. Making the zero group as small and as
conservative as possible is the point; a "never seen" claim should have to survive the
most sceptical accounting available, not the most flattering one.

The surname measure has an obvious weakness in the other direction — "Smith" collides
across thousands of people, and a one-word name is its own surname — so both are kept
in separate columns rather than merged into one number, and the analysis reports which
it is using.

Reuses the API client, adaptive pacing and circuit breaker from count_frequencies, so
it is resumable and rate-limit-tolerant on the same terms.

    python count_surnames.py                 # every held-out historical figure
    python count_surnames.py --zero-only     # just the 583 that scored zero

Needs outbound HTTPS — run it on the cluster login node.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import count_frequencies as CF                                       # noqa: E402

RESULTS = os.path.join(HERE, "results")
IN = os.path.join(RESULTS, "entity_counts.csv")
OUT = os.path.join(RESULTS, "surname_counts.csv")
FIELDS = ["entity_type", "name", "surname", "full_count", "surname_count",
          "index_used", "error"]


def surname(name):
    """Last whitespace token. For a one-word name that is the name itself, which is
    correct — there is no shorter form to fall back to.

    The heuristic misfires on a minority of Wikidata names: "Johann Bernhard Bach the
    younger" yields "younger", "…Nizam-ul-Mulk Asaf Jah" yields "Jah". Both then return
    a large count, which REMOVES the entity from the never-seen group. Every failure of
    this rule therefore makes the surviving group smaller and the claim harder to make,
    which is the direction an error should point when the claim is "the model never saw
    this person"."""
    parts = str(name).split()
    return parts[-1] if parts else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=None)
    ap.add_argument("--zero-only", action="store_true",
                    help="only entities whose full name scored zero")
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--give-up-after", type=int, default=25)
    args = ap.parse_args()

    if not os.path.exists(IN):
        sys.exit(f"no {IN} — run count_frequencies.py first")
    rows = [r for r in csv.DictReader(open(IN))
            if not r.get("error") and r["entity_type"] == "historical_figure"]
    if args.zero_only:
        rows = [r for r in rows if float(r["count"] or 0) == 0]

    # A one-word name has no distinct surname to look up: the answer is already in
    # entity_counts.csv, so querying it again would spend an API call to learn nothing.
    work, free = [], []
    for r in rows:
        s = surname(r["name"])
        (free if s == r["name"] else work).append((r, s))
    print(f"[plan] {len(rows)} figures | {len(work)} need a surname query | "
          f"{len(free)} are single-word (surname == name, no call needed)")

    index = CF.pick_index(args.index, False)
    os.makedirs(RESULTS, exist_ok=True)

    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            if not r.get("error"):
                done.add(r["name"])
        print(f"[resume] {len(done)} already counted")

    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
            for r, s in free:                       # free rows cost nothing, write once
                w.writerow({"entity_type": r["entity_type"], "name": r["name"],
                            "surname": s, "full_count": r["count"],
                            "surname_count": r["count"], "index_used": index,
                            "error": ""})
        todo = [(r, s) for r, s in work if r["name"] not in done]
        print(f"[plan] {len(todo)} left to query")
        pace, streak, n_err, t0 = CF.Pace(args.sleep), 0, 0, time.time()
        for i, (r, s) in enumerate(todo, 1):
            c, _, err = CF.count_one(index, s, pace=pace)
            streak = streak + 1 if err else 0
            n_err += bool(err)
            if not err:
                pace.faster()
            w.writerow({"entity_type": r["entity_type"], "name": r["name"],
                        "surname": s, "full_count": r["count"],
                        "surname_count": "" if c is None else c,
                        "index_used": index, "error": err or ""})
            f.flush()
            if i % 200 == 0 or i == len(todo):
                rate = i / max(1e-9, time.time() - t0)
                print(f"  {i}/{len(todo)}  {rate:.1f}/s  errors={n_err}  "
                      f"eta={(len(todo) - i) / max(rate, 1e-9) / 60:.1f} min", flush=True)
            if streak >= args.give_up_after:
                print(f"\n[stop] {streak} failures in a row — re-run to resume", flush=True)
                break
            pace.wait()
    print(f"[write] {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
