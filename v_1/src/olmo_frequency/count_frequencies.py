#!/usr/bin/env python3
"""STEP 3a: how often does each entity appear in OLMo's training data?

This is the whole point of using OLMo. Every other arm in the thesis can be probed
but not audited — nobody publishes Llama's or Qwen's training data, so "salient" vs
"obscure" stays a judgement call. OLMo ships an open corpus, so the judgement call can
be replaced by a count.

Counts come from infini-gram (arXiv 2401.17377), which serves exact n-gram counts over
Dolma-class corpora through a free web API — no local suffix array, no download. One
POST per entity string.

    {"index": ..., "query_type": "count", "query": "Ashurbanipal"}
        -> {"count": 4213, "approx": false}

WHICH INDEX MATTERS. OLMo-2-1124-7B was pretrained on olmo-mix-1124, which is not
byte-identical to Dolma v1.7. If the exact index is not served, a near neighbour still
gives a usable frequency ORDERING (the thing the correlation actually uses), but the
substitution has to be stated, so the index that answered is written into every row of
the CSV and `--strict` refuses anything but an exact match.

SAMPLING. historical_figure holds ~37.5k people; at the API's polite rate that is most
of a day. `--n-sample` draws a century-stratified subset instead (default 4000), which
is plenty for both the overall correlation and the within-century control, and keeps
every century populated. assyrian_ruler is tiny and is always counted whole.

RESUMABLE. Rows are appended and flushed as they arrive and the output is re-read on
start, so an interrupted run continues instead of restarting. Re-running after a
completed run is a no-op.

    python count_frequencies.py --list-indexes
    python count_frequencies.py                       # the real run
    python count_frequencies.py --n-sample 1000       # a quicker first pass

Needs outbound HTTPS to api.infini-gram.io — run it on the cluster login node.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
WM = os.path.join(os.path.dirname(HERE), "world_models")
DATA = os.path.join(WM, "data", "entity_datasets")
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "entity_counts.csv")

API = "https://api.infini-gram.io/"
# olmo-mix-1124 is OLMo-2-1124-7B's own pretraining mix; the dolma indexes are the
# documented fallbacks, in decreasing order of closeness.
INDEX_PREFS = ["v4_olmo-mix-1124_llama", "v4_dolma-v1_7_llama", "v4_dolma-v1_6_llama"]

FIELDS = ["entity_type", "name", "query", "count", "approx", "index_used",
          "target", "century", "short_name", "error"]


# --------------------------------------------------------------------------- API

def api(payload, timeout=30):
    req = urllib.request.Request(
        API, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def list_indexes():
    """The API reports the valid index list inside the error for a bogus one."""
    try:
        r = api({"index": "__nope__", "query_type": "count", "query": "a"})
    except urllib.error.HTTPError as e:
        r = json.load(e)
    except Exception as e:                                           # noqa: BLE001
        return f"could not reach the API: {type(e).__name__}: {e}"
    return r.get("error") or json.dumps(r)


def pick_index(requested, strict):
    """Return the index that actually answers, or exit with a clear reason."""
    order = [requested] if requested else list(INDEX_PREFS)
    for idx in order:
        try:
            r = api({"index": idx, "query_type": "count", "query": "Ashurbanipal"})
        except Exception as e:                                       # noqa: BLE001
            print(f"[index] {idx}: unreachable ({type(e).__name__}: {e})")
            continue
        if "error" in r:
            print(f"[index] {idx}: {r['error']}")
            continue
        print(f"[index] using {idx}  (probe query 'Ashurbanipal' -> {r.get('count')})")
        if strict and idx != INDEX_PREFS[0]:
            sys.exit(f"--strict: wanted {INDEX_PREFS[0]}, only {idx} is served. "
                     "Either drop --strict and record the substitution in RESULTS.md, "
                     "or stop — a count from a different corpus cannot support a causal "
                     "claim about this checkpoint.")
        return idx
    sys.exit("no usable infini-gram index. Run --list-indexes to see what is served, "
             "and check this host actually has outbound HTTPS.")


def count_one(index, query, retries=8, pace=None):
    """One count, with backoff. Returns (count, approx, error-or-None).

    The first run of this script sailed through ~100 queries and then failed on
    essentially every one after — the signature of a server-side rate limit, not of
    bad queries. Two changes follow from that:

      * a 429 (or a transport error, which is how a throttling proxy often shows up)
        waits and RETRIES the same query rather than burning one of a handful of
        attempts, with the wait capped so a long stall cannot run away.
      * `pace`, if given, is nudged upward on every throttle, so the run self-corrects
        to whatever rate the API will actually accept instead of hammering it at a
        fixed interval that has already been rejected.
    """
    delay = 2.0
    for i in range(retries):
        try:
            r = api({"index": index, "query_type": "count", "query": query})
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                if pace is not None:
                    pace.slower()
                time.sleep(min(delay, 60))
                delay *= 2
                continue
            return None, None, f"HTTP {e.code}"
        except Exception as e:                                       # noqa: BLE001
            if pace is not None:
                pace.slower()
            time.sleep(min(delay, 60))
            delay *= 2
            if i == retries - 1:
                return None, None, f"{type(e).__name__}: {e}"
            continue
        if "error" in r:
            return None, None, str(r["error"])[:160]
        return int(r.get("count", 0)), bool(r.get("approx", False)), None
    return None, None, f"rate-limited, {retries} attempts exhausted"


class Pace:
    """Adaptive delay between calls: back off hard on a throttle, creep back down
    while things are going well. Keeps a long run near the fastest rate the API
    tolerates without needing the right --sleep guessed up front."""

    def __init__(self, start, cap=8.0):
        self.d, self.cap, self.hits = start, cap, 0

    def slower(self):
        self.hits += 1
        self.d = min(self.cap, max(0.2, self.d * 2))

    def faster(self):
        self.d = max(0.05, self.d * 0.97)

    def wait(self):
        time.sleep(self.d)


# ------------------------------------------------------------------------ inputs

def rows_historical(n_sample, seed):
    """Century-stratified sample of historical figures, with the probe's target."""
    # HELD-OUT ROWS ONLY. The analysis uses generalisation error, so a counted entity
    # that sat in the probe's training split is a wasted API call — it never appears in
    # the join. Sampling from is_test up front makes every call count.
    rows = [r for r in csv.DictReader(open(os.path.join(DATA, "historical_figure.csv")))
            if r.get("death_year") not in (None, "", "nan")
            and str(r.get("is_test", "")).strip().lower() == "true"]
    by_century = {}
    for r in rows:
        try:
            c = int(float(r["death_century"]))
        except (TypeError, ValueError):
            continue
        by_century.setdefault(c, []).append(r)

    rng = random.Random(seed)
    picked = []
    if n_sample and n_sample < len(rows):
        # even quota per century, then top up from the biggest bins so the total is
        # hit exactly even when thin centuries cannot fill their share
        per = max(1, n_sample // max(1, len(by_century)))
        for c, rs in by_century.items():
            rng.shuffle(rs)
            picked += rs[:per]
        leftovers = [r for c, rs in by_century.items() for r in rs[per:]]
        rng.shuffle(leftovers)
        picked += leftovers[:max(0, n_sample - len(picked))]
    else:
        picked = [r for rs in by_century.values() for r in rs]

    out = []
    for r in picked:
        out.append({"entity_type": "historical_figure",
                    "name": r["name"],
                    "query": r["name"],
                    "target": r["death_year"],
                    "century": int(float(r["death_century"])),
                    # single-token surnames like "Adams" collide with common words;
                    # flagged, not dropped, so the analysis can test both ways
                    "short_name": int(len(r["name"].split()) < 2)})
    return out


def rows_assyrian():
    """Every Assyrian ruler — the obscure end, and only a few dozen rows."""
    seen, out = set(), []
    for r in csv.DictReader(open(os.path.join(DATA, "assyrian_ruler.csv"))):
        nm = r["name"]
        if nm in seen:
            continue
        seen.add(nm)
        # same column and sign convention the probe itself uses (entity_data.FEATURES);
        # these are BCE dates stored as positive years, so bigger = older
        try:
            yr = float(r.get("death_year") or "nan")
        except ValueError:
            yr = float("nan")
        out.append({"entity_type": "assyrian_ruler", "name": nm, "query": nm,
                    "target": "" if yr != yr else str(yr),
                    "century": int(yr // 100 * 100) if yr == yr else 0,
                    "short_name": int(len(nm.split()) < 2)})
    return out


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=None, help="force one infini-gram index")
    ap.add_argument("--strict", action="store_true",
                    help="refuse anything but OLMo's own training mix")
    # restricting to held-out rows already cuts 37.5k people to ~7.5k, which is a
    # ~50 min run, so the default counts all of them and the flag is for a quick pass
    ap.add_argument("--n-sample", type=int, default=0,
                    help="historical figures to sample (0 = every held-out row, ~7.5k)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.4,
                    help="starting seconds between calls; adapts up on throttling "
                         "and back down when clear, so this is a hint not a fixed rate")
    ap.add_argument("--list-indexes", action="store_true")
    args = ap.parse_args()

    if args.list_indexes:
        print(list_indexes())
        return 0

    index = pick_index(args.index, args.strict)
    os.makedirs(RESULTS, exist_ok=True)

    # resume: anything already counted (successfully) is not re-queried
    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            if not r.get("error"):
                done.add((r["entity_type"], r["name"]))
        print(f"[resume] {len(done)} rows already counted in {OUT}")

    work = rows_assyrian() + rows_historical(args.n_sample, args.seed)
    todo = [r for r in work if (r["entity_type"], r["name"]) not in done]
    print(f"[plan] {len(work)} entities, {len(todo)} left to count, index={index}")
    if not todo:
        print("[done] nothing to do")
        return 0

    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        t0, n_err, seen_err = time.time(), 0, {}
        pace = Pace(args.sleep)
        for i, r in enumerate(todo, 1):
            c, approx, err = count_one(index, r["query"], pace=pace)
            if err:
                n_err += 1
                # keep one example of each distinct failure: "errors=298" on its own
                # says nothing about whether to wait, fix a query, or stop
                seen_err.setdefault(err.split(":")[0][:60], (r["name"], err))
            else:
                pace.faster()
            w.writerow({**r, "count": "" if c is None else c,
                        "approx": "" if approx is None else int(approx),
                        "index_used": index, "error": err or ""})
            f.flush()                    # a killed run keeps everything it earned
            if i % 100 == 0 or i == len(todo):
                rate = i / max(1e-9, time.time() - t0)
                print(f"  {i}/{len(todo)}  {rate:.1f}/s  errors={n_err}  "
                      f"delay={pace.d:.2f}s  throttles={pace.hits}  "
                      f"eta={(len(todo) - i) / max(rate, 1e-9) / 60:.1f} min",
                      flush=True)
                for kind, (nm, msg) in list(seen_err.items())[:3]:
                    print(f"      e.g. {nm!r}: {msg}", flush=True)
                seen_err.clear()
            pace.wait()

    print(f"[write] {OUT}  ({n_err} errors)")
    if n_err:
        print("      re-run to retry the failed rows; successes are not re-queried")
    return 0


if __name__ == "__main__":
    sys.exit(main())
