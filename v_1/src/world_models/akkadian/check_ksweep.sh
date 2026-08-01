#!/bin/bash
# One-line status for the PLS k-sweep jobs (WAk 20110 fragments, WBk 20111 cell B).
# Both jobs commit_push their own results, so completion is visible from the git log
# alone — no cluster access needed. Cheap enough to poll.
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)" || exit 1
git fetch -q origin main 2>/dev/null
A=$(git log --oneline origin/main | grep -c "^[0-9a-f]* WAk:")
B=$(git log --oneline origin/main | grep -c "^[0-9a-f]* WBk:")
echo "WAk ${A}/15 arms | WBk ${B}/14 arms"
[ "$A" -ge 15 ] && [ "$B" -ge 14 ] && echo "DONE" || echo "PENDING"
