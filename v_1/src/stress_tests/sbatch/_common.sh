# _common.sh — shared, concurrency-robust git sync for the stress-test sbatch jobs.
#
# WHY: every job runs from the SAME NFS working copy (~/projects/HUJI-THESIS--YARIN)
# and does pull / commit / push against `main`. When jobs (or array tasks) overlap
# they raced and produced the failures seen in the logs:
#   * "error: cannot pull with rebase: You have unstaged changes"  (J8 x5)
#   * "fatal: Cannot rebase onto multiple branches"                 (J5b)
# Root causes: `git pull --rebase origin main` chokes on a dirty tree, and a stale
# multi-line FETCH_HEAD left by a concurrent fetch makes `pull --rebase` ambiguous.
#
# FIX: serialize all git mutations with an flock (best-effort; harmless if flock is
# unavailable or NFS locking is flaky), always rebase onto a SINGLE ref (FETCH_HEAD
# after an explicit `git fetch origin main`), and `--autostash` so an unclean tree
# never aborts the rebase. Any half-finished rebase/merge or stale index.lock from a
# killed job is cleared first.
#
# Source it AFTER `cd`-ing into the repo:
#     source v_1/src/stress_tests/sbatch/_common.sh
#     sync_main
#     ... produce results ...
#     commit_push "J?: message (job ${SLURM_JOB_ID})" path/to/results/*.json

GIT_LOCK="${GIT_LOCK:-$HOME/.hujithesis_git.lock}"

_git_abort_inflight() {
    git rebase --abort 2>/dev/null || true
    git merge  --abort 2>/dev/null || true
    # a CORRUPT rebase state (job killed mid-rebase) survives a failed --abort
    # and then blocks every future rebase with "there is already a rebase-merge
    # directory" (seen: J22b/J22c, jobs 12743/12744). We run under the git lock,
    # so any rebase dir still present here is stale — remove it outright.
    rm -rf .git/rebase-merge .git/rebase-apply 2>/dev/null || true
    # drop a stale lock left by a killed job (only if clearly not in active use)
    find .git -maxdepth 1 -name index.lock -mmin +2 -delete 2>/dev/null || true
}

# Bring the working tree up to date with origin/main, robust to concurrent jobs.
sync_main() {
    ( flock -w 300 9 || true
      _git_abort_inflight
      for _i in 1 2 3 4 5; do
          if git fetch origin main && git rebase --autostash FETCH_HEAD; then
              break
          fi
          _git_abort_inflight
          sleep $((RANDOM % 8 + 5))
      done
    ) 9>"$GIT_LOCK"
}

# Push whatever is already committed onto main, rebasing onto the latest first.
push_main() {
    ( flock -w 600 9 || true
      _git_abort_inflight
      for _i in 1 2 3 4 5; do
          if git fetch origin main && git rebase --autostash FETCH_HEAD \
             && git push origin HEAD:main; then
              echo "[push_main] pushed"; break
          fi
          _git_abort_inflight
          sleep $((RANDOM % 8 + 5))
      done
    ) 9>"$GIT_LOCK"
}

# Stage + commit the given paths and push — all inside one lock so overlapping
# array tasks never collide on the index. Usage: commit_push "msg" path [path...]
commit_push() {
    local msg="$1"; shift
    ( flock -w 600 9 || true
      _git_abort_inflight
      git add "$@" 2>/dev/null || true
      if git diff --cached --quiet; then
          echo "[commit_push] nothing to commit"
      else
          git commit -m "$msg" || echo "[commit_push] commit failed"
      fi
      for _i in 1 2 3 4 5; do
          if git fetch origin main && git rebase --autostash FETCH_HEAD \
             && git push origin HEAD:main; then
              echo "[commit_push] pushed"; break
          fi
          _git_abort_inflight
          sleep $((RANDOM % 8 + 5))
      done
    ) 9>"$GIT_LOCK"
}
