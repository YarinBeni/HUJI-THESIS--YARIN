# _sandbox.sh — concurrency-robust git sync for the chrono sbatch jobs,
# adapted from v_1/src/stress_tests/sbatch/_common.sh but targeting the
# SANDBOX branch: chrono cluster jobs sync and push yarin-sandbox, NEVER
# main (SLA section 8). Same machinery: all git mutations serialized
# under an flock, rebase always onto the single ref FETCH_HEAD after an
# explicit fetch, --autostash so a dirty NFS tree never aborts, stale
# rebase state / index.lock from killed jobs cleared first.
#
# chrono/artifacts is gitignored, so jobs must copy the small results
# (results.parquet, gate reports) into chrono/reports/ — which IS
# tracked — before commit_push_sandbox.
#
# Source it AFTER cd-ing into the repo:
#     source chrono/sbatch/_sandbox.sh
#     sync_sandbox
#     ... produce results ...
#     commit_push_sandbox "C?: message (job ${SLURM_JOB_ID})" chrono/reports

GIT_LOCK="${GIT_LOCK:-$HOME/.hujithesis_git.lock}"
SANDBOX_BRANCH="${SANDBOX_BRANCH:-yarin-sandbox}"

_git_abort_inflight() {
    git rebase --abort 2>/dev/null || true
    git merge  --abort 2>/dev/null || true
    # a corrupt rebase dir from a killed job survives a failed --abort
    # and blocks every later rebase; we hold the lock, so any state
    # still present here is stale — remove it outright.
    rm -rf .git/rebase-merge .git/rebase-apply 2>/dev/null || true
    find .git -maxdepth 1 -name index.lock -mmin +2 -delete 2>/dev/null || true
}

# Bring the working tree up to date with origin/yarin-sandbox.
sync_sandbox() {
    ( flock -w 300 9 || true
      _git_abort_inflight
      for _i in 1 2 3 4 5; do
          if git fetch origin "${SANDBOX_BRANCH}" \
             && git rebase --autostash FETCH_HEAD; then
              break
          fi
          _git_abort_inflight
          sleep $((RANDOM % 8 + 5))
      done
    ) 9>"$GIT_LOCK"
}

# Stage + commit the given paths and push to yarin-sandbox — all inside
# one lock so overlapping array tasks never collide on the index.
# Usage: commit_push_sandbox "msg" path [path...]
commit_push_sandbox() {
    local msg="$1"; shift
    ( flock -w 600 9 || true
      _git_abort_inflight
      git add "$@" 2>/dev/null || true
      if git diff --cached --quiet; then
          echo "[commit_push_sandbox] nothing to commit"
      else
          git commit -m "$msg" || echo "[commit_push_sandbox] commit failed"
      fi
      for _i in 1 2 3 4 5; do
          if git fetch origin "${SANDBOX_BRANCH}" \
             && git rebase --autostash FETCH_HEAD \
             && git push origin "HEAD:${SANDBOX_BRANCH}"; then
              echo "[commit_push_sandbox] pushed to ${SANDBOX_BRANCH}"
              break
          fi
          _git_abort_inflight
          sleep $((RANDOM % 8 + 5))
      done
    ) 9>"$GIT_LOCK"
}
