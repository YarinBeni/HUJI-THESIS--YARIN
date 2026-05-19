# Prompt Approval — Phase 1b (pv0/pv1/pv2/pv3)

Status: APPROVED
Approved by: Yarin Benizri
Approval message: "approve"
Approval timestamp: 2026-05-19T13:00:51Z

## Locked notes from review session

- pv0 has **no system prompt** — harness MUST send `system_prompt=""` (empty
  string) to suppress Qwen's default system message. Flag to W2.C harness builder.
- pv2 few-shot pool: W2.C selects 5 holdout fragment IDs AFTER Phase 0's
  `build_balanced_subset.py` locks the eval set. Fragments must be confirmed
  out of all MC draws via `balanced_subset/draws_matrix.npy`.
- Activation pooling: last token inside `<<FRAG>>...</FRAG>>` span on ALL four
  variants. Layer sweep: L0, L15 (Round 1 best), L-1.
- Year output convention: positive integer BCE.
- Ruler normalization: canonical table in pv0.md applies to all four variants.

## File hashes (SHA-256, post-approval)

| File | SHA-256 |
|---|---|
| pv0.md | 624708e3e66a31709b244d2f8ddc77d7556628be31c70e40f30bb9ee043e4d01 |
| pv1.md | 1413becf49cdb9cdcf3a3b66135b821871a73d78b8d38e3c9aab8653c80cfa78 |
| pv2.md | f6397dcd92cb9dadabe9c8e3ce9e8994e440e9f3e17fd5cca02a801e0f2f57df |
| pv3.md | db306a0f4cce10dfbc0acb5fe182db11d72d9e004ddec4d106340082788fd5cd |

## Re-approval policy

If any of these files is edited, this APPROVED.md must be invalidated and
Yarin must re-approve before any new cluster inference job is submitted.
