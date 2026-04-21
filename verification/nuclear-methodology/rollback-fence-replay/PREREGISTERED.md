# Rollback Fence Replay v1

## Claim Boundary

Cloud-only signed-memory rollback reconstruction test. No local `.hlx` bit
identity and no numerical KV<->SSM state transfer claim.

## Null Hypothesis

After a rollback fence, later reconstruction cannot tell active state from
fenced state.

## Alternative Hypothesis

A signed rollback fence makes a bad memory causally visible for audit while
excluding it from active reconstruction.

## Pass Criteria

- Root, bad decision, fence, and recovery records are all signed.
- The forensic model rejects the bad decision memory ID.
- The forensic model keeps the rollback fence visible.
- The forensic model selects the recovery decision.
- Final policy preserves signed hmem and rollback <= 15 minutes.
- The auditor returns JSON verdict `pass` with no gate failures.

## Controls

- The bad memory remains present and signed.
- The fence is a separate signed memory.
- The model must not simply delete history; it must distinguish inactive from
  non-existent.
