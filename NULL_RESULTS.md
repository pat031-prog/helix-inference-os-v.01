# HeliX Null Results and Blocked Lanes

These lanes stay visible on purpose. They are evidence about the boundaries of the current repo state, not cleanup mistakes.

## Current Null / Failed Lanes

| Result | Current status | Why it stays public | Public wording |
| --- | --- | --- | --- |
| Active Memory AB current local run | `failed_current_local` | The current local test fails because `context_hit_count == 0` in the seeded trial set | Do not cite this as an active positive claim until the retrieval path is fixed and the suite is green again. |
| Cloud Amnesia Derby current local run | `failed_current_local` | The current local test also fails because `context_hit_count == 0` | Keep as a visible regression, not as a green memory-quality claim. |
| Ghost v2 doppelganger / contaminated raw retrieval lane | `falsification_preserved` | It demonstrates why adjudication matters and why raw top-k retrieval is not enough | Cite only as methodology and failure-preservation evidence, not as a product win. |
| Cross-provider triangulation | `blocked_methodology_guardrail` | The lane explicitly blocks overclaiming before enough power or external support exists | It is a guardrail, not public proof of hidden model identity. |

## Blocked / Not Reproducible In Current Checkout

| Lane | Current status | Public wording |
| --- | --- | --- |
| HXQ Zamba2 local quality lane | `blocked_runtime` | Loader/non-finite/runtime issues mean it should not be cited as verified quality evidence. |
| Branch-pruning forensics suite | `partially_skipped` | Keep as a methodology lane, not a green reproducible claim in this checkout. |
| Policy RAG legal debate suite | `skipped_rust_extension_dependency` | Enterprise-facing, but not reproducible in the current checkout without the required extension state. |

## Historical vs Current

Some raw artifacts remain historically interesting even when the current local test lane is red. The rule is:

- historical artifact value may remain
- current local failure must still be stated explicitly

For current public wording, prefer the current suite status over the strongest historical screenshot.
