# Learned Experience Libraries

This folder contains the **27 atomic experiences** that TSGD distilled from
the MATH Precalculus Level 5 training split with the Grok-4.1 Fast
Non-Reasoning backbone — the exact library used for the headline numbers in
Table 1 and the ablation in Table 2 of the paper.

## Files

| File | Subject | Backbone | Size | Notes |
|---|---|---|---|---|
| `math_precalculus_level5_27_experiences.jsonl` | MATH Precalculus Level 5 | Grok-4.1 Fast Non-Reasoning | 27 entries | Final post-regularization library (epoch 3) |

## File format

Each line is a single experience in JSONL:

```json
{
  "id": "b02c3b7d",
  "condition": "Trig function as quadratic in bounded variable u=cos θ or sin θ ∈[-1,1].",
  "strategy":  "Complete square A(u-h)²+k; min/max by vertex position relative to [-1,1].",
  "warning":   "Verify algebra: expand back to match coefficients exactly.",
  "subject":   "Precalculus",
  "level":     null,
  "source_id": ["MATH_1305", "MATH_815"],
  "created_by_agent": "Pi_reg",
  "success_count": 0,
  "usage_count": 0,
  "created_at": "2026-01-28 15:14:59",
  "utility_score": 0.0
}
```

- **`condition`**: the trigger predicate ("when do I use this strategy?")
- **`strategy`**: the actual reasoning rule to follow
- **`warning`**: a failure mode the optimizer learned the hard way; surfaced
  to the solver as part of the prompt to prevent regressions
- **`source_id`**: the training MATH problem ids that contributed to this
  entry (useful for tracing how a rule was learned)
- **`created_by_agent`**: `Pi_init` (seeded by initializer), `Pi_opt` (added
  or edited by optimizer), or `Pi_reg` (consolidated by regularizer)
- **`usage_count`** / **`success_count`**: rolling statistics gathered during
  training; reset to 0 in the released snapshot

## Loading

```python
import json
from pathlib import Path

path = Path("learned_libraries/math_precalculus_level5_27_experiences.jsonl")
experiences = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
print(f"Loaded {len(experiences)} experiences")
for e in experiences[:3]:
    print(f"- [{e['id']}] {e['condition']} -> {e['strategy']}")
```

To use this library at inference time, point `src/.env`'s
`EXPERIENCE_DIR` at the parent folder (or modify `scripts/run_eval.sh`).

## Coming later

We plan to release libraries for the remaining MATH subjects, AIME, and the
cross-domain benchmarks (HumanEval, MBPP, ToolBench) in the PMLR
post-conference revision window.
