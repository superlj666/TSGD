# Data

The headline experiments in the paper use:

| Dataset | Use | Source |
|---|---|---|
| **MATH** (Hendrycks et al.) | training + in-distribution evaluation | [HuggingFace `EleutherAI/hendrycks_math`](https://huggingface.co/datasets/EleutherAI/hendrycks_math) |
| **AIME 2024 / 2025** | out-of-distribution evaluation | [HuggingFace `Maxwell-Jia/AIME_2024`](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) and 2025 variants |
| **HumanEval / MBPP / ToolBench** | cross-domain validation (Appendix F) | HuggingFace, see individual cards |

## Downloading the full MATH dataset

```bash
pip install datasets
python -c "from datasets import load_dataset; load_dataset('EleutherAI/hendrycks_math', 'precalculus')"
```

Then export to JSONL in the format expected by the scripts:

```python
from datasets import load_dataset
import json, pathlib

ds = load_dataset("EleutherAI/hendrycks_math", "precalculus")
out = pathlib.Path("data")
out.mkdir(exist_ok=True)

for split_name, split in ds.items():
    target = out / f"math_precalculus_5_{split_name}.jsonl"
    with target.open("w") as f:
        for row in split:
            if row["level"] == "Level 5":  # we use Level 5 only
                f.write(json.dumps(row) + "\n")
    print(f"Wrote {target} ({sum(1 for _ in target.open())} examples)")
```

## Sample data shipped with this repo

A tiny sample (5 problems per subject) is included under
`data/samples/` so you can smoke-test the pipeline without downloading the
full benchmark. **Do not use the sample data for evaluation** — accuracy
numbers on 5 examples are not meaningful.

## File format

Each line of `math_*.jsonl` is one problem:

```json
{"problem": "...", "level": "Level 5", "type": "Precalculus", "solution": "...", "answer": "..."}
```

The `solution` field is used by the Initializer to distill seed experiences;
the `answer` field is used by the grader (`src/tools/grader.py`).
