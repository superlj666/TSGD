# TSGD: Textual Stochastic Gradient Descent

Official implementation of **"Textual Stochastic Gradient Descent: Discrete
Optimization of External Memory for Reasoning Language Agents"** (ICML 2026).

TSGD treats a natural-language *experience library* $\Phi$ as a learnable,
non-parametric object and optimizes it under an explicit capacity budget via
discrete Add / Edit / Delete operations driven by *textual gradients*. A
dual-verification mechanism (local repair + global non-regression) keeps the
library well-behaved, and periodic regularization compresses it down to a
compact set of high-utility rules.

On MATH and AIME benchmarks this yields up to **+18.7 pp** over zero-shot
baselines while compressing hundreds of raw experiences into **≈ 30** rules.

> **Paper:** [OpenReview](https://openreview.net/forum?id=zW11MhFLD)
> **Authors:** Jian Li, Hua Huang (corresponding) — Beijing Normal University

---

## ⚙️ Setup

```bash
git clone https://github.com/superlj666/TSGD.git
cd TSGD

# Python environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# API keys
cp src/.env.example src/.env
# Edit src/.env and fill in OPENAI_API_KEY / OPENAI_BASE_URL
# (also works with OpenAI-compatible endpoints such as OpenRouter, xAI, etc.)
```

## 📁 Repository layout

```text
TSGD/
├── src/
│   ├── agents/              # The six TSGD agents
│   │   ├── base_agent.py        # Shared abstract base
│   │   ├── initializer.py       # π_init: distill seed experiences from GT trajectories
│   │   ├── retriever.py         # π_retr: hard-filter + dense recall
│   │   ├── solver.py            # π_solver: reason with retrieved experiences
│   │   ├── evaluator.py         # π_validator: local repair + global non-regression
│   │   ├── optimizer.py         # π_opt: failure-driven Add / Edit / Delete (textual gradient)
│   │   └── regularizer.py       # π_reg: hierarchical pruning under capacity budget B
│   ├── core/
│   │   ├── experience_pool.py   # The Φ data structure
│   │   ├── train.py             # The TSGD optimization loop
│   │   └── inference.py         # Retrieve-then-solve inference engine
│   ├── tools/
│   │   ├── prompts.yaml         # All LLM prompts (role-specific)
│   │   ├── grader.py            # Math correctness checking
│   │   ├── utils.py             # Config, LLM client wrappers, locks
│   │   ├── generate_index.py    # Build embedding index for retrieval
│   │   └── quick_query.py       # CLI helper for ad-hoc retrieval
│   └── .env.example
├── scripts/
│   ├── run_initialization.sh    # Seed library construction (π_init + π_reg)
│   ├── run_train.sh             # Iterative TSGD optimization
│   └── run_eval.sh              # Inference & Pass@k evaluation
├── data/                        # Sample data (see data/README.md)
├── learned_libraries/           # Released learned libraries (see README inside)
├── requirements.txt
├── CITATION.bib
└── LICENSE
```

## 🚀 Quick start (three stages)

### 1) Initialization — seed the library from training trajectories

```bash
bash scripts/run_initialization.sh
```

This invokes the **Initializer** ($\pi_{\text{init}}$) on a small set of
ground-truth trajectories to produce atomic Condition→Strategy experiences,
followed by an initial pass of the **Regularizer** ($\pi_{\text{reg}}$) for
de-particularization and clustering.

### 2) Optimization — run the TSGD loop

```bash
bash scripts/run_train.sh
```

On each failed training query the **Optimizer** ($\pi_{\text{opt}}$) proposes
a discrete operation (Add / Edit / Delete) as a *textual gradient*; the
**Validator** ($\pi_{\text{validator}}$) gates it through local repair + global
non-regression (Definition 3.3 of the paper); the **Regularizer** is invoked
every $N{=}50$ steps to keep $|\Phi| \le B$.

### 3) Evaluation — measure Pass@k on a held-out test set

```bash
bash scripts/run_eval.sh
```

Loads the optimized library and reports Pass@1 / Pass@k under the standard
retrieve-then-solve pipeline.

## 📊 Reproducing paper numbers

The paper's headline results are on MATH (Level 5) and AIME 2024/2025 with
Grok-4.1 Fast Non-Reasoning and GPT-4o-mini as backbones. To reproduce:

1. Set the backbone in `scripts/run_train.sh` (`MODEL_NAME=...`) and provide
   the corresponding API key in `src/.env`.
2. Fetch the full MATH dataset (sample subset already in `data/`; see
   `data/README.md` for full-dataset instructions).
3. Run the three stages in order.

The 27 learned rules used at evaluation time are released under
[`learned_libraries/`](learned_libraries/) (one JSONL file). See
[`learned_libraries/README.md`](learned_libraries/README.md) for the file
format and a loading snippet.

## 🧪 Implementation notes

- The optimizer maintains the library in **JSONL** form, where each line is a
  single experience with fields `{condition, strategy, warning, ...}`.
- Retrieval uses a two-stage hybrid: hard subject filtering (when metadata is
  available) followed by dense recall over `text-embedding-3-large` vectors.
  In open-domain settings (no metadata) the hard filter is bypassed.
- Concurrency: training is multi-threaded with `Read` / `Write` locks on
  $\Phi$, so the library evolves atomically as updates are accepted.
- All LLM calls go through an OpenAI-compatible interface; we have tested
  with OpenAI, OpenRouter, xAI Grok, and self-hosted vLLM endpoints.

## 📜 Citation

```bibtex
@inproceedings{li2026tsgd,
  title     = {Textual Stochastic Gradient Descent: Discrete Optimization of External Memory for Reasoning Language Agents},
  author    = {Li, Jian and Huang, Hua},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## 📄 License

MIT — see `LICENSE`. Datasets retain their original licenses (MATH:
MIT; AIME: public competition).

## 🙏 Acknowledgements

This work is supported in part by the National Natural Science Foundation of
China (No. 62576041, 62106257, 62437001) and by the Fundamental Research
Funds for the Central Universities (No. 2253500001, No. 2251200169).
