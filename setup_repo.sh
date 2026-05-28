#!/usr/bin/env bash
# ============================================================================
# TSGD public repository setup script.
#
# What it does (idempotent — safe to re-run):
#   1. Copies the cleaned Python sources from icml2026/Rebuttal/Additional/Code/
#      into ./src/ and ./scripts/, skipping __pycache__, debug/, .env, etc.
#   2. Renames legacy internal codenames (STGD / PTGD / ExpLearn / Self-Taught
#      Gradient Descent) to the published name (TSGD / Textual Stochastic
#      Gradient Descent).
#   3. Adds the package __init__.py files needed for clean imports.
#   4. Populates data/samples/ from ../ExpLearn/wasted/glm/data/MATH/ if found.
#   5. git init, first commit, and configures the remote
#      git@github.com:superlj666/TSGD.git. It does NOT push — final push is
#      manual so you can review the staged content first.
#
# Usage:
#   cd "/Users/superlj666/Library/CloudStorage/OneDrive-个人/Writing/Writing_202512_ExpSearch/TSGD-public"
#   bash setup_repo.sh
#   # Inspect, then:
#   git remote -v
#   git log
#   git push -u origin main
# ============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "${REPO_ROOT}")"            # Writing_202512_ExpSearch/
CODE_SRC="${PARENT_DIR}/icml2026/Rebuttal/Additional/Code"
EXPLEARN_DIR="${PARENT_DIR}/ExpLearn"

cd "${REPO_ROOT}"

# ---------- Step 1: copy sources --------------------------------------------
echo "[1/6] Copying sources from ${CODE_SRC} ..."
if [ ! -d "${CODE_SRC}" ]; then
  echo "  ERROR: ${CODE_SRC} not found. Aborting."
  exit 1
fi

# src/ tree
mkdir -p src/agents src/core src/tools scripts

# rsync excludes cruft cleanly; fall back to cp if rsync missing
copy_clean() {
  local src="$1"
  local dst="$2"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a \
      --exclude='__pycache__/' \
      --exclude='*.pyc' \
      --exclude='.DS_Store' \
      --exclude='debug/' \
      --exclude='.env' \
      "${src}/" "${dst}/"
  else
    cp -R "${src}/" "${dst}/"
    find "${dst}" \( -name __pycache__ -o -name "*.pyc" -o -name .DS_Store -o -name debug \) -exec rm -rf {} + 2>/dev/null || true
    [ -f "${dst}/.env" ] && rm -f "${dst}/.env"
  fi
}

copy_clean "${CODE_SRC}/src/agents" src/agents
copy_clean "${CODE_SRC}/src/core"   src/core
copy_clean "${CODE_SRC}/src/tools"  src/tools

cp -f "${CODE_SRC}/src/.env.example" src/.env.example
cp -f "${CODE_SRC}/requirements.txt" ./requirements.txt
cp -f "${CODE_SRC}/run_initialization.sh" scripts/run_initialization.sh
cp -f "${CODE_SRC}/run_train.sh"          scripts/run_train.sh
cp -f "${CODE_SRC}/run_eval.sh"           scripts/run_eval.sh
chmod +x scripts/*.sh

# ---------- Step 2: add __init__.py for clean packaging ----------------------
echo "[2/6] Creating empty __init__.py files ..."
touch src/__init__.py src/agents/__init__.py src/core/__init__.py src/tools/__init__.py

# ---------- Step 3: rename internal codenames -> TSGD -----------------------
echo "[3/6] Renaming STGD / PTGD / Self-Taught Gradient Descent -> TSGD ..."

# Use perl -i for portable in-place edit (macOS BSD sed has different syntax).
RENAME_PATTERNS=(
  's/Self-Taught Gradient Descent/Textual Stochastic Gradient Descent/g'
  's/Prompt-Tuning Gradient Descent/Textual Stochastic Gradient Descent/g'
  's/Dual-Loop Cognitive Architecture/TSGD Multi-Agent Architecture/g'
  's/\bSTGD\b/TSGD/g'
  's/\bPTGD\b/TSGD/g'
  's/\bExpLearn\b/TSGD/g'
)

# Apply across all the staged code files.
find src scripts -type f \( -name "*.py" -o -name "*.sh" -o -name "*.yaml" -o -name "*.example" \) | while read -r f; do
  for pat in "${RENAME_PATTERNS[@]}"; do
    perl -i -pe "${pat}" "${f}"
  done
done

# ---------- Step 4: populate data/samples/ if the original tree is around ---
echo "[4/6] Populating data/samples/ (best effort) ..."
SAMPLE_SRC="${EXPLEARN_DIR}/wasted/glm/data/MATH"
if [ -d "${SAMPLE_SRC}" ]; then
  mkdir -p data/samples
  # grab up to 2 example files per subject
  for subject_dir in "${SAMPLE_SRC}/train"/*/; do
    subject=$(basename "${subject_dir}")
    count=0
    for f in "${subject_dir}"*.json; do
      [ -f "${f}" ] || continue
      cp -f "${f}" "data/samples/${subject}_$(basename ${f})"
      count=$((count + 1))
      [ "${count}" -ge 2 ] && break
    done
  done
  echo "  Copied $(ls data/samples/*.json 2>/dev/null | wc -l) sample files"
else
  echo "  (skipped: ${SAMPLE_SRC} not found)"
fi

# ---------- Step 5: git init + initial commit (no remote) -------------------
echo "[5/6] Initializing git repository ..."
if [ ! -d .git ]; then
  git init -b main
  git add .
  git commit -m "Initial commit: TSGD reference implementation (ICML 2026)"
  echo "  git initialized on branch 'main' with one commit."
else
  echo "  (skipped: .git already exists; if you want a fresh repo, remove .git/ first)"
fi

# ---------- Step 6: final instructions ---------------------------------------
echo ""
echo "============================================================================"
echo "Setup complete. The local repo is ready; remote is intentionally NOT"
echo "configured so you can pick HTTPS or SSH yourself."
echo ""
echo "Next steps:"
echo ""
echo "  1) Sanity check:"
echo "       ls -la"
echo "       git log --oneline"
echo "       git status"
echo ""
echo "  2) Wire up the remote and push (your GitHub repo must exist first):"
echo "       git remote add origin https://github.com/superlj666/TSGD.git"
echo "       git branch -M main           # no-op if already main"
echo "       git push -u origin main"
echo ""
echo "  3) Confirm in the browser:"
echo "       https://github.com/superlj666/TSGD"
echo "============================================================================"
