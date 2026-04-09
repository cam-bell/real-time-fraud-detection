#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=false
WITH_OUTPUTS=false
WITH_MODELS=false

usage() {
  cat <<'EOF'
Usage: bash scripts/vortex_cleanup.sh [--dry-run] [--with-outputs] [--with-models]

Deletes regenerable local artifacts from the fraud project.

Options:
  --dry-run       Print what would be removed without deleting anything.
  --with-outputs  Also delete generated files under outputs/.
  --with-models   Also delete generated model artifacts under models/.
EOF
}

remove_path() {
  local target="$1"
  if [[ ! -e "$target" ]]; then
    return
  fi

  if [[ "$DRY_RUN" == true ]]; then
    printf '[dry-run] remove %s\n' "$target"
    return
  fi

  rm -rf "$target"
  printf 'removed %s\n' "$target"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      ;;
    --with-outputs)
      WITH_OUTPUTS=true
      ;;
    --with-models)
      WITH_MODELS=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

cd "$ROOT_DIR"

remove_path ".venv"
remove_path "venv"
remove_path "env"
remove_path ".pytest_cache"
remove_path ".ruff_cache"

while IFS= read -r -d '' path; do
  remove_path "$path"
done < <(
  find . \
    \( -path "./.git" -o -path "./.venv" -o -path "./venv" -o -path "./env" \) -prune \
    -o -type d \( -name "__pycache__" -o -name ".ipynb_checkpoints" \) -print0
)

while IFS= read -r -d '' path; do
  remove_path "$path"
done < <(
  find . \
    \( -path "./.git" -o -path "./.venv" -o -path "./venv" -o -path "./env" \) -prune \
    -o -type f \( -name "*.pyc" -o -name ".DS_Store" \) -print0
)

if [[ "$WITH_OUTPUTS" == true ]]; then
  remove_path "outputs"
fi

if [[ "$WITH_MODELS" == true ]]; then
  remove_path "models"
fi
