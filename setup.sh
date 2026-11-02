#!/usr/bin/env bash
# Bootstrap codemason on THIS machine: (re)creates .venv and installs deps.
#
# WHY: a virtualenv is machine-specific — .venv/bin/python is a symlink to the
# local Python interpreter, and packages may be hardlinked from a local cache.
# Copying .venv to another box gives "Broken symlink ... was the underlying
# Python interpreter removed?". So NEVER ship .venv; ship the source and run this.
set -euo pipefail
cd "$(dirname "$0")"

rm -rf .venv
if command -v uv >/dev/null 2>&1; then
  echo "[setup] uv found — creating .venv (python 3.11)"
  uv venv --python 3.11 .venv 2>/dev/null || uv venv .venv
  VIRTUAL_ENV="$PWD/.venv" uv pip install -e .
else
  PY="$(command -v python3.11 || command -v python3)"
  echo "[setup] using $PY"
  "$PY" -m venv .venv
  .venv/bin/python -m pip install -U pip -q
  .venv/bin/python -m pip install -e . -q
fi

# Put `codemason` on PATH. ~/.local/bin is the standard per-user bin dir;
# create it if missing, and warn if the shell won't search it.
BIN="$HOME/.local/bin"
mkdir -p "$BIN"
ln -sfn "$PWD/.venv/bin/codemason" "$BIN/codemason"

echo
echo "[setup] done."
echo "        linked: $BIN/codemason -> $PWD/.venv/bin/codemason"
case ":$PATH:" in
  *":$BIN:"*) echo "        run:    codemason" ;;
  *) echo "        NOTE: $BIN is not on PATH — add it to your shell rc, or run $PWD/.venv/bin/codemason" ;;
esac
