#!/usr/bin/env bash
# Serve Muse Glimmer locally with llama.cpp (b10353 or newer, per the Meta docs).
#
# Two modes:
#  1. Local GGUF files you downloaded (recommended on small machines):
#       GLIMMER_GGUF=models/Muse-Glimmer-30B-IQ3_XXS.gguf \
#       GLIMMER_MMPROJ=models/mmproj-Muse-Glimmer-30B-f16.gguf ./scripts/start_glimmer.sh
#     Leave GLIMMER_MMPROJ unset for text-only (saves memory; photo diagnosis then won't work).
#  2. Nothing set: the official Q4_K_M (~17 GB + ~1.4 GB vision) is downloaded on first run.
#     Needs ~24 GB of GPU or unified memory.
#
# Memory knobs:
#   GLIMMER_NGL  layers on the GPU (default 99 = all). On a 4 GB laptop GPU start around 8
#                and raise it until llama.cpp reports out of memory, then step back.
#   GLIMMER_CTX  context tokens (default 8192: 6 guide passages + reasoning + answer fit easily).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"     # relative model paths are relative to the project folder

# Pick up GLIMMER_* settings from .env: a later line beats an earlier one,
# and anything already set in the shell beats .env.
if [[ -f .env ]]; then
  declare -A FROM_FILE=()
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?(GLIMMER_[A-Z_]+)=(.*)$ ]] || continue
    key="${BASH_REMATCH[2]}"; val="${BASH_REMATCH[3]}"
    val="${val%%[[:space:]]#*}"; val="${val%"${val##*[![:space:]]}"}"; val="${val#\"}"; val="${val%\"}"
    FROM_FILE[$key]="$val"
  done < .env
  for key in "${!FROM_FILE[@]}"; do
    if [[ -z "${!key:-}" ]]; then export "$key=${FROM_FILE[$key]}"; fi
  done
fi

: "${GLIMMER_API_KEY:?Set GLIMMER_API_KEY in .env (a password you make up; NOT your Meta key)}"
CTX="${GLIMMER_CTX:-8192}"
NGL="${GLIMMER_NGL:-99}"

# Find llama.cpp: GLIMMER_LLAMA_DIR, then the project's llama.cpp/ folder, then PATH.
SERVE=()
for d in "${GLIMMER_LLAMA_DIR:-}" "$ROOT/llama.cpp" "$ROOT/llama.cpp/build/bin"; do
  [[ -n "$d" ]] || continue
  if [[ -x "$d/llama-server" ]]; then SERVE=("$d/llama-server"); break; fi
  if [[ -x "$d/llama" ]]; then SERVE=("$d/llama" serve); break; fi
done
if [[ ${#SERVE[@]} -eq 0 ]]; then
  if command -v llama-server >/dev/null 2>&1; then SERVE=(llama-server)
  elif command -v llama >/dev/null 2>&1; then SERVE=(llama serve)
  else echo "llama.cpp not found (looked in GLIMMER_LLAMA_DIR, $ROOT/llama.cpp and PATH)." >&2; exit 1; fi
fi
echo "Using ${SERVE[*]}" >&2

if [[ -n "${GLIMMER_GGUF:-}" ]]; then
  [[ -f "$GLIMMER_GGUF" ]] || { echo "No file at GLIMMER_GGUF=$GLIMMER_GGUF" >&2; exit 1; }
  MODEL=(-m "$GLIMMER_GGUF")
  if [[ -n "${GLIMMER_MMPROJ:-}" ]]; then
    [[ -f "$GLIMMER_MMPROJ" ]] || { echo "No file at GLIMMER_MMPROJ=$GLIMMER_MMPROJ" >&2; exit 1; }
    MODEL+=(--mmproj "$GLIMMER_MMPROJ")
  else
    echo "GLIMMER_MMPROJ not set: text only. Photo diagnosis needs the vision projector." >&2
  fi
else
  echo "GLIMMER_GGUF not set in .env: downloading Meta's official Q4 build (~18.5 GB, needs ~24 GB of memory). Ctrl+C to stop." >&2
  MODEL=(-hf meta-models/Muse-Glimmer-30B-GGUF --hf-file Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf)
fi

# Optional flags are only passed if this llama.cpp build lists them in --help (flags get renamed between releases).
HELP="$("${SERVE[@]}" --help 2>&1 || true)"
knows() { [[ -z "$HELP" ]] || grep -Eq -- "(^|[^[:alnum:]-])$1([^[:alnum:]-]|$)" <<<"$HELP"; }
EXTRA=()
[[ "$NGL" == auto ]] || EXTRA+=(-ngl "$NGL")
if [[ "${GLIMMER_NO_MMAP:-0}" == 1 ]]; then
  if knows --no-mmap; then EXTRA+=(--no-mmap)
  else echo "NOTE: this llama.cpp build has no --no-mmap option, so it's left out (same results)." >&2; fi
fi
if [[ -n "${GLIMMER_REASONING_BUDGET:-}" ]]; then
  if knows --reasoning-budget; then EXTRA+=(--reasoning-budget "$GLIMMER_REASONING_BUDGET")
  else echo "NOTE: this llama.cpp build has no --reasoning-budget option, so thinking isn't capped." >&2; fi
fi
if knows --chat-template-kwargs; then EXTRA+=(--chat-template-kwargs '{"reasoning_strength":"low"}'); fi

# Sampling from the model card (temp 1.0, top_p 0.95, top_k 64). Never temperature 0: it loops.
exec "${SERVE[@]}" "${MODEL[@]}" \
  -a muse-glimmer -c "$CTX" -np 1 "${EXTRA[@]}" \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --host 127.0.0.1 --port 8080 --api-key "$GLIMMER_API_KEY"
