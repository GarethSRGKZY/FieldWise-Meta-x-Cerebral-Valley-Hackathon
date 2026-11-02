"""Start Muse Glimmer with llama.cpp: the same as start_glimmer.ps1 / .sh, in plain Python.

    python scripts/start_glimmer.py

Works on Windows without touching PowerShell's execution policy. Reads .env (GLIMMER_GGUF,
GLIMMER_MMPROJ, GLIMMER_NGL, GLIMMER_CTX, GLIMMER_API_KEY), finds llama-server in the
project's llama.cpp folder (or GLIMMER_LLAMA_DIR, or PATH) and runs it in this window.
Leave the window open while using Fieldwise; Ctrl+C stops it.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WIN = os.name == "nt"
OFFICIAL = ["-hf", "meta-models/Muse-Glimmer-30B-GGUF", "--hf-file", "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"]


def find_llama() -> tuple[Path | None, list[str]]:
    """(executable, extra args). Looks in GLIMMER_LLAMA_DIR, then ./llama.cpp (up to 3 levels down,
    since Windows "Extract All" adds a subfolder), then PATH."""
    names = [("llama-server.exe", []), ("llama.exe", ["serve"])] if WIN else [("llama-server", []), ("llama", ["serve"])]
    for d in filter(None, [os.environ.get("GLIMMER_LLAMA_DIR"), str(ROOT / "llama.cpp")]):
        d = Path(d)
        if not d.is_dir():
            continue
        for name, pre in names:
            hits = sorted((p for p in d.rglob(name) if len(p.relative_to(d).parts) <= 3), key=lambda p: len(p.parts))
            if hits:
                return hits[0], pre
    for name, pre in (("llama-server", []), ("llama", ["serve"])):
        found = shutil.which(name)
        if found:
            return Path(found), pre
    return None, []


def server_help(exe: Path, pre: list[str]) -> str:
    """llama-server's --help text, so optional flags are only passed when this build knows them
    (llama.cpp renames and drops flags between releases). Empty if --help couldn't be read."""
    try:
        r = subprocess.run([str(exe), *pre, "--help"], capture_output=True, text=True, timeout=90,
                           encoding="utf-8", errors="replace")
        return (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.TimeoutExpired):
        return ""


def knows(help_text: str, flag: str) -> bool:
    """True if `flag` appears as a whole option in the help text (or the help couldn't be read: then try it)."""
    if not help_text:
        return True
    return re.search(rf"(?<![\w-]){re.escape(flag)}(?![\w-])", help_text) is not None


def _path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    from fieldwise.config import settings   # loads .env into the environment

    key = settings.glimmer_api_key
    if not key:
        sys.exit("Set GLIMMER_API_KEY in .env (a password you make up; NOT your Meta key).")
    if settings.meta_api_key and key == settings.meta_api_key:
        sys.exit("GLIMMER_API_KEY is your Meta key. Use a made-up password instead.")
    ctx = os.environ.get("GLIMMER_CTX") or "8192"
    ngl = os.environ.get("GLIMMER_NGL") or "99"
    url = urlparse(settings.glimmer_base_url)
    host, port = url.hostname or "127.0.0.1", str(url.port or 8080)

    exe, pre = find_llama()
    if not exe:
        sys.exit(f"llama.cpp not found. Unzip the Windows CUDA release and its cudart zip into {ROOT / 'llama.cpp'}")
    print(f"Using {exe}")
    if WIN and not list(exe.parent.glob("cudart64_*.dll")):
        print(f"WARNING: no cudart64_*.dll next to {exe.name}. Unzip the matching cudart-llama-bin-win-cuda-*.zip "
              f"into {exe.parent}, or the GPU won't be used.")

    gguf, mmproj = os.environ.get("GLIMMER_GGUF"), os.environ.get("GLIMMER_MMPROJ")
    if gguf:
        if not _path(gguf).exists():
            sys.exit(f"No file at GLIMMER_GGUF={gguf}. Run python scripts/download_glimmer.py")
        model = ["-m", str(_path(gguf))]
        if mmproj:
            if not _path(mmproj).exists():
                sys.exit(f"No file at GLIMMER_MMPROJ={mmproj}")
            model += ["--mmproj", str(_path(mmproj))]
        else:
            print("GLIMMER_MMPROJ not set: text only. Photo diagnosis needs the vision projector.")
    else:
        print("WARNING: GLIMMER_GGUF not set in .env: downloading Meta's official Q4 build "
              "(~18.5 GB, needs ~24 GB of memory). Ctrl+C to stop.")
        model = OFFICIAL

    # Sampling from the model card (temp 1.0, top_p 0.95, top_k 64). Never temperature 0: it loops.
    # GLIMMER_NGL=auto: don't pass -ngl, so llama.cpp's automatic fitting picks how many layers go on the GPU.
    ngl_args = [] if ngl.lower() == "auto" else ["-ngl", ngl]
    cmd = [str(exe), *pre, *model, "-a", settings.glimmer_model, *ngl_args, "-c", ctx, "-np", "1",
           "--temp", "1.0", "--top-p", "0.95", "--top-k", "64", "--host", host, "--port", port, "--api-key", key]

    # Optional flags. Each is only passed if this llama.cpp build lists it in --help, so a renamed
    # or removed flag gives a warning instead of "error: invalid argument".
    #   --chat-template-kwargs      server-wide reasoning_strength (the app also sends it with every request)
    #   GLIMMER_NO_MMAP=1           load the whole model into RAM up front instead of reading it from disk on demand
    #   GLIMMER_REASONING_BUDGET=N  cap Glimmer's thinking tokens (Meta docs: reasoning can't be turned off, only capped)
    wanted: list[tuple[str, list[str], str]] = [
        ("--chat-template-kwargs", [json.dumps({"reasoning_strength": settings.glimmer_reasoning_strength})],
         "fine: the app sends reasoning_strength with every request anyway"),
    ]
    if os.environ.get("GLIMMER_NO_MMAP", "0") == "1":
        wanted.append(("--no-mmap", [], "the model is read from disk on demand instead: slower first answer, same results"))
    if os.environ.get("GLIMMER_REASONING_BUDGET"):
        wanted.append(("--reasoning-budget", [os.environ["GLIMMER_REASONING_BUDGET"]],
                       "Glimmer's thinking isn't capped; answers may take longer"))
    help_text = server_help(exe, pre)
    if not help_text:
        print("Couldn't read llama-server --help; passing the optional flags anyway.")
    for flag, values, if_skipped in wanted:
        if knows(help_text, flag):
            cmd += [flag, *values]
            continue
        print(f"NOTE: this llama.cpp build has no {flag} option, so it's left out ({if_skipped}).")
        stem = flag.lstrip("-").split("-")[-1]                     # e.g. "mmap", "budget"
        similar = [ln.strip() for ln in help_text.splitlines() if stem in ln.lower()][:4]
        if similar:
            print("      Related options in this build's --help (paste these into the chat if you want it back):")
            for ln in similar:
                print(f"        {ln[:110]}")
    print(f"Loading the model. Ready when this window says the server is listening on http://{host}:{port}.")
    print("Leave this window open while you use Fieldwise. Ctrl+C stops it.\n", flush=True)
    try:
        rc = subprocess.call(cmd, cwd=ROOT)
    except KeyboardInterrupt:
        rc = 0
    if rc:
        print(f"\nllama.cpp stopped (exit code {rc}). The lines above say why; paste the last 20 into the chat.")


if __name__ == "__main__":
    main()
