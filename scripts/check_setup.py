"""Check the hub setup and say what to fix. Safe to run any time; it never calls the Meta API.

    python scripts/check_setup.py

Checks .env, the model files, llama.cpp (and whether it can see the GPU), whether the
Glimmer server is answering, and the guide index. Paste the output into the chat if stuck.
"""
from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, BAD, WARN = "  [ok]  ", "  [FIX] ", "  [!]   "
problems = 0


def say(mark: str, msg: str) -> None:
    global problems
    problems += mark == BAD
    print(mark + msg)


def mask(v: str) -> str:
    return f"{v[:4]}… ({len(v)} chars)" if len(v) > 6 else "(set)"


def find_llama() -> Path | None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from start_glimmer import find_llama as _find   # same search as the launcher
    return _find()[0]


_NEEDS = re.compile(rb"(cudart64_\d+|cublas64_\d+|cublasLt64_\d+)\.dll", re.I)


def check_gpu_dlls(folder: Path) -> None:
    """Windows builds load the GPU backend from ggml-cuda.dll / ggml-vulkan.dll. If that DLL is missing,
    or a CUDA runtime DLL it imports isn't next to it, llama.cpp silently falls back to the CPU."""
    dlls = {p.name.lower() for p in folder.glob("*.dll")}
    backends = sorted(n for n in dlls if n.startswith("ggml-") and n[5:-4] not in ("base",) and not n[5:].startswith("cpu"))
    print(f"          GPU backend DLLs here: {', '.join(backends) or 'none'}")
    if not any(n in dlls for n in ("ggml-cuda.dll", "ggml-vulkan.dll")):
        has_cuda_inside = any(_NEEDS.search(p.read_bytes()) for p in [*folder.glob("*.exe"), *folder.glob("*.dll")]
                              if p.stat().st_size < 400_000_000 and not p.name.lower().startswith("cu"))
        if not has_cuda_inside:
            say(BAD, "No ggml-cuda.dll here: this looks like a CPU-only (or wrong) download. Get the "
                     "llama-b#####-bin-win-cuda-<ver>-x64.zip and unzip it into this folder.")
            return
    needed: set[str] = set()
    for p in folder.glob("*.dll"):
        if p.name.lower().startswith(("ggml", "llama", "mtmd")):
            needed |= {m.decode().lower() + ".dll" for m in _NEEDS.findall(p.read_bytes())}
    if not needed:
        return
    missing = sorted(n for n in needed if n not in dlls)
    have = sorted(n for n in dlls if n.startswith(("cudart64_", "cublas64_", "cublaslt64_")))
    if missing:
        say(BAD, f"The CUDA build needs {', '.join(missing)} but this folder has {', '.join(have) or 'no CUDA DLLs'}. "
                 "Download the cudart-llama-bin-win-cuda-<ver>-x64.zip whose <ver> matches the main zip "
                 "(12.x needs *_12.dll, 13.x needs *_13.dll) and unzip it into this same folder.")
    else:
        say(OK, f"CUDA runtime DLLs present: {', '.join(sorted(needed))}")


def main() -> None:
    print(f"Fieldwise setup check ({platform.system()} {platform.release()}, Python {platform.python_version()})")
    print(f"Project folder: {ROOT}\n")

    # ---- .env and keys
    print(".env")
    if not (ROOT / ".env").exists():
        say(BAD, "No .env file. Copy .env.example to .env in this folder, then fill it in.")
    from fieldwise.config import settings  # reads .env

    if settings.meta_api_key:
        mark = OK if settings.meta_api_key.startswith("LLM_") else WARN
        say(mark, f"MODEL_API_KEY {mask(settings.meta_api_key)}"
            + ("" if mark == OK else "  (Meta keys normally start with LLM_)"))
    else:
        say(WARN, "MODEL_API_KEY not set: no Muse Spark or SAM 3.1 (cloud) until you add it.")
    gk = settings.glimmer_api_key or ""
    if not gk:
        say(BAD, "GLIMMER_API_KEY not set. Make up any password and put it in .env.")
    elif settings.meta_api_key and gk == settings.meta_api_key:
        say(BAD, "GLIMMER_API_KEY is your Meta key. Use a made-up password instead; it's shown on the llama.cpp command line.")
    else:
        say(OK, "GLIMMER_API_KEY set (local password)")
    m = settings.glimmer_model
    if m.endswith(".gguf") or "/" in m or "\\" in m:
        say(BAD, f"GLIMMER_MODEL={m} is a file path. Set GLIMMER_MODEL=muse-glimmer and put the path in GLIMMER_GGUF.")
    else:
        say(OK, f"GLIMMER_MODEL={m}")

    # ---- model files
    print("\nModel files")
    for var, min_gb in (("GLIMMER_GGUF", 5.0), ("GLIMMER_MMPROJ", 0.3)):
        val = os.environ.get(var, "")
        if not val:
            say(BAD if var == "GLIMMER_GGUF" else WARN, f"{var} not set in .env"
                + (" (photo diagnosis needs it)" if var == "GLIMMER_MMPROJ" else ""))
            continue
        p = Path(val) if Path(val).is_absolute() else ROOT / val
        if not p.exists():
            say(BAD, f"{var}: no file at {p}. Run python scripts/download_glimmer.py")
        elif p.stat().st_size / 1e9 < min_gb:
            say(BAD, f"{var}: {p.name} is only {p.stat().st_size / 1e9:.2f} GB, probably an unfinished download. "
                     "Run python scripts/download_glimmer.py again.")
        else:
            say(OK, f"{var}: {p.name} ({p.stat().st_size / 1e9:.2f} GB)")

    # ---- llama.cpp
    print("\nllama.cpp")
    exe = find_llama()
    if not exe:
        say(BAD, f"llama-server not found in {ROOT / 'llama.cpp'} or on PATH.")
    else:
        say(OK, f"found {exe}")
        if os.name == "nt" or list(exe.parent.glob("*.dll")):
            check_gpu_dlls(exe.parent)
        for args in (["--version"], ["--list-devices"]):
            try:
                r = subprocess.run([str(exe), *args], capture_output=True, text=True, timeout=60)
                out = (r.stdout + r.stderr).strip().splitlines()
                keep = [l for l in out if any(w in l.lower() for w in ("version", "build", "cuda", "vulkan", "device", "error"))]
                for line in (keep or out)[-6:]:
                    print(f"          {line.strip()[:150]}")
                if args == ["--list-devices"]:
                    gpu = any("cuda" in l.lower() or "vulkan" in l.lower() for l in out)
                    say(OK if gpu else WARN, "GPU visible to llama.cpp" if gpu else
                        "No CUDA/Vulkan device listed: llama.cpp will run on the CPU only (slow).")
            except (OSError, subprocess.TimeoutExpired) as e:
                say(BAD, f"Couldn't run {exe.name} {' '.join(args)}: {e}")

    # ---- is the Glimmer server answering?
    print("\nGlimmer server")
    from fieldwise.providers.glimmer import GlimmerClient
    ok, why = GlimmerClient().health_detail()
    say(OK if ok else BAD, f"{settings.glimmer_base_url}: {why}")

    # ---- guides
    print("\nExtension guides")
    from fieldwise.retrieval import GuideIndex
    n = len(GuideIndex.load(settings.index_path))
    if n:
        say(OK, f"{n} guide passages indexed")
    else:
        say(BAD, "No guide index. Copy data/guides/sources.example.json to sources.json, "
                 "then run python scripts/ingest_guides.py")

    print("\n" + ("All good. Start the app: uvicorn app.server:app --host 0.0.0.0 --port 8000" if not problems
                  else f"{problems} thing(s) to fix above."))


if __name__ == "__main__":
    main()
