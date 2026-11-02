"""Run the Glimmer (on-device) half of the Fieldwise benchmark on a free Kaggle GPU notebook.

One command per notebook cell (see fieldwise_glimmer_kaggle.ipynb):

    python kaggle_glimmer.py setup          # copy project, install, build llama.cpp (CUDA), download Meta's Q4 GGUF
    python kaggle_glimmer.py serve          # start llama-server in the background and wait until it's ready
    python kaggle_glimmer.py smoke          # 3 questions + 3 photos; checks confidences come from logprobs
    python kaggle_glimmer.py bench qa       # all questions  (resumable: re-run to continue)
    python kaggle_glimmer.py bench images   # all photos     (resumable)
    python kaggle_glimmer.py pack           # -> /kaggle/working/glimmer_results.zip to download
    python kaggle_glimmer.py demo [IDs]     # live clip for the video: the app pipeline with Glimmer answering
                                            # (after setup + serve; IDs from bench/demo_examples.py)
    python kaggle_glimmer.py ui             # the real Fieldwise app with live Glimmer, at a temporary public link
    python kaggle_glimmer.py ui --spark     # ...with live Muse Spark + SAM 3.1 too (key from Kaggle Secrets)
    python kaggle_glimmer.py stop           # take the app and its link down

The model is Meta's official Muse Glimmer 30B Q4_K_M GGUF plus its vision projector, served by
llama.cpp with the same launcher and settings as the hub (scripts/start_glimmer.py). Only the
hardware differs, so record it in the write-up: the GPU model comes from nvidia-smi below.

Notebook settings: Accelerator = GPU (T4 x2 if offered), Internet = On.
"""
from __future__ import annotations

import glob
import json
import os
import secrets
import shutil
import subprocess
import sys
import re
import time
import urllib.request
import zipfile
from pathlib import Path

SCRIPT_VERSION = "2026-11-01c"     # printed on every run, so an old copy in the Kaggle dataset is easy to spot
STEPS = ("setup", "serve", "smoke", "bench", "pack", "demo", "ui", "stop")
INPUT = Path(os.environ.get("FW_KAGGLE_INPUT", "/kaggle/input"))
WORKING = Path(os.environ.get("FW_KAGGLE_WORKING", "/kaggle/working"))
SCRATCH = Path(os.environ.get("FW_KAGGLE_SCRATCH", "/tmp"))      # big, not saved as notebook output
TEST_MODE = os.environ.get("FW_KAGGLE_TEST") == "1"               # only for testing this script with a stub

WORK = WORKING / "fieldwise"
STATE = WORKING / "glimmer_state.json"
LLAMA_SRC = SCRATCH / "llama.cpp"
LLAMA_BIN = LLAMA_SRC / "build" / "bin"
MODEL_DIR = SCRATCH / "glimmer-model"
SERVER_LOG = SCRATCH / "glimmer_server.log"
REPO = "meta-models/Muse-Glimmer-30B-GGUF"
PORT = 8080


def sh(cmd: list[str] | str, **kw) -> None:
    print("$", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, shell=isinstance(cmd, str), **kw)


def load_state() -> dict:
    if not STATE.exists():
        sys.exit("Run `setup` first.")
    return json.loads(STATE.read_text())


def child_env(state: dict) -> dict:
    """The same GLIMMER_* settings the hub uses, pointed at the Kaggle files."""
    env = dict(os.environ)
    env.update({
        "GLIMMER_API_KEY": state["api_key"], "GLIMMER_LLAMA_DIR": str(LLAMA_BIN),
        "GLIMMER_GGUF": state["gguf"], "GLIMMER_MMPROJ": state["mmproj"], "GLIMMER_NGL": state["ngl"],
        "GLIMMER_CTX": "8192", "GLIMMER_BASE_URL": f"http://127.0.0.1:{PORT}/v1",
        "FIELDWISE_DOTENV": "0", "PYTHONUNBUFFERED": "1",
    })
    return env


# ------------------------------------------------------------------ setup
def find_project() -> Path:
    hits = sorted(INPUT.glob("**/fieldwise/config.py"))
    if not hits:
        sys.exit(f"Can't find the Fieldwise dataset under {INPUT}. Add your private dataset "
                 "(made with bench/pack_for_kaggle.py) via 'Add Input'.")
    return hits[0].parent.parent


def gpus() -> list[tuple[str, int]]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [(n.strip(), int(m)) for n, m in (line.rsplit(",", 1) for line in out.strip().splitlines() if line)]


def pick_model_files(files: list[str]) -> tuple[str, str | None]:
    ggufs = [f for f in files if f.endswith(".gguf")]
    main = [f for f in ggufs if "q4_k_m" in f.lower() and "mmproj" not in f.lower()]
    proj = sorted((f for f in ggufs if "mmproj" in f.lower()),
                  key=lambda f: (0 if "f16" in f.lower() and "bf16" not in f.lower() else 1, f))
    if not main:
        sys.exit(f"No Q4_K_M GGUF in {REPO}. Files there: {ggufs}")
    return main[0], (proj[0] if proj else None)


LIBCUDA_PATTERNS = (
    "/usr/local/cuda*/lib64/stubs/libcuda.so", "/usr/local/cuda*/targets/*/lib/stubs/libcuda.so",
    "/usr/lib/x86_64-linux-gnu/libcuda.so", "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
    "/usr/local/nvidia/lib64/libcuda.so*", "/usr/lib64-nvidia/libcuda.so*",
)


def find_libcuda(patterns=LIBCUDA_PATTERNS) -> str | None:
    """The CUDA driver library ggml-cuda links against. Kaggle's image has the toolkit but CMake
    doesn't always find libcuda, which fails with 'Target ggml-cuda links to CUDA::cuda_driver'."""
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def build_llama() -> None:
    if not LLAMA_SRC.exists():
        sh(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(LLAMA_SRC)])
    build = LLAMA_SRC / "build"
    # 60 = P100, 75 = T4: building only these keeps the compile to minutes, not an hour.
    base = ["cmake", "-S", str(LLAMA_SRC), "-B", str(build), "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=60;75",
            "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"]
    lib = find_libcuda()
    print(f"CUDA driver library: {lib or 'not found'}")
    driver = [f"-DCUDA_cuda_driver_LIBRARY={lib}"] if lib else []
    # Attempt 1: point CMake at libcuda. Attempt 2: also turn off CUDA virtual memory (VMM), the
    # feature that needs libcuda at link time; llama.cpp then uses its older memory pool.
    attempts = [base + driver, base + driver + ["-DGGML_CUDA_NO_VMM=ON"]]
    for i, args in enumerate(attempts, start=1):
        shutil.rmtree(build, ignore_errors=True)        # a failed configure leaves a bad cache behind
        try:
            sh(args)
            break
        except subprocess.CalledProcessError:
            if i == len(attempts):
                sys.exit("llama.cpp wouldn't configure with CUDA. Send the CMake error above to Claude.")
            print(f"CMake attempt {i} failed; retrying without CUDA VMM.", flush=True)
    sh(["cmake", "--build", str(build), "--config", "Release", "-j", str(os.cpu_count() or 4), "--target", "llama-server"])


def setup() -> None:
    src = find_project()
    print(f"Project found at {src}")
    keep = WORK / "bench" / "results"
    shutil.copytree(src, WORK, dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("results") if keep.exists() else None)
    (WORK / "bench" / "results").mkdir(parents=True, exist_ok=True)

    if not TEST_MODE:
        sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(WORK / "requirements.txt")])

    g = gpus()
    print("GPUs:", g or "none found")
    if not g and not TEST_MODE:
        sys.exit("No GPU. In the notebook's right panel set Accelerator = GPU (T4 x2 if offered), then restart.")
    total_mib = sum(m for _, m in g)

    if not (LLAMA_BIN / "llama-server").exists():
        build_llama()
    sh([str(LLAMA_BIN / "llama-server"), "--version"])

    free_gb = shutil.disk_usage(SCRATCH).free / 1e9
    print(f"Free space in {SCRATCH}: {free_gb:.0f} GB")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if TEST_MODE:   # no network in tests: stand-in files
        main_file, proj_file = "test-Q4_K_M.gguf", "mmproj-test-f16.gguf"
        for f in (main_file, proj_file):
            (MODEL_DIR / f).write_bytes(b"x")
        gguf, mmproj = str(MODEL_DIR / main_file), str(MODEL_DIR / proj_file)
    else:
        from huggingface_hub import hf_hub_download, list_repo_files
        main_file, proj_file = pick_model_files(list_repo_files(REPO))
        if free_gb < 22 and not (MODEL_DIR / main_file).exists():
            sys.exit("Not enough disk for the 18 GB model download.")
        print(f"Downloading {main_file} and {proj_file} from {REPO} ...", flush=True)
        gguf = hf_hub_download(REPO, main_file, local_dir=MODEL_DIR)
        mmproj = hf_hub_download(REPO, proj_file, local_dir=MODEL_DIR) if proj_file else ""

    sh([sys.executable, "scripts/ingest_guides.py"], cwd=WORK, env={**os.environ, "FIELDWISE_DOTENV": "0"})

    old = json.loads(STATE.read_text()) if STATE.exists() else {}
    running = {k: old[k] for k in ("pid", "app_pid", "app_mode", "tunnel_pid") if k in old}   # keep track of live processes
    state = {**running, "api_key": old.get("api_key") or secrets.token_hex(16), "gguf": gguf, "mmproj": mmproj,
             # Two T4s (2 x 15 GiB) hold the whole model; one 16 GB card can't, so let llama.cpp fit it.
             "ngl": "99" if total_mib >= 20000 else "auto", "gpus": g, "repo": REPO, "files": [main_file, proj_file]}
    STATE.write_text(json.dumps(state, indent=2))
    print(f"\nSetup done. GPU layers: {state['ngl']} (total GPU memory {total_mib} MiB). Next: serve")


# ------------------------------------------------------------------ serve
def healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3) as r:
            return r.status == 200
    except OSError:
        return False


def serve() -> None:
    state = load_state()
    if healthy():
        print("Glimmer is already running.")
        return
    log = SERVER_LOG.open("w")
    proc = subprocess.Popen([sys.executable, str(WORK / "scripts" / "start_glimmer.py")], cwd=WORK,
                            env=child_env(state), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state["pid"] = proc.pid
    STATE.write_text(json.dumps(state, indent=2))
    t0 = time.time()
    while not healthy():
        if proc.poll() is not None:
            print(SERVER_LOG.read_text()[-4000:])
            sys.exit(f"llama-server stopped (exit {proc.returncode}). The log above says why.")
        if time.time() - t0 > 1800:
            sys.exit("Still not ready after 30 min; see " + str(SERVER_LOG))
        time.sleep(10)
        print(f"  loading... {time.time() - t0:.0f}s", flush=True)
    print(f"Glimmer is ready after {time.time() - t0:.0f}s.")
    for line in SERVER_LOG.read_text(errors="replace").splitlines():
        if any(k in line for k in ("offloaded", "CUDA", "buffer size", "build", "listening")):
            print("  " + line.strip()[:160])


# ------------------------------------------------------------------ runs
def run_bench(task: str, extra: list[str]) -> None:
    state = load_state()
    if not healthy():
        sys.exit("Glimmer isn't running. Run the `serve` cell first (it stops when the session restarts).")
    cmd = [sys.executable, "bench/run_bench.py", "--task", task, "--setup", "local", *extra]
    if TEST_MODE:
        cmd += ["--allow-stub", "--no-strict"]
    sh(cmd, cwd=WORK, env=child_env(state))


def smoke() -> None:
    state = load_state()
    for task in ("qa", "images"):
        run_bench(task, ["--tag", "smoke", "--limit", "3"])
        sh([sys.executable, "bench/peek.py", f"{task}_local_smoke"], cwd=WORK, env=child_env(state))
    print("\nCheck above: each confidence should say (logprobs), and nothing should say ERR.\n"
          "If so, run the full benchmark cells. Smoke files are left out of the results zip.")


def demo(extra: list[str]) -> None:
    """Live proof for the video: demo questions and photos through the real app pipeline (bench/live_demo.py)."""
    state = load_state()
    if not healthy():
        sys.exit("Glimmer isn't running. Run the `serve` cell first (it stops when the session restarts).")
    env = child_env(state)
    env.update({"FIELDWISE_FORCE_OFFLINE": "1", "FIELDWISE_GLIMMER_REPLAY": "0", "MODEL_API_KEY": ""})
    # No .env on Kaggle: use the app's thresholds (chosen on the dev split) unless set in the notebook.
    env.setdefault("FIELDWISE_QA_THRESHOLD", "0.90")
    env.setdefault("FIELDWISE_DX_THRESHOLD", "0.05")
    cmd = [sys.executable, "bench/live_demo.py", *extra] + (["--allow-stub"] if TEST_MODE else [])
    sh(cmd, cwd=WORK, env=env)


APP_PORT = 8000
APP_LOG = SCRATCH / "fieldwise_app.log"
TUNNEL_LOG = SCRATCH / "tunnel.log"
CLOUDFLARED = SCRATCH / "cloudflared"
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"


def app_up() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{APP_PORT}/api/status", timeout=3) as r:
            return r.status == 200
    except OSError:
        return False


def meta_key() -> str | None:
    """The Meta key from the environment or Kaggle Secrets (Add-ons > Secrets, label MODEL_API_KEY). Never printed."""
    if os.environ.get("MODEL_API_KEY"):
        return os.environ["MODEL_API_KEY"]
    try:
        from kaggle_secrets import UserSecretsClient          # only exists on Kaggle
        return UserSecretsClient().get_secret("MODEL_API_KEY") or None
    except Exception:  # noqa: BLE001  (not on Kaggle, secret not attached, ...)
        return None


def find_procs(*needles: str) -> list[int]:
    """PIDs whose command line contains every needle (read from /proc; never this script or its shell)."""
    me = {os.getpid(), os.getppid()}
    out = []
    for d in Path("/proc").glob("[0-9]*"):
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        pid = int(d.name)
        if pid not in me and pid > 1 and all(n in cmd for n in needles):
            out.append(pid)
    return out


def terminate(pids: list[int]) -> None:
    """SIGTERM, wait up to 5 s, then SIGKILL. Only real child PIDs: never 0, 1, this script or its shell."""
    me = {os.getpid(), os.getppid()}
    pids = [p for p in set(pids) if isinstance(p, int) and p > 1 and p not in me]
    for p in pids:
        try:
            os.kill(p, 15)
        except OSError:
            pass
    for _ in range(50):
        alive = []
        for p in pids:
            try:
                os.kill(p, 0)
                alive.append(p)
            except OSError:
                pass
        if not alive:
            return
        time.sleep(0.1)
    for p in alive:
        try:
            os.kill(p, 9)
        except OSError:
            pass


def stop_app(state: dict) -> None:
    terminate([state.pop("app_pid", 0)] + find_procs("uvicorn", "app.server:app"))
    state.pop("app_mode", None)


def stop_tunnel(state: dict) -> None:
    terminate([state.pop("tunnel_pid", 0)] + find_procs("cloudflared", "tunnel"))


def ui(extra: list[str] | None = None) -> None:
    """The real app (same code as the laptop hub) with live Glimmer, reachable through a Cloudflare quick tunnel.
    Default: the Meta API is never called (hub forced offline, low-confidence items queued).
    --spark: live Muse Spark escalation and SAM 3.1 with the key from Kaggle Secrets; the app's account guard
    pauses Spark after 2 content-filter refusals and stops all Meta calls if the key is refused."""
    spark = "--spark" in (extra or [])
    state = load_state()
    if not healthy():
        sys.exit("Glimmer isn't running. Run the `serve` cell first (it stops when the session restarts).")
    if not (WORK / "app" / "server.py").exists():
        sys.exit("The dataset has no app/ folder. Re-pack on the laptop with the current bench/pack_for_kaggle.py "
                 "(it now includes app/), upload a new dataset version, switch the notebook to it and re-run setup.")
    env = child_env(state)
    if spark:
        key = meta_key()
        if not key:
            sys.exit("No Meta key found. In the notebook editor: Add-ons > Secrets > Add a new secret, label "
                     "MODEL_API_KEY, paste the key, and tick it for this notebook. Then run this cell again.")
        env.update({"FIELDWISE_FORCE_OFFLINE": "0", "FIELDWISE_GLIMMER_REPLAY": "0", "FIELDWISE_CACHE_ONLY": "0",
                    "MODEL_API_KEY": key, "FIELDWISE_SPARK_MAX_REFUSALS": "2"})
        print(f"Meta key found ({len(key)} characters{', starts LLM_' if key.startswith('LLM_') else ''}). "
              "Spark and SAM calls are live and billed.")
    else:
        env.update({"FIELDWISE_FORCE_OFFLINE": "1", "FIELDWISE_GLIMMER_REPLAY": "0", "FIELDWISE_CACHE_ONLY": "1",
                    "MODEL_API_KEY": ""})
    env.setdefault("FIELDWISE_QA_THRESHOLD", "0.90")      # the app's thresholds, chosen on the dev split
    env.setdefault("FIELDWISE_DX_THRESHOLD", "0.05")
    mode = "spark" if spark else "offline"
    if app_up() and state.get("app_mode") != mode:         # switching modes: restart the app with the new settings
        print(f"Restarting the app in {mode} mode ...", flush=True)
        stop_app(state)
        for _ in range(20):
            if not app_up():
                break
            time.sleep(0.5)
    if not app_up():
        log = APP_LOG.open("w")
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.server:app", "--host", "127.0.0.1",
                                 "--port", str(APP_PORT)], cwd=WORK, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        state["app_pid"], state["app_mode"] = proc.pid, mode
        for _ in range(60):
            if app_up():
                break
            if proc.poll() is not None:
                print(APP_LOG.read_text()[-3000:])
                sys.exit("The Fieldwise app stopped; the log above says why.")
            time.sleep(1)
        else:
            sys.exit(f"The app didn't start within 60 s; see {APP_LOG}")
    print(f"Fieldwise app is running on the Kaggle machine (port {APP_PORT}), "
          + ("with live Muse Spark and SAM 3.1." if spark else "with the Meta API off."))
    if os.environ.get("FW_KAGGLE_NO_TUNNEL") == "1":           # tests
        STATE.write_text(json.dumps(state, indent=2))
        return

    if not CLOUDFLARED.exists():
        print("Downloading cloudflared (Cloudflare's tunnel client) ...", flush=True)
        urllib.request.urlretrieve(CLOUDFLARED_URL, CLOUDFLARED)
        CLOUDFLARED.chmod(0o755)
    stop_tunnel(state)                                              # one link at a time
    log = TUNNEL_LOG.open("w")
    tun = subprocess.Popen([str(CLOUDFLARED), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{APP_PORT}"],
                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    state["tunnel_pid"] = tun.pid
    STATE.write_text(json.dumps(state, indent=2))
    url = None
    for _ in range(60):
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", TUNNEL_LOG.read_text(errors="replace"))
        if m:
            url = m.group(0)
            break
        if tun.poll() is not None:
            break
        time.sleep(1)
    if not url:
        print(TUNNEL_LOG.read_text(errors="replace")[-2000:])
        sys.exit("No tunnel link appeared. Check that the notebook's Internet setting is On; the log above says more.")
    print(f"\n  Open on your phone or laptop:  {url}\n")
    print("Notes:\n"
          "  - Glimmer runs live on this Kaggle GPU (about 20 s per question, 13 s per photo on 2x T4).\n"
          + ("  - Low-confidence items go to Muse Spark live (redacted). After 2 content-filter refusals the app pauses\n"
             "    Spark until you run the stop cell and then this one again; if Meta refuses the key, all Meta calls stop.\n"
             if spark else
             "  - The Meta API is off here (forced offline), so low-confidence items are queued, not sent to Spark.\n")
          + 
          "  - Anyone with the link can use the app while it's up. Don't share it; run the `stop` cell when done.\n"
          "  - Label any recording 'Live on a Kaggle T4 GPU', not the laptop hub.\n"
          "  - The link may take ~20 s to start working. Check Kaggle's terms allow tunnels before relying on this.")


def stop() -> None:
    state = load_state()
    stop_tunnel(state)
    stop_app(state)
    STATE.write_text(json.dumps(state, indent=2))
    print("App and link stopped. Glimmer keeps running (it stops when the session ends).")


def pack() -> None:
    res = WORK / "bench" / "results"
    out = WORKING / "glimmer_results.zip"
    files = [p for p in sorted(res.glob("*")) if p.is_file() and "smoke" not in p.name
             and ("local" in p.name or p.name.startswith("live_demo"))]
    if not files:
        sys.exit("No local results yet. Run the bench cells first.")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, f"bench/results/{p.name}")
        if SERVER_LOG.exists():   # first lines hold the llama.cpp build and GPU offload details
            z.writestr("bench/results/glimmer_server_log_head.txt",
                       "\n".join(SERVER_LOG.read_text(errors="replace").splitlines()[:400]))
        z.writestr("bench/results/kaggle_hardware.json", json.dumps({"gpus": load_state().get("gpus")}, indent=2))
    print(f"Wrote {out} with {len(files)} result files. Download it from the notebook's Output panel and\n"
          "unzip it inside your laptop's fieldwise folder (it only adds files to bench\\results).")


def main() -> None:
    print(f"kaggle_glimmer.py version {SCRIPT_VERSION} ({Path(__file__).resolve()})", flush=True)
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    step = sys.argv[1]
    if step not in STEPS:
        sys.exit(f"Unknown step {step!r}. This copy knows: {', '.join(STEPS)}.\n"
                 "If a step from the notebook is missing, the Kaggle dataset has an older copy of this script: "
                 "re-pack on the laptop (python bench/pack_for_kaggle.py), upload it as a new dataset version, "
                 "and switch the notebook's input to that version.")
    if step == "setup":
        setup()
    elif step == "serve":
        serve()
    elif step == "smoke":
        smoke()
    elif step == "bench" and len(sys.argv) >= 3 and sys.argv[2] in ("qa", "images"):
        run_bench(sys.argv[2], sys.argv[3:])
    elif step == "pack":
        pack()
    elif step == "demo":
        demo(sys.argv[2:])
    elif step == "ui":
        ui(sys.argv[2:])
    elif step == "stop":
        stop()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
