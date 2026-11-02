# Serve Muse Glimmer locally with llama.cpp on Windows (PowerShell). Same behaviour as start_glimmer.sh.
#
#   Put GLIMMER_API_KEY, GLIMMER_GGUF, GLIMMER_MMPROJ and GLIMMER_NGL in .env (see .env.example), then:
#   .\scripts\start_glimmer.ps1
#   To try another value once:  $env:GLIMMER_NGL = "12"; .\scripts\start_glimmer.ps1
#
# If PowerShell blocks the script:  powershell -ExecutionPolicy Bypass -File .\scripts\start_glimmer.ps1
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root     # relative model paths are relative to the project folder

# Pick up GLIMMER_* settings from .env: a later line beats an earlier one,
# and anything already set in this window beats .env.
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    $fromFile = @{}
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^\s*(?:export\s+)?(GLIMMER_[A-Z_]+)\s*=\s*(.*)$') {
            $fromFile[$Matches[1]] = ($Matches[2] -replace '\s+#.*$', '').Trim().Trim('"')
        }
    }
    foreach ($k in $fromFile.Keys) {
        if ($fromFile[$k] -and -not [Environment]::GetEnvironmentVariable($k)) { Set-Item -Path "Env:$k" -Value $fromFile[$k] }
    }
}

if (-not $env:GLIMMER_API_KEY) { throw "Set GLIMMER_API_KEY in .env (a password you make up; NOT your Meta key)" }
$ctx = if ($env:GLIMMER_CTX) { $env:GLIMMER_CTX } else { "8192" }
$ngl = if ($env:GLIMMER_NGL) { $env:GLIMMER_NGL } else { "99" }

# Find llama.cpp: GLIMMER_LLAMA_DIR, then the project's llama.cpp\ folder, then PATH.
# Newer builds may ship llama.exe with a `serve` subcommand; most ship llama-server.exe.
# "Extract All" often adds a subfolder, so look up to two levels down.
$dirs = @($env:GLIMMER_LLAMA_DIR, (Join-Path $root "llama.cpp")) | Where-Object { $_ -and (Test-Path $_) }
$exe = $null; $pre = @()
foreach ($d in $dirs) {
    $hit = Get-ChildItem -Path $d -Filter "llama-server.exe" -Recurse -Depth 2 -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($hit) { $exe = $hit.FullName; break }
    $hit = Get-ChildItem -Path $d -Filter "llama.exe" -Recurse -Depth 2 -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($hit) { $exe = $hit.FullName; $pre = @("serve"); break }
}
if (-not $exe) {
    if (Get-Command llama-server -ErrorAction SilentlyContinue) { $exe = "llama-server" }
    elseif (Get-Command llama -ErrorAction SilentlyContinue) { $exe = "llama"; $pre = @("serve") }
    else { throw "llama.cpp not found. Unzip the Windows CUDA release (and its cudart zip) into $(Join-Path $root 'llama.cpp')" }
}
Write-Host "Using $exe"
if (Test-Path $exe) {
    $exeDir = Split-Path -Parent $exe
    if (-not (Get-ChildItem -Path $exeDir -Filter "cudart64_*.dll" -ErrorAction SilentlyContinue)) {
        Write-Warning ("No cudart64_*.dll next to llama-server.exe in $exeDir. Unzip the cudart-llama-bin-win-cuda-*.zip " +
                       "into that same folder, or the GPU won't be used.")
    }
}

if ($env:GLIMMER_GGUF) {
    if (-not (Test-Path $env:GLIMMER_GGUF)) { throw "No file at GLIMMER_GGUF=$($env:GLIMMER_GGUF)" }
    $model = @("-m", $env:GLIMMER_GGUF)
    if ($env:GLIMMER_MMPROJ) {
        if (-not (Test-Path $env:GLIMMER_MMPROJ)) { throw "No file at GLIMMER_MMPROJ=$($env:GLIMMER_MMPROJ)" }
        $model += @("--mmproj", $env:GLIMMER_MMPROJ)
    } else {
        Write-Warning "GLIMMER_MMPROJ not set: text only. Photo diagnosis needs the vision projector."
    }
} else {
    Write-Warning "GLIMMER_GGUF not set in .env: downloading Meta's official Q4 build (~18.5 GB, needs ~24 GB of memory). Press Ctrl+C to stop."
    $model = @("-hf", "meta-models/Muse-Glimmer-30B-GGUF", "--hf-file", "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf")
}

# Sampling from the model card (temp 1.0, top_p 0.95, top_k 64). Never temperature 0: it loops.
# reasoning_strength is sent with every request by fieldwise/providers/glimmer.py, so no
# --chat-template-kwargs here (its JSON quoting differs between PowerShell versions).
Write-Host "Loading the model. It's ready when this window says the server is listening on http://127.0.0.1:8080."
Write-Host "Leave this window open while you use Fieldwise. Ctrl+C stops it."
# llama.cpp logs to stderr; some PowerShell hosts treat that as an error, so don't stop on it.
$ErrorActionPreference = "Continue"
$nglArgs = if ($ngl -eq "auto") { @() } else { @("-ngl", $ngl) }   # auto: llama.cpp picks the GPU layers
# Optional flags are only passed if this llama.cpp build lists them in --help (flags get renamed between releases).
$helpText = (& $exe @pre --help 2>&1 | Out-String)
function Test-Flag([string]$flag) {
    if (-not $helpText) { return $true }
    return $helpText -match ('(?<![\w-])' + [regex]::Escape($flag) + '(?![\w-])')
}
if ($env:GLIMMER_NO_MMAP -eq "1") {
    if (Test-Flag "--no-mmap") { $nglArgs += @("--no-mmap") }
    else { Write-Warning "This llama.cpp build has no --no-mmap option, so it's left out (model is read from disk on demand; same results)." }
}
if ($env:GLIMMER_REASONING_BUDGET) {
    if (Test-Flag "--reasoning-budget") { $nglArgs += @("--reasoning-budget", $env:GLIMMER_REASONING_BUDGET) }
    else { Write-Warning "This llama.cpp build has no --reasoning-budget option, so Glimmer's thinking isn't capped." }
}
& $exe @pre @model -a muse-glimmer @nglArgs -c $ctx -np 1 `
    --temp 1.0 --top-p 0.95 --top-k 64 `
    --host 127.0.0.1 --port 8080 --api-key $env:GLIMMER_API_KEY
Write-Host ""
Write-Warning "llama.cpp stopped (exit code $LASTEXITCODE). The lines above say why; paste the last 20 into the chat."
