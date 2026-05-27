# Launch GliaAnalysis using the project's own virtual environment.
# Run with:  .\run.ps1   (needs script execution enabled -- see note below).
# If you get "running scripts is disabled on this system", either run
#     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
# once, or just use run.bat instead (it is not affected by execution policy).
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$py   = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $py)) {
    Write-Error "No virtual environment at $py. Create it first: py -3.13 -m venv .venv; & '$py' -m pip install -r requirements-lock.txt"
    exit 1
}

& $py (Join-Path $root 'app.py') @args
