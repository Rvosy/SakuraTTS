# Prepare this repository's development environment without changing global Python.
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw 'This script prepares Windows development only.'
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'
}
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $environmentPython)) {
        & uv --offline venv --python 3.11 .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create Python 3.11 environment.' }
    }
    & $environmentPython -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) else 1)'
    if ($LASTEXITCODE -ne 0) { throw 'Existing .venv is not Python 3.11; it has been left unchanged.' }

    # cu128 supports the RTX 50 series. PyTorch is for conversion and validation.
    & uv --offline pip install --python $environmentPython 'torch==2.7.1+cu128' 'torchaudio==2.7.1+cu128' --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'CUDA development dependencies could not be installed.' }
    & uv --offline pip install --python $environmentPython --requirement requirements/windows-dev.txt --editable '.[japanese,nvidia,dev]'
    if ($LASTEXITCODE -ne 0) { throw 'SakuraTTS dependencies could not be installed.' }
    & uv --offline pip check --python $environmentPython
    if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed.' }
    & $environmentPython -m sakuratts doctor --japanese --cuda
    if ($LASTEXITCODE -ne 0) { throw 'Environment checks failed; see the diagnostic output.' }
    Write-Host 'Development environment checks passed. For independent Windows synthesis and model preparation, see docs/setup-windows-nvidia.md.'
} finally {
    Pop-Location
}
