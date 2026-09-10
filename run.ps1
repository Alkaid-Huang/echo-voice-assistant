# One-click launcher: prefer the project venv, fall back to system python
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)

$root = $PSScriptRoot
Set-Location $root
$candidates = @(
    (Join-Path $root '.venv\Scripts\python.exe'),
    (Join-Path $root '..\Live2D-anget\.venv\Scripts\python.exe')
)

foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate) {
        & $candidate (Join-Path $root 'main.py') @Args
        exit $LASTEXITCODE
    }
}

& python (Join-Path $root 'main.py') @Args
