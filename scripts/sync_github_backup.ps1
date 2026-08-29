param(
    [switch]$Commit,
    [switch]$Push
)

$ErrorActionPreference = "Stop"
$RepoPath = Split-Path -Parent $PSScriptRoot
$PythonPath = Join-Path $RepoPath ".venv\Scripts\python.exe"
if (-not $env:GBM_BACKUP_KEY) {
    throw "Define GBM_BACKUP_KEY en tu sesión antes de crear el respaldo."
}
$Arguments = @(
    (Join-Path $PSScriptRoot "github_backup.py"),
    "--repo", $RepoPath,
    "--database", (Join-Path $RepoPath "data\portfolio.db"),
    "--stage"
)
if ($Commit) { $Arguments += "--commit" }
if ($Push) { $Arguments += "--push" }
& $PythonPath @Arguments
