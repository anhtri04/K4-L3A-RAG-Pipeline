# Forward all arguments to Task 2's football crawler.
# Try an existing environment, then installed Python, then Codex's bundled runtime.
$projectRoot = $PSScriptRoot
$candidates = @(
    @{ Executable = (Join-Path $projectRoot '.venv\Scripts\python.exe'); Prefix = @() }
)
foreach ($commandName in @('python', 'py')) {
    $command = Get-Command $commandName -CommandType Application -ErrorAction SilentlyContinue
    if ($command -and $command.Source -notlike '*\Microsoft\WindowsApps\*') {
        $prefix = @()
        if ($commandName -eq 'py') { $prefix = @('-3') }
        $candidates += @{ Executable = $command.Source; Prefix = $prefix }
    }
}
if ($env:USERPROFILE) {
    $candidates += @{
        Executable = (Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe')
        Prefix = @()
    }
}

foreach ($candidate in $candidates) {
    if (-not (Test-Path -LiteralPath $candidate.Executable -PathType Leaf)) { continue }
    $prefix = $candidate.Prefix
    try {
        & $candidate.Executable @prefix -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) { continue }
    } catch { continue }

    Push-Location -LiteralPath $projectRoot
    try {
        & $candidate.Executable @prefix -m src.task2_crawl_news @args
        $crawlerExitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    exit $crawlerExitCode
}

Write-Error 'Python 3.10+ was not found. Install Python, then run: python -m src.task2_crawl_news'
exit 1
