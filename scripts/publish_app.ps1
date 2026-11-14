# Run locally from PowerShell. Never stages datasets or full research directories.
param([switch]$DryRun)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
function Invoke-Git {
    & git @args
    if ($LASTEXITCODE -ne 0) { throw "git $($args[0]) failed (exit $LASTEXITCODE). No force-push or automatic reset was attempted." }
}
$remote = git remote get-url origin
if ($LASTEXITCODE -ne 0 -or $remote -notmatch 'github.com[:/]benbitter/Robust-Zero-Shot(\.git)?$') {
    throw 'Origin differs from the reviewed TerraCLIP repository. Inspect git remote -v before publishing.'
}
$files = @(
    '.gitignore', '.gitattributes', 'readme.md', 'requirements-app.txt',
    'scripts/publish_app.ps1', 'deployment', 'models/clip-vit-base-patch16', 'frontend',
    'backend/__init__.py','backend/main.py','backend/portable.py',
    'backend/final_benchmark_results.py','backend/image_processing.py',
    'backend/inference_service.py','backend/model_registry.py','backend/research.py',
    'backend/schemas.py','backend/v2_results.py',
    'production_20/__init__.py','production_20/common.py','production_20/taxonomy.py',
    'production_20/features.py','production_20/learning.py','production_20/benchmark.py',
    'production_20/service.py','production_20/data.py',
    'production_v2/__init__.py','production_v2/taxonomy.py','production_v2/common.py',
    'src/__init__.py','src/config.py','src/utils.py','src/tta.py',
    'src/e6_corruptions.py','src/e3_runtime.py','src/visual_adapter.py'
)

# Preflight is read-only, including when a previous failed add left staged files.
foreach ($path in $files) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Release path is missing: $path" }
    & git check-ignore --no-index --quiet -- $path
    if ($LASTEXITCODE -eq 0) { throw "Explicit release path is ignored: $path" }
    if ($LASTEXITCODE -ne 1) { throw "Could not check ignore rules for: $path" }
}
$releaseFiles = @(Invoke-Git ls-files --cached --others --exclude-standard -- @files | Sort-Object -Unique)
if (-not $releaseFiles.Count) { throw 'Release file list is empty.' }
foreach ($name in $releaseFiles) {
    if (-not (Test-Path -LiteralPath $name -PathType Leaf)) { throw "Release file is missing: $name" }
    # Pass paths as arguments: PowerShell's CRLF stdin would otherwise become
    # part of the filename read by git check-ignore --stdin on Windows.
    & git check-ignore --no-index --quiet -- $name
    if ($LASTEXITCODE -eq 0) { throw "Ignored release file: $name" }
    if ($LASTEXITCODE -ne 1) { throw "Could not check ignore rules for: $name" }
    if ($name -match '(^|/)(data|EuroSAT_RGB|results|node_modules|\.venv|dist|__pycache__|tests|\.cache)(/|$)' -or
        $name -match '(^|/)\.env($|\.)' -or $name -match '\.(parquet|csv|log|pem|key|pfx|p12)$' -or
        $name -match '(^|/)(credentials|secrets)(\.|/|$)' -or $name -match '(^|/)id_(rsa|ed25519)') {
        throw "Forbidden release file: $name. Nothing has been staged by this run."
    }
}
# Never incorporate unrelated staged work. Matching release files may be
# restaged from the working tree, allowing recovery from the earlier failed add.
$staged = @(Invoke-Git diff --cached --name-only)
$unexpected = @($staged | Where-Object { $_ -notin $releaseFiles })
if ($unexpected.Count) {
    throw "Unrelated staged files found; review them manually first: $($unexpected -join ', ')"
}
$weights = @('models/clip-vit-base-patch16/pytorch_model.bin','deployment/m3.pt','deployment/m4.pt')
foreach ($weight in $weights) {
    if ($weight -notin $releaseFiles) { throw "Required model is missing from release: $weight" }
    $attribute = Invoke-Git check-attr filter -- $weight
    if ($attribute -ne "${weight}: filter: lfs") { throw "Git LFS rule missing for: $weight" }
}
$branch = Invoke-Git branch --show-current
if (-not $branch) { throw 'Detached HEAD: choose a branch before publishing.' }
Write-Host "Validated release: $($releaseFiles.Count) files; $($staged.Count) already staged."
$releaseFiles | ForEach-Object { Write-Host "  $_" }
if ($DryRun) {
    Write-Host 'Dry run complete. No index, Git configuration, commit or remote changes were made.'
    return
}

Invoke-Git lfs install --local
# Stage precisely the preflighted files, not their containing directories.
Invoke-Git add -- @releaseFiles
$names = @(Invoke-Git diff --cached --name-only)
if (@($names | Where-Object { $_ -notin $releaseFiles }).Count) { throw 'Unexpected staged changes; inspect the index before committing.' }
$lfs = git lfs ls-files --name-only
if ($LASTEXITCODE -ne 0) { throw 'Could not verify Git LFS index entries.' }
foreach ($weight in $weights) {
    if ($weight -notin $lfs) { throw "Model weight is not stored with Git LFS: $weight" }
}
Invoke-Git diff --cached --stat
if ($names) { Invoke-Git commit -m 'Publish portable TerraCLIP six-model web application and local setup guide' }
Invoke-Git push -u origin $branch
Write-Host "Published app and Git LFS weights to origin/$branch. Research artifacts were not staged."
