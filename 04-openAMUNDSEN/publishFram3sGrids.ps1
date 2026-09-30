<# Publish a validated staged collection using native Windows network-drive I/O.
   No source dataset is deleted: replaced files are moved into dated archives.
   The journal and file hashes support rollback after an interrupted publication. #>
param(
    [Parameter(Mandatory=$true)][string]$WorkRoot,
    [Parameter(Mandatory=$true)][string]$DestinationRoot,
    [Parameter(Mandatory=$true)][string]$RecordRoot,
    [switch]$Publish
)
$ErrorActionPreference = 'Stop'
$OutputRoot = Join-Path $WorkRoot 'output'
$Aoi = Join-Path $OutputRoot '01-aoi'
$Manifest = Get-Content -LiteralPath (Join-Path $WorkRoot 'artifact_manifest.json') -Raw | ConvertFrom-Json
$StackManifest = Get-Content -LiteralPath (Join-Path $Aoi 'collection_manifest.json') -Raw | ConvertFrom-Json
if ($StackManifest.Count -ne 75) { throw 'Expected 75 validated stacks' }
foreach ($Resolution in @(50,100,250,500,1000)) {
    $Validation = Get-Content -LiteralPath (Join-Path $WorkRoot "validation_$Resolution.json") -Raw | ConvertFrom-Json
    if ($Validation.stacks -ne 15 -or $Validation.failures.Count -ne 0) { throw "Validation failed: $Resolution" }
}
$Qgis = Get-Content -LiteralPath (Join-Path $Aoi 'qgis_validation.json') -Raw | ConvertFrom-Json
if ($Qgis.invalid_layers.Count -ne 0 -or $Qgis.subregions -ne 90) { throw 'QGIS validation failed' }
$Vectors = Get-Content -LiteralPath (Join-Path $Aoi 'vector_validation.json') -Raw | ConvertFrom-Json
if (-not $Vectors.attributes_preserved -or $Vectors.subregions -ne 90) { throw 'Vector validation failed' }
if (-not $Publish) {
    Write-Output "Ready: $($Manifest.Count) artifacts, 75 stacks, 90 subregions. Use -Publish to archive and copy."
    exit 0
}
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
New-Item -ItemType Directory -Force -Path $RecordRoot | Out-Null
$Journal = Join-Path $RecordRoot "publication_$Stamp.json"
$SourceManifest = Get-Content -LiteralPath (Join-Path $Aoi 'source_manifest.json') -Raw | ConvertFrom-Json
foreach ($Kind in @('dem','lc','provinces','subregions')) {
    $Item = $SourceManifest.$Kind
    $Relative = if ($Item.origin_relative) { $Item.origin_relative } else { $Item.path }
    $Path = Join-Path $DestinationRoot $Relative
    if ((Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLower() -ne $Item.sha256) {
        throw "Source changed since staging: $Path"
    }
}
$Moves = @()
$OldAoi = Join-Path $DestinationRoot '01-aoi'
foreach ($Item in Get-ChildItem -LiteralPath $OldAoi -Force) {
    if ($Item.Name -ne 'archive') {
        $Moves += [pscustomobject]@{source=$Item.FullName; target=(Join-Path $OldAoi "archive\$Stamp\$($Item.Name)")}
    }
}
$ArchivePaths = @('05-dem\rofental','06-srf\rofental',
                  '03-landcover\lc_eusalp\openAMUNDSEN-euregio','03-landcover\lc_eusalp\openAMUNDSEN-rofental')
foreach ($LayerRoot in @('03-landcover','05-dem','06-srf','07-svf')) {
    foreach ($Region in @('euregio','tyrol','north_tyrol','south_tyrol','trentino')) {
        $ArchivePaths += "$LayerRoot\$Region"
    }
    $ArchivePaths += "$LayerRoot\README.txt"
}
foreach ($Relative in $ArchivePaths) {
    $Parts = $Relative -split '\\',2
    $Source = Join-Path $DestinationRoot $Relative
    if (Test-Path -LiteralPath $Source) {
        $Moves += [pscustomobject]@{source=$Source; target=(Join-Path $DestinationRoot "$($Parts[0])\archive\$Stamp\$($Parts[1])")}
    }
}
foreach ($File in $Manifest) {
    $Target = Join-Path $DestinationRoot $File.path
    if (Test-Path -LiteralPath $Target) {
        $CoveredByArchive = $false
        foreach ($Move in $Moves) {
            if ($Target -eq $Move.source -or $Target.StartsWith($Move.source+'\',[StringComparison]::OrdinalIgnoreCase)) {
                $CoveredByArchive = $true
                break
            }
        }
        if (-not $CoveredByArchive) { throw "Unexpected publication collision before archiving: $Target" }
    }
}
$OldFiles = @()
foreach ($Move in $Moves) {
    $Item = Get-Item -LiteralPath $Move.source -Force
    $Files = if ($Item.PSIsContainer) { @(Get-ChildItem -LiteralPath $Move.source -File -Recurse -Force) } else { @($Item) }
    foreach ($File in $Files) {
        $Suffix = $File.FullName.Substring($Move.source.Length)
        $OldFiles += [pscustomobject]@{source=$File.FullName; archive=($Move.target+$Suffix); bytes=$File.Length;
            sha256=(Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLower()}
    }
}
$State = [ordered]@{stamp=$Stamp; status='inventoried'; moves=$Moves; archived_files=$OldFiles; published_files=0}
$State | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Journal -Encoding UTF8
foreach ($Move in $Moves) {
    if (Test-Path -LiteralPath $Move.target) { throw "Archive destination exists: $($Move.target)" }
    New-Item -ItemType Directory -Force -Path (Split-Path $Move.target -Parent) | Out-Null
    Move-Item -LiteralPath $Move.source -Destination $Move.target
}
$State.status='archived'
$State | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Journal -Encoding UTF8
foreach ($File in $OldFiles) {
    if ((Get-FileHash -LiteralPath $File.archive -Algorithm SHA256).Hash.ToLower() -ne $File.sha256) {
        throw "Archive verification failed: $($File.archive)"
    }
}
foreach ($File in $Manifest) {
    $Source = Join-Path $OutputRoot $File.path
    $Target = Join-Path $DestinationRoot $File.path
    if (Test-Path -LiteralPath $Target) { throw "Unexpected publication collision: $Target" }
    New-Item -ItemType Directory -Force -Path (Split-Path $Target -Parent) | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Target
    $Actual = Get-Item -LiteralPath $Target
    if ($Actual.Length -ne $File.bytes -or (Get-FileHash -LiteralPath $Target -Algorithm SHA256).Hash.ToLower() -ne $File.sha256) {
        throw "Publication verification failed: $Target"
    }
    $State.published_files++
    if ($State.published_files % 100 -eq 0) {
        $State.status='publishing'
        $State | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Journal -Encoding UTF8
        Write-Output "Verified $($State.published_files) / $($Manifest.Count) published files"
    }
}
$State.status='complete'
$State | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Journal -Encoding UTF8
$Moves | Export-Csv -LiteralPath (Join-Path $RecordRoot "archive_paths_$Stamp.csv") -NoTypeInformation -Encoding UTF8
Write-Output "Publication complete: $Journal"
