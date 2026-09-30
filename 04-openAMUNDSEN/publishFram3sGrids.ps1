<# Publish final files; archive superseded AOI and raw sources outside active delivery.
   Existing terrain grids must match staged hashes and remain in place. #>
param(
    [Parameter(Mandatory=$true)][string]$WorkRoot,
    [Parameter(Mandatory=$true)][string]$DestinationRoot,
    [Parameter(Mandatory=$true)][string]$RecordRoot,
    [switch]$Publish
)
$ErrorActionPreference = 'Stop'
$WorkRoot = (Get-Item -LiteralPath $WorkRoot).FullName
$DestinationRoot = (Get-Item -LiteralPath $DestinationRoot).FullName
$OutputRoot = Join-Path $WorkRoot 'output'
function Read-Record($Name) { Get-Content -LiteralPath (Join-Path $WorkRoot $Name) -Raw | ConvertFrom-Json }
function Hash($Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLower() }
$Manifest = Read-Record 'artifact_manifest.json'
$Specs = Read-Record 'grid_specifications.json'
$Stacks = Read-Record 'collection_manifest.json'
if ($Specs.Count -ne 75 -or $Stacks.Count -ne 75) { throw 'Expected 75 validated stacks' }
$Expected = @{}
foreach ($Name in @('README.txt','aoi.gpkg','aoi_overview.qgz','aoi_overview.png','aoi_overview.pdf')) {
    $Expected["01-aoi/$Name"] = $true
}
$Roots = @{roi='01-aoi'; lc='03-landcover'; dem='05-dem'; srf='06-srf'; svf='07-svf'}
foreach ($Spec in $Specs) {
    $Variant = "$($Spec.region)/buffer_$('{0:D5}' -f [int]$Spec.buffer_m)m"
    $Domain = "$($Spec.region)_b$('{0:D5}' -f [int]$Spec.buffer_m)"
    $Expected["01-aoi/$Variant/$Domain.zip"] = $true
    foreach ($Kind in $Roots.Keys) {
        foreach ($Extension in @('tif','asc','prj')) {
            $Expected["$($Roots[$Kind])/$Variant/$($Spec.resolution_m)m/${Kind}_${Domain}_$($Spec.resolution_m).$Extension"] = $true
        }
    }
}
if ($Expected.Count -ne 1145 -or $Manifest.Count -ne 1145) { throw 'Expected exactly 1145 delivery files' }
$Seen = @{}
foreach ($File in $Manifest) {
    $Relative = $File.path.Replace('\','/')
    if (-not $Expected.ContainsKey($Relative) -or $Seen.ContainsKey($Relative)) { throw "Unexpected/duplicate delivery path: $Relative" }
    $Seen[$Relative] = $true
    $Source = Join-Path $OutputRoot $Relative
    if (-not (Test-Path -LiteralPath $Source) -or (Get-Item -LiteralPath $Source).Length -ne $File.bytes) {
        throw "Missing/changed staged file: $Relative"
    }
}
$Delivery = Read-Record 'delivery_validation.json'
if ($Delivery.files -ne 1145 -or $Delivery.failures.Count -ne 0) { throw 'Whitelist validation failed' }
foreach ($Resolution in @(50,100,250,500,1000)) {
    $Validation = Read-Record "validation_$Resolution.json"
    if ($Validation.stacks -ne 15 -or $Validation.failures.Count -ne 0) { throw "Validation failed: $Resolution" }
}
$Qgis = Read-Record 'qgis_validation.json'
if ($Qgis.invalid_layers.Count -ne 0 -or $Qgis.subregions -ne 90 -or $Qgis.roi_rasters -ne 75) { throw 'QGIS validation failed' }
$Vectors = Read-Record 'vector_validation.json'
if (-not $Vectors.attributes_preserved -or $Vectors.subregions -ne 90) { throw 'Vector validation failed' }
$Core = Read-Record 'roi_context_validation.json'
if ($Core.failures.Count -ne 0 -or $Core.comparisons -ne 50) { throw 'ROI context validation failed' }
$Stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$ArchiveRoot = Join-Path (Split-Path $DestinationRoot -Parent) "90-archive\grid_collection_cleanup\$Stamp"
$Moves = @()
$Paths = @('01-aoi', '03-landcover\forest','03-landcover\glacier','03-landcover\lc_eusalp')
foreach ($Root in @('03-landcover','05-dem','06-srf','07-svf')) {
    $Paths += "$Root\archive", "$Root\README.txt"
    if (Test-Path -LiteralPath (Join-Path $DestinationRoot $Root)) {
        $Paths += @(Get-ChildItem -LiteralPath (Join-Path $DestinationRoot $Root) -Recurse -File -Force |
            Where-Object { $_.Name.EndsWith('.aux.xml') -and $_.FullName -notlike '*\archive\*' -and
                $_.FullName -notlike '*\lc_eusalp\*' -and $_.FullName -notlike '*\forest\*' -and $_.FullName -notlike '*\glacier\*' } |
            ForEach-Object { $_.FullName.Substring($DestinationRoot.Length+1) })
    }
}
foreach ($Relative in $Paths) {
    $Source = Join-Path $DestinationRoot $Relative
    if (Test-Path -LiteralPath $Source) {
        $Moves += [pscustomobject]@{source=$Source; target=(Join-Path $ArchiveRoot $Relative)}
    }
}
function Is-Archived($Path) {
    foreach ($Move in $Moves) {
        if ($Path -eq $Move.source -or $Path.StartsWith($Move.source+'\',[StringComparison]::OrdinalIgnoreCase)) { return $true }
    }
    return $false
}
# Reject unknown active files before any mutation; do not silently archive unrelated material.
foreach ($Root in $Roots.Values) {
    $Folder = Join-Path $DestinationRoot $Root
    if (Test-Path -LiteralPath $Folder) {
        foreach ($File in Get-ChildItem -LiteralPath $Folder -Recurse -File -Force) {
            $Relative = $File.FullName.Substring($DestinationRoot.Length+1).Replace('\','/')
            if (-not (Is-Archived $File.FullName) -and -not $Expected.ContainsKey($Relative)) {
                throw "Unexpected active file before archiving: $Relative"
            }
        }
    }
}
if (-not $Publish) {
    Write-Output "Ready for hash preflight: 1145 artifacts, 75 stacks, 90 subregions; $($Moves.Count) archive moves."
    exit 0
}
New-Item -ItemType Directory -Force -Path $RecordRoot | Out-Null
$Journal = Join-Path $RecordRoot "publication_$Stamp.json"
$SourceManifest = Read-Record 'source_manifest.json'
foreach ($Kind in @('dem','lc','provinces','subregions')) {
    $Item = $SourceManifest.$Kind
    $Relative = if ($Item.origin_relative) { $Item.origin_relative } else { $Item.path }
    $Path = Join-Path $DestinationRoot $Relative
    Write-Output "Checking original source: $Kind"
    if ((Hash $Path) -ne $Item.sha256) { throw "Source changed since staging: $Path" }
}
$Preserved = @()
$Copies = @()
$Checked = 0
foreach ($File in $Manifest) {
    $Target = Join-Path $DestinationRoot $File.path.Replace('/','\')
    if ((Test-Path -LiteralPath $Target) -and -not (Is-Archived $Target)) {
        if ((Get-Item -LiteralPath $Target).Length -ne $File.bytes -or (Hash $Target) -ne $File.sha256) {
            throw "Existing terrain differs from staged data before archiving: $Target"
        }
        $Preserved += $File
    } else {
        if ((Hash (Join-Path $OutputRoot $File.path.Replace('/','\'))) -ne $File.sha256) { throw "Staged content changed: $($File.path)" }
        $Copies += $File
    }
    $Checked++
    if ($Checked % 100 -eq 0) { Write-Output "Hash preflight: $Checked / 1145 files" }
}
$OldFiles = @()
foreach ($Move in $Moves) {
    $Item = Get-Item -LiteralPath $Move.source -Force
    $Files = if ($Item.PSIsContainer) { @(Get-ChildItem -LiteralPath $Move.source -File -Recurse -Force) } else { @($Item) }
    foreach ($File in $Files) {
        $OldFiles += [pscustomobject]@{source=$File.FullName; archive=($Move.target+$File.FullName.Substring($Move.source.Length));
            bytes=$File.Length; last_write_utc=$File.LastWriteTimeUtc.ToString('o')}
    }
}
$State = [ordered]@{stamp=$Stamp; status='inventoried'; archive_root=$ArchiveRoot; moves=$Moves; archived_files=$OldFiles;
    archive_verification='Same-volume Move-Item, complete path inventory, size and modification time verified';
    preserved_files=$Preserved; copied_files=$Copies; published_files=0}
function Save-State { $State | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Journal -Encoding UTF8 }
Save-State
foreach ($Move in $Moves) {
    if (Test-Path -LiteralPath $Move.target) { throw "Archive destination exists: $($Move.target)" }
    New-Item -ItemType Directory -Force -Path (Split-Path $Move.target -Parent) | Out-Null
    Move-Item -LiteralPath $Move.source -Destination $Move.target
}
$State.status='archived'
Save-State
foreach ($File in $OldFiles) {
    $Actual = Get-Item -LiteralPath $File.archive
    if ($Actual.Length -ne $File.bytes -or $Actual.LastWriteTimeUtc.ToString('o') -ne $File.last_write_utc -or
        (Test-Path -LiteralPath $File.source)) { throw "Archive move verification failed: $($File.archive)" }
}
foreach ($File in $Copies) {
    $Source = Join-Path $OutputRoot $File.path.Replace('/','\')
    $Target = Join-Path $DestinationRoot $File.path.Replace('/','\')
    if (Test-Path -LiteralPath $Target) { throw "Unexpected publication collision: $Target" }
    New-Item -ItemType Directory -Force -Path (Split-Path $Target -Parent) | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Target
    if ((Get-Item -LiteralPath $Target).Length -ne $File.bytes -or (Hash $Target) -ne $File.sha256) {
        throw "Publication verification failed: $Target"
    }
    $State.published_files++
    if ($State.published_files % 50 -eq 0) { Save-State; Write-Output "Published and verified $($State.published_files) / $($Copies.Count) files" }
}
$ActualPaths = @{}
foreach ($Root in $Roots.Values) {
    foreach ($File in Get-ChildItem -LiteralPath (Join-Path $DestinationRoot $Root) -Recurse -File -Force) {
        $ActualPaths[$File.FullName.Substring($DestinationRoot.Length+1).Replace('\','/')] = $true
    }
}
if ($ActualPaths.Count -ne 1145) { throw "Final file count is $($ActualPaths.Count), expected 1145" }
foreach ($Path in $ActualPaths.Keys) { if (-not $Expected.ContainsKey($Path)) { throw "Unexpected published path: $Path" } }
$State.status='complete'
Save-State
$Moves | Export-Csv -LiteralPath (Join-Path $RecordRoot "archive_paths_$Stamp.csv") -NoTypeInformation -Encoding UTF8
Write-Output "Publication complete: $Journal; $($Preserved.Count) files preserved in place, $($Copies.Count) replaced."
