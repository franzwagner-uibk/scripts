param(
    [ValidateSet("start", "stop", "status", "logs", "clean")]
    [string]$Action = "start",
    [string]$RepoDir = "C:\Users\franz\Nextcloud\PhD\openamundsen_da",
    [string]$ContainerName = "oa-da-docs-preview",
    [string]$BundleCacheDir = "",
    [int]$Port = 4001,
    [int]$LiveReloadPort = 35730,
    [switch]$OpenBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-DockerReady {
    docker version *> $null
    return ($LASTEXITCODE -eq 0)
}

function Ensure-DockerReady {
    if (Test-DockerReady) {
        return
    }

    $dockerDesktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path $dockerDesktop)) {
        throw "Docker Desktop not found: $dockerDesktop"
    }

    Write-Host "Starting Docker Desktop..."
    Start-Process $dockerDesktop | Out-Null

    $deadline = (Get-Date).AddMinutes(3)
    do {
        Start-Sleep -Seconds 3
        if (Test-DockerReady) {
            return
        }
    } while ((Get-Date) -lt $deadline)

    throw "Docker daemon did not become ready within 3 minutes."
}

function Require-Repo {
    if (-not (Test-Path $RepoDir)) {
        throw "RepoDir does not exist: $RepoDir"
    }
    $docsDir = Join-Path $RepoDir "docs"
    if (-not (Test-Path $docsDir)) {
        throw "docs directory not found: $docsDir"
    }
    return $docsDir
}

function Test-ContainerRunning {
    param([string]$Name)
    $raw = & docker ps -q --filter ("name={0}" -f $Name)
    if ($null -eq $raw) {
        return $false
    }
    $id = [string]$raw
    $id = $id.Trim()
    return (-not [string]::IsNullOrWhiteSpace($id))
}

function Start-Preview {
    $docsDir = Require-Repo
    $bundleCache = if ([string]::IsNullOrWhiteSpace($BundleCacheDir)) {
        Join-Path $env:LOCALAPPDATA "openamundsen_da\docs-bundle-cache"
    } else {
        $BundleCacheDir
    }
    $docsMount = (($docsDir -replace "\\", "/") + ":/srv/jekyll")
    $bundleMount = (($bundleCache -replace "\\", "/") + ":/usr/local/bundle")

    if (-not (Test-Path $bundleCache)) {
        New-Item -ItemType Directory -Path $bundleCache | Out-Null
    }

    Ensure-DockerReady

    if (Test-ContainerRunning -Name $ContainerName) {
        Write-Host "Docs preview already running: http://127.0.0.1:$Port/"
        if ($OpenBrowser) {
            Start-Process ("http://127.0.0.1:{0}/" -f $Port) | Out-Null
        }
        return
    }

    cmd /c "docker rm -f $ContainerName >nul 2>&1"

    $dockerArgs = @(
        "run", "-d",
        "--name", $ContainerName,
        "-p", "${Port}:4000",
        "-p", "${LiveReloadPort}:35729",
        "-v", $docsMount,
        "-v", $bundleMount,
        "-w", "/srv/jekyll",
        "jekyll/jekyll:4",
        "sh", "-lc",
        "bundle check || bundle install --jobs 4 --retry 3; JEKYLL_CACHE_DIR=/tmp/jekyll-cache bundle exec jekyll serve --host 0.0.0.0 --port 4000 --livereload --incremental --force_polling --destination /tmp/_site --config _config.yml,_config_dev.yml"
    )

    $containerId = & docker @dockerArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to start docs preview container."
    }

    Write-Host "Started container: $containerId"
    Write-Host "Waiting for http://127.0.0.1:$Port/ ..."

    $deadline = (Get-Date).AddMinutes(3)
    do {
        Start-Sleep -Seconds 2
        try {
            $resp = Invoke-WebRequest -UseBasicParsing -Uri ("http://127.0.0.1:{0}/" -f $Port) -TimeoutSec 5
            if ($resp.StatusCode -eq 200) {
                Write-Host "Docs preview ready: http://127.0.0.1:$Port/"
                if ($OpenBrowser) {
                    Start-Process ("http://127.0.0.1:{0}/" -f $Port) | Out-Null
                }
                return
            }
        } catch {
            # Keep waiting while bundler/jekyll initializes.
        }
    } while ((Get-Date) -lt $deadline)

    Write-Warning "Preview did not become reachable within timeout. Showing recent logs:"
    & docker logs --tail 120 $ContainerName
    throw "Docs preview startup timed out."
}

function Stop-Preview {
    Ensure-DockerReady
    cmd /c "docker rm -f $ContainerName >nul 2>&1"
}

function Show-Status {
    Ensure-DockerReady
    & docker ps -a --filter ("name={0}" -f $ContainerName) --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

function Show-Logs {
    Ensure-DockerReady
    & docker logs --tail 200 $ContainerName
}

function Clean-RepoDocsArtifacts {
    $docsDir = Require-Repo
    $bundleCache = if ([string]::IsNullOrWhiteSpace($BundleCacheDir)) {
        Join-Path $env:LOCALAPPDATA "openamundsen_da\docs-bundle-cache"
    } else {
        $BundleCacheDir
    }
    $paths = @(
        (Join-Path $RepoDir ".bundle-cache"), # legacy repo-local cache path
        (Join-Path $docsDir ".jekyll-cache"),
        (Join-Path $docsDir ".jekyll-metadata"),
        $bundleCache
    )

    foreach ($p in $paths) {
        if (Test-Path $p) {
            Remove-Item -Recurse -Force $p
            Write-Host "Removed $p"
        }
    }
}

switch ($Action) {
    "start"  { Start-Preview; break }
    "stop"   { Stop-Preview; break }
    "status" { Show-Status; break }
    "logs"   { Show-Logs; break }
    "clean"  { Clean-RepoDocsArtifacts; break }
}
