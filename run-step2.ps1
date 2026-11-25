param(
    [Parameter(Mandatory = $true)]
    [string]$Video,
    [string]$Preview = 'roi_preview.jpg',
    [string]$Manual
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$ffmpeg = Get-ChildItem -LiteralPath (Join-Path $root '.tools\ffmpeg') -Filter ffmpeg.exe -Recurse -File |
    Select-Object -First 1 -ExpandProperty FullName
$ffprobe = Get-ChildItem -LiteralPath (Join-Path $root '.tools\ffmpeg') -Filter ffprobe.exe -Recurse -File |
    Select-Object -First 1 -ExpandProperty FullName
if (-not (Test-Path -LiteralPath $python) -or -not $ffmpeg -or -not $ffprobe) {
    throw 'Thiếu môi trường .venv hoặc FFmpeg trong .tools.'
}
if ($Manual -and $Manual.Split(',').Count -ne 4) {
    throw 'ROI thủ công cần đúng 4 tỷ lệ: x_min,y_min,x_max,y_max.'
}

$arguments = @('-m', 'core.roi_detector', $Video, '--preview', $Preview,
               '--ffmpeg', $ffmpeg, '--ffprobe', $ffprobe)
if ($Manual) {
    $arguments += '--manual'
    $arguments += $Manual.Split(',')
}
Push-Location $root
try {
    & $python @arguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
