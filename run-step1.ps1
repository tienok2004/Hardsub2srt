param(
    [Parameter(Mandatory = $true)]
    [string]$Video,
    [ValidateRange(0.01, 1000)]
    [double]$Fps = 3
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Get-ChildItem -LiteralPath (Join-Path $root '.tools\python') -Filter python.exe -Recurse -File |
    Select-Object -First 1 -ExpandProperty FullName
$ffmpeg = Get-ChildItem -LiteralPath (Join-Path $root '.tools\ffmpeg') -Filter ffmpeg.exe -Recurse -File |
    Select-Object -First 1 -ExpandProperty FullName
$ffprobe = Get-ChildItem -LiteralPath (Join-Path $root '.tools\ffmpeg') -Filter ffprobe.exe -Recurse -File |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $python -or -not $ffmpeg -or -not $ffprobe) {
    throw 'Thiếu Python hoặc FFmpeg trong .tools. Hãy cài lại công cụ trước khi chạy.'
}

Push-Location $root
try {
    & $python -m core.gpu
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python -m core.video_sampler $Video --fps $Fps --ffmpeg $ffmpeg --ffprobe $ffprobe
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
