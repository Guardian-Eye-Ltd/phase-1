param (
    [switch]$Help
)

if ($Help) {
    Write-Host "GuardianEye Demo Cameras Startup Script"
    Write-Host "---------------------------------------"
    Write-Host "This script starts MediaMTX and streams 6 local MP4 files as RTSP sources."
    Write-Host "Ensure ffmpeg is installed and added to your system PATH."
    Write-Host "Ensure demo videos are placed in backend/demo_streams/"
    exit 0
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$BackendDir = Split-Path -Parent $ScriptDir
$DemoStreamsDir = Join-Path $BackendDir "demo_streams"

# 1. Check for FFmpeg
$FFmpegExe = "ffmpeg"
if (-not (Get-Command "ffmpeg" -ErrorAction SilentlyContinue)) {
    $FFmpegExe = Join-Path $ScriptDir "ffmpeg.exe"
    if (-not (Test-Path $FFmpegExe)) {
        Write-Host "FFmpeg not found in PATH or scripts directory. Downloading FFmpeg..."
        $DownloadUrl = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        $ZipPath = Join-Path $ScriptDir "ffmpeg.zip"
        Invoke-WebRequest -Uri $DownloadUrl -OutFile $ZipPath
        
        Write-Host "Extracting FFmpeg..."
        Expand-Archive -Path $ZipPath -DestinationPath $ScriptDir -Force
        
        # The zip extracts to a folder like ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe
        $ExtractedExe = Get-ChildItem -Path $ScriptDir -Filter "ffmpeg.exe" -Recurse | Select-Object -First 1
        if ($ExtractedExe) {
            Copy-Item $ExtractedExe.FullName -Destination $FFmpegExe
        }
        
        Remove-Item $ZipPath
        # Clean up the extracted folder
        $ExtractedFolder = Get-ChildItem -Path $ScriptDir -Directory -Filter "ffmpeg-master-latest-win64-gpl"
        if ($ExtractedFolder) { Remove-Item $ExtractedFolder.FullName -Recurse -Force }
        
        if (-not (Test-Path $FFmpegExe)) {
            Write-Error "Failed to download or extract FFmpeg."
            exit 1
        }
        Write-Host "FFmpeg downloaded successfully."
    }
}

# 2. Check for MediaMTX
$MediaMTXExe = Join-Path $ScriptDir "mediamtx.exe"
if (-not (Test-Path $MediaMTXExe)) {
    Write-Host "MediaMTX not found in scripts directory. Downloading v1.6.0..."
    $DownloadUrl = "https://github.com/bluenviron/mediamtx/releases/download/v1.6.0/mediamtx_v1.6.0_windows_amd64.zip"
    $ZipPath = Join-Path $ScriptDir "mediamtx.zip"
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $ZipPath
    Expand-Archive -Path $ZipPath -DestinationPath $ScriptDir -Force
    Remove-Item $ZipPath
    if (-not (Test-Path $MediaMTXExe)) {
        Write-Error "Failed to download or extract MediaMTX."
        exit 1
    }
    Write-Host "MediaMTX downloaded successfully."
}

# 3. Check for Demo Videos
$MissingFiles = $false
for ($i = 1; $i -le 6; $i++) {
    $camId = "{0:D2}" -f $i
    $videoPath = Join-Path $DemoStreamsDir "camera${camId}.mp4"
    if (-not (Test-Path $videoPath)) {
        Write-Host "Missing demo video file: $videoPath" -ForegroundColor Red
        $MissingFiles = $true
    }
}

if ($MissingFiles) {
    Write-Host ""
    Write-Host "ERROR: One or more required demo videos are missing." -ForegroundColor Red
    Write-Host "Please place camera01.mp4 through camera06.mp4 in the backend/demo_streams/ directory."
    Write-Host "You can use any standard H.264 MP4 file."
    exit 1
}

# 4. Start MediaMTX
Write-Host "Starting MediaMTX RTSP Server..."
Start-Process -FilePath $MediaMTXExe -WindowStyle Minimized -PassThru | Out-Null
Start-Sleep -Seconds 2

# 5. Start FFmpeg Streams
Write-Host "Starting FFmpeg streams..."
for ($i = 1; $i -le 6; $i++) {
    $camId = "{0:D2}" -f $i
    $videoPath = Join-Path $DemoStreamsDir "camera${camId}.mp4"
    $rtspUrl = "rtsp://127.0.0.1:8554/camera${camId}"
    
    # FFmpeg command: Read file in real-time (-re), loop infinitely (-stream_loop -1), copy codec to RTSP
    $ffmpegArgs = "-re -stream_loop -1 -i `"$videoPath`" -c copy -f rtsp -rtsp_transport tcp $rtspUrl"
    
    Start-Process -FilePath $FFmpegExe -ArgumentList $ffmpegArgs -WindowStyle Minimized
    Write-Host "Started stream: $rtspUrl" -ForegroundColor Green
}

Write-Host ""
Write-Host "All demo cameras are now streaming!" -ForegroundColor Cyan
Write-Host "To stop the streams, you will need to manually close the MediaMTX and FFmpeg windows."
