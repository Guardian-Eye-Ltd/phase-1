import React, { useRef, useEffect, useState, useImperativeHandle, forwardRef } from "react";
import { Play, Pause, Volume2, VolumeX, Maximize, Eye, EyeOff } from "lucide-react";
import { Detection } from "../types/analysis";

interface VideoOverlayPlayerProps {
    streamUrl: string;
    detections: Detection[];
    mimeType?: string;
}

export interface VideoOverlayPlayerRef {
    seekTo: (timestamp: number) => void;
}

export const VideoOverlayPlayer = forwardRef<VideoOverlayPlayerRef, VideoOverlayPlayerProps>(
    ({ streamUrl, detections, mimeType = "video/mp4" }, ref) => {
        const videoRef = useRef<HTMLVideoElement>(null);
        const canvasRef = useRef<HTMLCanvasElement>(null);

        const [isPlaying, setIsPlaying] = useState(false);
        const [isMuted, setIsMuted] = useState(true);
        const [currentTime, setCurrentTime] = useState(0);
        const [duration, setDuration] = useState(0);
        const [overlayEnabled, setOverlayEnabled] = useState(true);
        const [videoError, setVideoError] = useState<string | null>(null);

        // Expose seekTo function to parent
        useImperativeHandle(ref, () => ({
            seekTo: (timestamp: number) => {
                if (videoRef.current) {
                    videoRef.current.currentTime = timestamp;
                    videoRef.current.play().catch(() => { });
                }
            },
        }));

        // Synchronize canvas bounding box rendering with video frame time
        useEffect(() => {
            const video = videoRef.current;
            const canvas = canvasRef.current;
            if (!video || !canvas) return;

            let animationFrameId: number;

            const renderOverlay = () => {
                if (canvas && video && overlayEnabled) {
                    const ctx = canvas.getContext("2d");
                    if (ctx) {
                        canvas.width = video.clientWidth || 640;
                        canvas.height = video.clientHeight || 360;
                        ctx.clearRect(0, 0, canvas.width, canvas.height);

                        const currTime = video.currentTime;

                        // Find detections matching current video timestamp (within 0.5s window)
                        const activeDets = detections.filter(
                            (d) => Math.abs(d.timestamp - currTime) <= 0.5
                        );

                        activeDets.forEach((det) => {
                            const [normX1, normY1, normX2, normY2] = det.bbox;
                            const x1 = normX1 * canvas.width;
                            const y1 = normY1 * canvas.height;
                            const w = (normX2 - normX1) * canvas.width;
                            const h = (normY2 - normY1) * canvas.height;

                            const isPerson = det.class_name.toLowerCase() === "person";
                            const strokeColor = isPerson ? "#00F0FF" : "#10B981"; // Cyan for person, Emerald for objects
                            const fillColor = isPerson ? "rgba(0, 240, 255, 0.15)" : "rgba(16, 185, 129, 0.15)";

                            // Draw Bounding Box
                            ctx.strokeStyle = strokeColor;
                            ctx.lineWidth = 2;
                            ctx.fillStyle = fillColor;
                            ctx.strokeRect(x1, y1, w, h);
                            ctx.fillRect(x1, y1, w, h);

                            // Draw Label Tag
                            const trackLabel = det.track_id ? `#${det.track_id}` : "";
                            const labelText = `${det.class_name} ${trackLabel} (${Math.round(det.confidence * 100)}%)`;

                            ctx.font = "bold 11px Inter, sans-serif";
                            const textWidth = ctx.measureText(labelText).width;
                            const tagHeight = 18;

                            ctx.fillStyle = strokeColor;
                            ctx.fillRect(x1, Math.max(0, y1 - tagHeight), textWidth + 10, tagHeight);

                            ctx.fillStyle = "#070A11";
                            ctx.fillText(labelText, x1 + 5, Math.max(12, y1 - 4));
                        });
                    }
                }
                animationFrameId = requestAnimationFrame(renderOverlay);
            };

            animationFrameId = requestAnimationFrame(renderOverlay);
            return () => cancelAnimationFrame(animationFrameId);
        }, [detections, overlayEnabled]);

        const togglePlay = () => {
            if (videoRef.current) {
                if (isPlaying) {
                    videoRef.current.pause();
                } else {
                    videoRef.current.play();
                }
                setIsPlaying(!isPlaying);
            }
        };

        const toggleMute = () => {
            if (videoRef.current) {
                videoRef.current.muted = !isMuted;
                setIsMuted(!isMuted);
            }
        };

        const handleSeekChange = (e: React.ChangeEvent<HTMLInputElement>) => {
            const newTime = parseFloat(e.target.value);
            if (videoRef.current) {
                videoRef.current.currentTime = newTime;
                setCurrentTime(newTime);
            }
        };

        const formatTime = (seconds: number) => {
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
        };

        return (
            <div className="relative w-full rounded-xl overflow-hidden border border-cyan-500/30 bg-[#070A11] shadow-2xl group">

                {/* Video Element & Overlay Canvas */}
                <div className="relative w-full aspect-video bg-black flex items-center justify-center">
                    <video
                        key={streamUrl}
                        ref={videoRef}
                        src={streamUrl}
                        muted={isMuted}
                        preload="metadata"
                        onTimeUpdate={() => {
                            if (videoRef.current) setCurrentTime(videoRef.current.currentTime);
                        }}
                        onLoadedMetadata={() => {
                            if (videoRef.current) { setDuration(videoRef.current.duration); setVideoError(null); }
                        }}
                        onPlay={() => setIsPlaying(true)}
                        onPause={() => setIsPlaying(false)}
                        onError={(e) => {
                            const vid = e.currentTarget;
                            const code = vid.error?.code;
                            const msgs: Record<number, string> = {
                                1: "Video loading aborted",
                                2: "Network error while loading video — check CORS / backend is running",
                                3: "Video decoding failed — codec not supported by this browser",
                                4: "Access denied — session may have expired, please refresh the page",
                            };
                            setVideoError(msgs[code ?? 4] ?? "Unknown video error");
                        }}
                        className="w-full h-full object-contain"
                    />

                    {/* Video Error Overlay */}
                    {videoError && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 backdrop-blur-sm text-center px-6">
                            <div className="text-red-400 text-4xl mb-3">⚠</div>
                            <p className="text-red-300 text-sm font-mono font-semibold">{videoError}</p>
                            <p className="text-slate-400 text-xs font-mono mt-2">Ensure the backend is running and you are logged in with a valid session.</p>
                            <div className="flex gap-3 mt-4">
                                <button
                                    onClick={() => {
                                        setVideoError(null);
                                        if (videoRef.current) {
                                            videoRef.current.load();
                                        }
                                    }}
                                    className="px-4 py-2 bg-cyan-950 hover:bg-cyan-900 text-cyan-300 border border-cyan-500/40 rounded-lg text-xs font-mono transition-colors"
                                >
                                    ↺ Retry Stream
                                </button>
                                <button
                                    onClick={() => {
                                        localStorage.removeItem("access_token");
                                        localStorage.removeItem("refresh_token");
                                        window.location.href = "/login";
                                    }}
                                    className="px-4 py-2 bg-red-950 hover:bg-red-900 text-red-300 border border-red-500/40 rounded-lg text-xs font-mono transition-colors"
                                >
                                    🔐 Clear Session & Re-Login
                                </button>
                            </div>
                        </div>
                    )}

                    <canvas
                        ref={canvasRef}
                        className="absolute inset-0 w-full h-full pointer-events-none"
                    />

                    {/* Overlay Toggle Badge */}
                    <button
                        onClick={() => setOverlayEnabled(!overlayEnabled)}
                        className={`absolute top-3 right-3 px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 border backdrop-blur-md transition-all ${overlayEnabled
                            ? "border-cyan-400 bg-cyan-950/80 text-cyan-300 shadow-md shadow-cyan-950/80"
                            : "border-slate-800 bg-slate-900/80 text-slate-400"
                            }`}
                    >
                        {overlayEnabled ? <Eye className="w-3.5 h-3.5 text-cyan-400" /> : <EyeOff className="w-3.5 h-3.5" />}
                        {overlayEnabled ? "AI Bounding Box Overlay Active" : "Overlay Disabled"}
                    </button>
                </div>

                {/* Custom Video Controls Bar */}
                <div className="p-3 bg-[#0C101C] border-t border-cyan-900/40 flex items-center gap-3">
                    <button
                        onClick={togglePlay}
                        className="p-2 rounded-lg bg-cyan-950/80 border border-cyan-500/40 text-cyan-400 hover:bg-cyan-900/60 transition-colors"
                    >
                        {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 ml-0.5" />}
                    </button>

                    <span className="font-mono text-xs text-slate-300 min-w-[85px]">
                        {formatTime(currentTime)} / {formatTime(duration)}
                    </span>

                    <input
                        type="range"
                        min="0"
                        max={duration || 100}
                        step="0.1"
                        value={currentTime}
                        onChange={handleSeekChange}
                        className="flex-1 h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                    />

                    <button
                        onClick={toggleMute}
                        className="p-2 rounded-lg bg-slate-900 text-slate-400 hover:text-slate-200 transition-colors"
                    >
                        {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
                    </button>

                    <button
                        onClick={() => {
                            if (videoRef.current) videoRef.current.requestFullscreen?.();
                        }}
                        className="p-2 rounded-lg bg-slate-900 text-slate-400 hover:text-slate-200 transition-colors"
                    >
                        <Maximize className="w-4 h-4" />
                    </button>
                </div>
            </div>
        );
    }
);

VideoOverlayPlayer.displayName = "VideoOverlayPlayer";
