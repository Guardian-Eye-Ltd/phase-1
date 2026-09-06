import React, { useState } from "react";
import { Sliders, Cpu, Eye, Zap, X } from "lucide-react";

interface AnalysisModalProps {
    isOpen: boolean;
    onClose: () => void;
    onStartAnalysis: (config: { sampling_fps: number; confidence_threshold: number; tracking_enabled: boolean }) => void;
    isSubmitting?: boolean;
}

export const AnalysisModal: React.FC<AnalysisModalProps> = ({
    isOpen,
    onClose,
    onStartAnalysis,
    isSubmitting = false,
}) => {
    const [samplingFps, setSamplingFps] = useState<number>(2.0);
    const [confidence, setConfidence] = useState<number>(0.4);
    const [trackingEnabled, setTrackingEnabled] = useState<boolean>(true);

    if (!isOpen) return null;

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        onStartAnalysis({
            sampling_fps: samplingFps,
            confidence_threshold: confidence,
            tracking_enabled: trackingEnabled,
        });
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
            <div className="w-full max-w-lg rounded-xl border border-cyan-500/30 bg-[#0C101C] p-6 shadow-2xl shadow-cyan-950/50 text-slate-100">

                {/* Header */}
                <div className="flex items-center justify-between border-b border-cyan-900/40 pb-4 mb-6">
                    <div className="flex items-center gap-3">
                        <div className="p-2.5 rounded-lg bg-cyan-950 border border-cyan-500/40 text-cyan-400">
                            <Cpu className="w-6 h-6 animate-pulse" />
                        </div>
                        <div>
                            <h2 className="text-lg font-bold tracking-wide text-cyan-300">
                                Configure Video Analysis Pipeline
                            </h2>
                            <p className="text-xs text-slate-400">
                                Post-Event CCTV Computer Vision & Tracking Engine
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="text-slate-400 hover:text-cyan-400 transition-colors p-1"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                <form onSubmit={handleSubmit} className="space-y-6">
                    {/* Frame Sampling Rate */}
                    <div>
                        <label className="block text-xs font-semibold uppercase tracking-wider text-slate-300 mb-2 flex items-center gap-2">
                            <Zap className="w-4 h-4 text-amber-400" />
                            Frame Sampling Rate (FPS)
                        </label>
                        <div className="grid grid-cols-4 gap-2">
                            {[1.0, 2.0, 5.0, 10.0].map((fps) => (
                                <button
                                    key={fps}
                                    type="button"
                                    onClick={() => setSamplingFps(fps)}
                                    className={`py-2 px-3 rounded-lg text-xs font-mono font-medium border transition-all ${samplingFps === fps
                                        ? "border-cyan-400 bg-cyan-950/80 text-cyan-300 shadow-md shadow-cyan-900/40"
                                        : "border-slate-800 bg-slate-900/60 text-slate-400 hover:border-slate-700"
                                        }`}
                                >
                                    {fps} FPS
                                </button>
                            ))}
                        </div>
                        <p className="text-[11px] text-slate-500 mt-1.5">
                            {samplingFps === 2.0 ? "Recommended for balanced analysis speed and accuracy." : `Extracts ${samplingFps} frame(s) per second.`}
                        </p>
                    </div>

                    {/* Confidence Threshold */}
                    <div>
                        <div className="flex justify-between items-center mb-2">
                            <label className="text-xs font-semibold uppercase tracking-wider text-slate-300 flex items-center gap-2">
                                <Sliders className="w-4 h-4 text-cyan-400" />
                                Detection Confidence Threshold
                            </label>
                            <span className="font-mono text-xs text-cyan-400 font-bold px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-800/40">
                                {Math.round(confidence * 100)}%
                            </span>
                        </div>
                        <input
                            type="range"
                            min="0.10"
                            max="0.90"
                            step="0.05"
                            value={confidence}
                            onChange={(e) => setConfidence(parseFloat(e.target.value))}
                            className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                        />
                        <div className="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
                            <span>10% (High Recall)</span>
                            <span>40% (Standard)</span>
                            <span>90% (High Precision)</span>
                        </div>
                    </div>

                    {/* Multi-Object Tracking Toggle */}
                    <div className="flex items-center justify-between p-3.5 rounded-lg border border-slate-800 bg-slate-900/50">
                        <div className="flex items-center gap-3">
                            <Eye className="w-5 h-5 text-emerald-400" />
                            <div>
                                <div className="text-xs font-semibold text-slate-200">
                                    Multi-Object Entity Tracking
                                </div>
                                <div className="text-[11px] text-slate-400">
                                    Assigns persistent Track IDs across video frames
                                </div>
                            </div>
                        </div>
                        <input
                            type="checkbox"
                            checked={trackingEnabled}
                            onChange={(e) => setTrackingEnabled(e.target.checked)}
                            className="w-4 h-4 rounded border-slate-700 text-cyan-500 focus:ring-cyan-500 focus:ring-offset-slate-900 accent-cyan-500 cursor-pointer"
                        />
                    </div>

                    {/* Model Summary Badge */}
                    <div className="p-3 rounded-lg bg-cyan-950/30 border border-cyan-900/40 text-[11px] text-cyan-300/80 font-mono space-y-1">
                        <div>Detector Engine: YOLOv8 / OpenCV Fallback (10 Classes)</div>
                        <div>Tracker Algorithm: ByteTrack / IoU Multi-Object Tracker</div>
                        <div>Original File Integrity: Immutable (SHA-256 Verified)</div>
                    </div>

                    {/* Actions */}
                    <div className="flex justify-end gap-3 pt-2">
                        <button
                            type="button"
                            onClick={onClose}
                            className="px-4 py-2 text-xs font-medium rounded-lg border border-slate-800 text-slate-300 hover:bg-slate-800/60 transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            disabled={isSubmitting}
                            className="px-5 py-2 text-xs font-semibold rounded-lg bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-lg shadow-cyan-900/50 transition-all flex items-center gap-2 disabled:opacity-50"
                        >
                            {isSubmitting ? (
                                <>
                                    <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                                    Starting Job...
                                </>
                            ) : (
                                <>Start Analysis Pipeline</>
                            )}
                        </button>
                    </div>
                </form>

            </div>
        </div>
    );
};
