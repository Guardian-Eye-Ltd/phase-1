import React from "react";
import { Activity, CheckCircle2, Clock, XCircle } from "lucide-react";
import { JobStage, JobStatus } from "../types/analysis";

interface AnalysisProgressProps {
    status: JobStatus;
    currentStage: JobStage;
    progress: number;
    onCancel?: () => void;
    errorMessage?: string;
}

const STAGES: { stage: JobStage; label: string }[] = [
    { stage: "VALIDATING", label: "Video Validation" },
    { stage: "EXTRACTING_METADATA", label: "Metadata Extraction" },
    { stage: "SAMPLING_FRAMES", label: "Frame Sampling" },
    { stage: "MOTION_ANALYSIS", label: "Motion Filtering" },
    { stage: "OBJECT_DETECTION", label: "Object Detection" },
    { stage: "TRACKING", label: "Multi-Object Tracking" },
    { stage: "KEYFRAME_EXTRACTION", label: "Keyframe Extraction" },
    { stage: "FINALIZING", label: "Finalizing Manifest" },
];

export const AnalysisProgress: React.FC<AnalysisProgressProps> = ({
    status,
    currentStage,
    progress,
    onCancel,
    errorMessage,
}) => {
    const currentStageIndex = STAGES.findIndex((s) => s.stage === currentStage);

    return (
        <div className="w-full rounded-xl border border-cyan-500/30 bg-[#0C101C] p-6 shadow-xl text-slate-100 mb-6">
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-cyan-950 border border-cyan-500/40 text-cyan-400">
                        <Activity className="w-5 h-5 animate-pulse" />
                    </div>
                    <div>
                        <h3 className="text-sm font-bold text-cyan-300 tracking-wide">
                            GuardianEye Computer Vision Engine
                        </h3>
                        <p className="text-xs text-slate-400">
                            Status: <span className="font-mono text-cyan-400 uppercase font-semibold">{status}</span>
                        </p>
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    <span className="font-mono text-lg font-bold text-cyan-400">
                        {Math.round(progress)}%
                    </span>
                    {status === "PROCESSING" && onCancel && (
                        <button
                            onClick={onCancel}
                            className="text-xs px-3 py-1.5 rounded-lg border border-red-500/40 bg-red-950/40 text-red-400 hover:bg-red-900/60 transition-colors"
                        >
                            Cancel Job
                        </button>
                    )}
                </div>
            </div>

            {/* Progress Bar */}
            <div className="w-full h-2.5 bg-slate-900 rounded-full overflow-hidden mb-6 border border-slate-800">
                <div
                    className="h-full bg-gradient-to-r from-cyan-500 via-blue-500 to-emerald-400 transition-all duration-500"
                    style={{ width: `${progress}%` }}
                />
            </div>

            {/* Error Display */}
            {status === "FAILED" && errorMessage && (
                <div className="p-3.5 rounded-lg bg-red-950/40 border border-red-900/50 text-red-300 text-xs flex items-start gap-2.5 mb-4">
                    <XCircle className="w-4 h-4 shrink-0 text-red-400 mt-0.5" />
                    <div>
                        <span className="font-semibold">Analysis Failed: </span>
                        {errorMessage}
                    </div>
                </div>
            )}

            {/* Stage Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                {STAGES.map((s, idx) => {
                    const isDone = currentStageIndex > idx || status === "COMPLETED";
                    const isCurrent = currentStageIndex === idx && status === "PROCESSING";

                    return (
                        <div
                            key={s.stage}
                            className={`p-2.5 rounded-lg border text-xs flex items-center gap-2 transition-all ${isDone
                                    ? "border-emerald-500/40 bg-emerald-950/20 text-emerald-300"
                                    : isCurrent
                                        ? "border-cyan-400 bg-cyan-950/60 text-cyan-200 shadow-md shadow-cyan-950/50"
                                        : "border-slate-800 bg-slate-900/40 text-slate-500"
                                }`}
                        >
                            {isDone ? (
                                <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                            ) : isCurrent ? (
                                <Activity className="w-4 h-4 text-cyan-400 shrink-0 animate-spin" />
                            ) : (
                                <Clock className="w-4 h-4 text-slate-600 shrink-0" />
                            )}
                            <span className="truncate font-medium">{s.label}</span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};
