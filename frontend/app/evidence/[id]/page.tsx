"use client";

import React, { useEffect, useState, useRef } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { evidenceService } from "@/services/evidenceService";
import { analysisService } from "@/services/analysisService";
import { Evidence } from "@/types/evidence";
import {
    AnalysisJob,
    Detection,
    Track,
    Keyframe,
    ActivityInterval,
    TimelineEvent,
    ManifestResponse
} from "@/types/analysis";
import Link from "next/link";
import { useParams } from "next/navigation";
import {
    Film,
    ShieldCheck,
    Copy,
    Check,
    ArrowLeft,
    RefreshCw,
    Eye,
    Crosshair,
    Clock,
    Sparkles,
    Layers,
    Cpu,
    FileCode,
    Image as ImageIcon,
    Activity as ActivityIcon,
    Play
} from "lucide-react";
import { AnalysisModal } from "@/components/AnalysisModal";
import { AnalysisProgress } from "@/components/AnalysisProgress";
import { VideoOverlayPlayer, VideoOverlayPlayerRef } from "@/components/VideoOverlayPlayer";

export default function EvidenceDetailsPage() {
    const params = useParams();
    const evidenceId = Number(params?.id);

    const [evidence, setEvidence] = useState<Evidence | null>(null);
    const [loading, setLoading] = useState(true);
    const [copied, setCopied] = useState(false);
    const [copiedManifestHash, setCopiedManifestHash] = useState(false);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [isSubmittingJob, setIsSubmittingJob] = useState(false);

    // Analysis Pipeline States
    const [job, setJob] = useState<AnalysisJob | null>(null);
    const [detections, setDetections] = useState<Detection[]>([]);
    const [tracks, setTracks] = useState<Track[]>([]);
    const [keyframes, setKeyframes] = useState<Keyframe[]>([]);
    const [activity, setActivity] = useState<ActivityInterval[]>([]);
    const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
    const [manifest, setManifest] = useState<ManifestResponse | null>(null);

    const [activeTab, setActiveTab] = useState<"overview" | "timeline" | "tracks" | "keyframes" | "activity" | "manifest">("overview");

    const playerRef = useRef<VideoOverlayPlayerRef>(null);
    const [streamUrl, setStreamUrl] = useState<string>("");
    const [playerKey, setPlayerKey] = useState(0);

    const fetchDetails = async () => {
        if (!evidenceId) return;
        setLoading(true);
        try {
            const data = await evidenceService.getEvidenceById(evidenceId);
            setEvidence(data);
            // Build stream URL fresh after fetching — ensures latest token from localStorage is used
            setStreamUrl(evidenceService.getStreamUrl(evidenceId));
            await loadAnalysisData();
        } catch (err) {
            console.error("Failed to load evidence details", err);
        } finally {
            setLoading(false);
        }
    };

    const loadAnalysisData = async () => {
        if (!evidenceId) return;
        try {
            const [detsData, tracksData, kfData, actData, tlData] = await Promise.all([
                analysisService.getDetections(evidenceId).catch(() => []),
                analysisService.getTracks(evidenceId).catch(() => []),
                analysisService.getKeyframes(evidenceId).catch(() => []),
                analysisService.getActivityIntervals(evidenceId).catch(() => []),
                analysisService.getTimeline(evidenceId).catch(() => [])
            ]);
            setDetections(detsData);
            setTracks(tracksData);
            setKeyframes(kfData);
            setActivity(actData);
            setTimeline(tlData);

            try {
                const manifestRes = await analysisService.getManifest(evidenceId);
                setManifest(manifestRes);
            } catch {
                setManifest(null);
            }
        } catch (err) {
            console.error("Failed loading forensic analysis data", err);
        }
    };

    useEffect(() => {
        fetchDetails();
    }, [evidenceId]);

    // Real-time job polling interval when job is QUEUED or PROCESSING
    useEffect(() => {
        if (!job || job.status === "COMPLETED" || job.status === "FAILED" || job.status === "CANCELLED") {
            return;
        }

        const interval = setInterval(async () => {
            try {
                const updatedJob = await analysisService.getAnalysisJob(job.job_id);
                setJob(updatedJob);

                if (updatedJob.status === "COMPLETED") {
                    clearInterval(interval);
                    loadAnalysisData();
                    fetchDetails();
                } else if (updatedJob.status === "FAILED" || updatedJob.status === "CANCELLED") {
                    clearInterval(interval);
                }
            } catch (err) {
                console.error("Failed polling job status", err);
            }
        }, 2000);

        return () => clearInterval(interval);
    }, [job]);

    const handleStartAnalysis = async (config: { sampling_fps: number; confidence_threshold: number; tracking_enabled: boolean }) => {
        if (!evidenceId) return;
        setIsSubmittingJob(true);
        try {
            const newJob = await analysisService.startAnalysis(evidenceId, config);
            setJob(newJob);
            setIsModalOpen(false);
        } catch (err) {
            console.error("Failed starting analysis job", err);
        } finally {
            setIsSubmittingJob(false);
        }
    };

    const handleCancelJob = async () => {
        if (!job) return;
        try {
            await analysisService.cancelAnalysisJob(job.job_id);
            setJob((prev) => (prev ? { ...prev, status: "CANCELLED" } : null));
        } catch (err) {
            console.error("Failed cancelling analysis job", err);
        }
    };

    const handleCopyHash = () => {
        if (evidence?.sha256_hash) {
            navigator.clipboard.writeText(evidence.sha256_hash);
            setCopied(true);
            setTimeout(() => setCopied(false), 2000);
        }
    };

    const handleSeek = (timestamp: number) => {
        if (playerRef.current) {
            playerRef.current.seekTo(timestamp);
        }
    };

    if (loading) {
        return (
            <DashboardLayout>
                <div className="py-20 text-center text-xs font-mono text-cyan-400 animate-pulse">
                    Loading Forensic Evidence Record #{evidenceId}...
                </div>
            </DashboardLayout>
        );
    }

    if (!evidence) {
        return (
            <DashboardLayout>
                <div className="py-20 text-center bg-[#0D1424] border border-slate-800 rounded-xl">
                    <Film className="w-12 h-12 text-slate-600 mx-auto mb-3" />
                    <h3 className="text-base font-semibold text-white">Evidence Record Not Found</h3>
                    <Link href="/evidence" className="mt-4 inline-block px-4 py-2 bg-cyan-950 text-cyan-400 border border-cyan-500/30 rounded text-xs font-mono">
                        Return to Evidence Vault
                    </Link>
                </div>
            </DashboardLayout>
        );
    }


    return (
        <DashboardLayout>
            <div className="space-y-6">

                {/* Top Header */}
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div>
                        <Link href="/evidence" className="text-xs font-mono text-cyan-400 hover:underline flex items-center mb-1">
                            <ArrowLeft className="w-3.5 h-3.5 mr-1" /> BACK TO EVIDENCE VAULT
                        </Link>
                        <div className="flex items-center space-x-3">
                            <h1 className="text-2xl font-bold text-white tracking-tight truncate max-w-md">{evidence.original_filename}</h1>
                            <span className={`px-2.5 py-0.5 rounded text-xs font-mono uppercase font-bold border ${evidence.status === "COMPLETED" ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30" :
                                evidence.status === "PROCESSING" ? "bg-amber-500/10 text-amber-400 border-amber-500/30" :
                                    "bg-slate-800 text-slate-300 border-slate-700"
                                }`}>
                                {evidence.status}
                            </span>
                        </div>
                        <p className="text-xs text-slate-400 font-mono mt-1">EVIDENCE RECORD ID: #{evidence.id} • UPLOADED {new Date(evidence.uploaded_at).toLocaleString()}</p>
                    </div>

                    <div className="flex items-center space-x-3">
                        <button
                            onClick={() => setIsModalOpen(true)}
                            className="px-4 py-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white rounded-lg text-xs font-semibold shadow-lg shadow-cyan-950/50 flex items-center transition-all"
                        >
                            <Cpu className="w-4 h-4 mr-2 animate-pulse" />
                            ANALYZE EVIDENCE
                        </button>
                    </div>
                </div>

                {/* Real-time Analysis Progress Component */}
                {job && (
                    <AnalysisProgress
                        status={job.status}
                        currentStage={job.current_stage}
                        progress={job.progress}
                        errorMessage={job.error_message}
                        onCancel={handleCancelJob}
                    />
                )}

                {/* Cryptographic SHA-256 Hash Card */}
                <div className="bg-[#070A11] border border-cyan-500/30 rounded-xl p-4 flex flex-col md:flex-row md:items-center justify-between gap-4 shadow-lg shadow-cyan-950/20">
                    <div className="flex items-center space-x-3">
                        <div className="p-2.5 bg-emerald-500/10 rounded-lg border border-emerald-500/30 text-emerald-400">
                            <ShieldCheck className="w-6 h-6" />
                        </div>
                        <div>
                            <div className="flex items-center space-x-2">
                                <span className="text-xs font-mono text-emerald-400 font-bold tracking-wider uppercase">CHAIN-OF-CUSTODY SHA-256 HASH VERIFIED</span>
                                <span className="text-[10px] font-mono text-slate-400 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">ORIGINAL EVIDENCE ISOLATED</span>
                            </div>
                            <p className="text-xs font-mono text-cyan-300 mt-1 break-all select-all font-semibold">
                                {evidence.sha256_hash}
                            </p>
                        </div>
                    </div>

                    <button
                        onClick={handleCopyHash}
                        className="px-3 py-1.5 bg-cyan-950/80 hover:bg-cyan-900/80 text-cyan-300 border border-cyan-500/30 rounded text-xs font-mono flex items-center shrink-0 transition-colors"
                    >
                        {copied ? <Check className="w-3.5 h-3.5 mr-1 text-emerald-400" /> : <Copy className="w-3.5 h-3.5 mr-1" />}
                        {copied ? "COPIED!" : "COPY SHA-256 HASH"}
                    </button>
                </div>

                {/* Video Player + Technical Info */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Interactive Bounding Box Overlay Video Player */}
                    <div className="lg:col-span-2">
                        <VideoOverlayPlayer
                            key={`${playerKey}-${streamUrl}`}
                            ref={playerRef}
                            streamUrl={streamUrl}
                            detections={detections}
                            mimeType={evidence.mime_type}
                        />
                    </div>

                    {/* Technical Video Info Card */}
                    <div className="bg-[#0D1424] border border-slate-800 rounded-xl p-5 flex flex-col justify-between space-y-4">
                        <div>
                            <h2 className="font-bold text-white text-sm border-b border-slate-800 pb-3 flex items-center">
                                <Layers className="w-4 h-4 text-cyan-400 mr-2" />
                                Technical Forensic Parameters
                            </h2>

                            <div className="mt-4 space-y-3 font-mono text-xs">
                                <div className="flex justify-between border-b border-slate-800/60 pb-2">
                                    <span className="text-slate-400">File Size:</span>
                                    <span className="text-white font-semibold">{(evidence.file_size / (1024 * 1024)).toFixed(2)} MB</span>
                                </div>
                                <div className="flex justify-between border-b border-slate-800/60 pb-2">
                                    <span className="text-slate-400">Duration:</span>
                                    <span className="text-white font-semibold">{evidence.duration ? `${evidence.duration} sec` : "N/A"}</span>
                                </div>
                                <div className="flex justify-between border-b border-slate-800/60 pb-2">
                                    <span className="text-slate-400">Resolution:</span>
                                    <span className="text-white font-semibold">{evidence.resolution || "1920x1080"}</span>
                                </div>
                                <div className="flex justify-between border-b border-slate-800/60 pb-2">
                                    <span className="text-slate-400">FPS Rate:</span>
                                    <span className="text-white font-semibold">{evidence.fps || 30.0} FPS</span>
                                </div>
                                <div className="flex justify-between border-b border-slate-800/60 pb-2">
                                    <span className="text-slate-400">Detections Count:</span>
                                    <span className="text-cyan-400 font-bold">{detections.length} Boxes</span>
                                </div>
                                <div className="flex justify-between pb-1">
                                    <span className="text-slate-400">Tracks Recorded:</span>
                                    <span className="text-emerald-400 font-bold">{tracks.length} Entities</span>
                                </div>
                            </div>
                        </div>

                        <div className="p-3 bg-cyan-950/30 border border-cyan-900/40 rounded-lg text-[11px] font-mono text-cyan-300/80">
                            <span className="text-cyan-400 font-bold block mb-1">CCTV POST-EVENT ANALYSIS</span>
                            Clicking any timeline event, keyframe, or tracked person automatically seeks the player to the exact video timestamp.
                        </div>
                    </div>
                </div>

                {/* Forensic Analysis Navigation Tabs */}
                <div className="border-b border-slate-800 pt-4">
                    <div className="flex overflow-x-auto space-x-2 pb-2">
                        {[
                            { id: "overview", label: "Analysis Overview", icon: Sparkles },
                            { id: "timeline", label: `Timeline (${timeline.length})`, icon: Clock },
                            { id: "tracks", label: `People & Tracks (${tracks.length})`, icon: Crosshair },
                            { id: "keyframes", label: `Keyframes (${keyframes.length})`, icon: ImageIcon },
                            { id: "activity", label: `Activity (${activity.length})`, icon: ActivityIcon },
                            { id: "manifest", label: "Cryptographic Manifest", icon: FileCode },
                        ].map((tab) => {
                            const Icon = tab.icon;
                            const isActive = activeTab === tab.id;
                            return (
                                <button
                                    key={tab.id}
                                    onClick={() => setActiveTab(tab.id as any)}
                                    className={`flex items-center px-4 py-2.5 rounded-lg text-xs font-mono transition-all shrink-0 ${isActive
                                        ? "bg-cyan-950 text-cyan-300 border border-cyan-500/40 font-bold shadow-md shadow-cyan-950/50"
                                        : "text-slate-400 hover:bg-slate-900 hover:text-slate-200 border border-transparent"
                                        }`}
                                >
                                    <Icon className="w-3.5 h-3.5 mr-2" />
                                    {tab.label}
                                </button>
                            );
                        })}
                    </div>
                </div>

                {/* Tab Content Display Cards */}
                <div className="bg-[#0D1424] border border-slate-800 rounded-xl p-6 shadow-xl">

                    {/* TAB 1: OVERVIEW */}
                    {activeTab === "overview" && (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                            <div className="p-4 bg-[#070A11] border border-slate-800 rounded-xl space-y-2">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-mono text-slate-400">Total Detections</span>
                                    <Eye className="w-4 h-4 text-cyan-400" />
                                </div>
                                <div className="text-2xl font-bold font-mono text-white">{detections.length}</div>
                                <div className="text-[11px] text-slate-500 font-mono">Bounding boxes identified</div>
                            </div>

                            <div className="p-4 bg-[#070A11] border border-slate-800 rounded-xl space-y-2">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-mono text-slate-400">Tracked Entities</span>
                                    <Crosshair className="w-4 h-4 text-emerald-400" />
                                </div>
                                <div className="text-2xl font-bold font-mono text-white">{tracks.length}</div>
                                <div className="text-[11px] text-slate-500 font-mono">Persistent track histories</div>
                            </div>

                            <div className="p-4 bg-[#070A11] border border-slate-800 rounded-xl space-y-2">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-mono text-slate-400">Keyframes Extracted</span>
                                    <ImageIcon className="w-4 h-4 text-amber-400" />
                                </div>
                                <div className="text-2xl font-bold font-mono text-white">{keyframes.length}</div>
                                <div className="text-[11px] text-slate-500 font-mono">SHA-256 hashed images</div>
                            </div>

                            <div className="p-4 bg-[#070A11] border border-slate-800 rounded-xl space-y-2">
                                <div className="flex items-center justify-between">
                                    <span className="text-xs font-mono text-slate-400">Activity Intervals</span>
                                    <ActivityIcon className="w-4 h-4 text-indigo-400" />
                                </div>
                                <div className="text-2xl font-bold font-mono text-white">{activity.length}</div>
                                <div className="text-[11px] text-slate-500 font-mono">Motion activity segments</div>
                            </div>
                        </div>
                    )}

                    {/* TAB 2: INTERACTIVE TIMELINE */}
                    {activeTab === "timeline" && (
                        <div className="space-y-4">
                            <h3 className="text-sm font-bold text-white tracking-wide font-mono mb-4 flex items-center gap-2">
                                <Clock className="w-4 h-4 text-cyan-400" />
                                Chronological Forensic Timeline (Click to Seek Video)
                            </h3>
                            {timeline.length === 0 ? (
                                <div className="text-center py-12 text-xs font-mono text-slate-500">
                                    No forensic timeline events recorded yet. Run "ANALYZE EVIDENCE" to generate events.
                                </div>
                            ) : (
                                <div className="relative border-l border-slate-800 ml-4 space-y-4 pl-6">
                                    {timeline.map((evt, idx) => (
                                        <div
                                            key={idx}
                                            onClick={() => handleSeek(evt.timestamp)}
                                            className="group p-3.5 rounded-lg border border-slate-800 bg-[#070A11] hover:border-cyan-500/40 transition-all cursor-pointer flex items-center justify-between"
                                        >
                                            <div className="space-y-1">
                                                <div className="flex items-center gap-2">
                                                    <span className="font-mono text-xs font-bold text-cyan-400 bg-cyan-950 px-2 py-0.5 rounded border border-cyan-800/40">
                                                        {evt.timestamp.toFixed(2)}s
                                                    </span>
                                                    <span className="text-xs font-semibold text-slate-200">{evt.title}</span>
                                                </div>
                                                <p className="text-xs text-slate-400 font-mono">{evt.description}</p>
                                            </div>

                                            <button className="px-2.5 py-1.5 rounded text-xs font-mono bg-cyan-950 text-cyan-400 border border-cyan-500/30 group-hover:bg-cyan-900 transition-colors flex items-center gap-1">
                                                <Play className="w-3 h-3" /> Seek
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {/* TAB 3: TRACKS & PEOPLE */}
                    {activeTab === "tracks" && (
                        <div className="space-y-4">
                            <h3 className="text-sm font-bold text-white tracking-wide font-mono mb-4 flex items-center gap-2">
                                <Crosshair className="w-4 h-4 text-emerald-400" />
                                Tracked Persons & Entities
                            </h3>
                            {tracks.length === 0 ? (
                                <div className="text-center py-12 text-xs font-mono text-slate-500">
                                    No persistent tracks recorded yet.
                                </div>
                            ) : (
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                    {tracks.map((t) => (
                                        <div
                                            key={t.id}
                                            onClick={() => handleSeek(t.first_seen_timestamp)}
                                            className="p-4 rounded-xl border border-slate-800 bg-[#070A11] hover:border-emerald-500/40 transition-all cursor-pointer space-y-3"
                                        >
                                            <div className="flex items-center justify-between">
                                                <span className="text-xs font-bold font-mono text-emerald-400 uppercase">
                                                    {t.class_name} #{t.track_number}
                                                </span>
                                                <span className="text-[10px] font-mono text-slate-400 bg-slate-900 px-2 py-0.5 rounded">
                                                    {t.duration.toFixed(1)}s duration
                                                </span>
                                            </div>
                                            <div className="text-xs text-slate-300 font-mono space-y-1">
                                                <div>First Seen: {t.first_seen_timestamp.toFixed(2)}s (Frame #{t.first_seen_frame})</div>
                                                <div>Last Seen: {t.last_seen_timestamp.toFixed(2)}s (Frame #{t.last_seen_frame})</div>
                                                <div>Observations: {t.observation_count} frames</div>
                                            </div>
                                            <button className="w-full py-1.5 rounded text-xs font-mono bg-emerald-950 text-emerald-300 border border-emerald-500/30 hover:bg-emerald-900 transition-colors flex items-center justify-center gap-1">
                                                <Play className="w-3 h-3" /> Jump to Track Start
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {/* TAB 4: KEYFRAMES GRID */}
                    {activeTab === "keyframes" && (
                        <div className="space-y-4">
                            <h3 className="text-sm font-bold text-white tracking-wide font-mono mb-4 flex items-center gap-2">
                                <ImageIcon className="w-4 h-4 text-amber-400" />
                                Extracted Keyframes & Cryptographic Artifact Hashes
                            </h3>
                            {keyframes.length === 0 ? (
                                <div className="text-center py-12 text-xs font-mono text-slate-500">
                                    No keyframes extracted yet.
                                </div>
                            ) : (
                                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                                    {keyframes.map((k) => (
                                        <div
                                            key={k.id}
                                            className="rounded-xl border border-slate-800 bg-[#070A11] overflow-hidden space-y-2 p-3 hover:border-amber-500/40 transition-all"
                                        >
                                            <div
                                                onClick={() => handleSeek(k.timestamp)}
                                                className="relative aspect-video bg-black rounded-lg overflow-hidden cursor-pointer group"
                                            >
                                                <img
                                                    src={k.image_path}
                                                    alt={`Keyframe ${k.frame_number}`}
                                                    className="w-full h-full object-cover group-hover:scale-105 transition-transform"
                                                />
                                                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                                    <Play className="w-8 h-8 text-amber-400" />
                                                </div>
                                            </div>

                                            <div className="space-y-1 font-mono text-xs">
                                                <div className="flex justify-between items-center text-slate-200 font-semibold">
                                                    <span>Frame #{k.frame_number}</span>
                                                    <span className="text-amber-400">{k.timestamp.toFixed(2)}s</span>
                                                </div>
                                                <div className="text-[11px] text-slate-400">
                                                    Reason: <span className="text-slate-200">{k.selection_reason}</span>
                                                </div>
                                                <div className="text-[10px] text-cyan-400 truncate">
                                                    SHA-256: {k.sha256_hash.slice(0, 16)}...
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {/* TAB 5: ACTIVITY INTERVALS */}
                    {activeTab === "activity" && (
                        <div className="space-y-4">
                            <h3 className="text-sm font-bold text-white tracking-wide font-mono mb-4 flex items-center gap-2">
                                <ActivityIcon className="w-4 h-4 text-indigo-400" />
                                Motion Activity Intervals
                            </h3>
                            {activity.length === 0 ? (
                                <div className="text-center py-12 text-xs font-mono text-slate-500">
                                    No activity intervals calculated yet.
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    {activity.map((act) => (
                                        <div
                                            key={act.id}
                                            onClick={() => handleSeek(act.start_time)}
                                            className="p-3 rounded-lg border border-slate-800 bg-[#070A11] flex items-center justify-between font-mono text-xs cursor-pointer hover:border-indigo-500/40 transition-all"
                                        >
                                            <div className="flex items-center gap-3">
                                                <span
                                                    className={`px-2 py-0.5 rounded text-[10px] font-bold ${act.activity_level === "HIGH ACTIVITY"
                                                        ? "bg-red-950 text-red-400 border border-red-800/40"
                                                        : "bg-slate-900 text-slate-400"
                                                        }`}
                                                >
                                                    {act.activity_level}
                                                </span>
                                                <span className="text-slate-200">
                                                    {act.start_time.toFixed(2)}s → {act.end_time.toFixed(2)}s
                                                </span>
                                            </div>

                                            <div className="text-slate-400">
                                                Motion Score: <span className="text-indigo-400 font-semibold">{act.motion_score}</span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {/* TAB 6: CRYPTOGRAPHIC MANIFEST */}
                    {activeTab === "manifest" && (
                        <div className="space-y-4">
                            <div className="flex items-center justify-between mb-2">
                                <h3 className="text-sm font-bold text-white tracking-wide font-mono flex items-center gap-2">
                                    <FileCode className="w-4 h-4 text-cyan-400" />
                                    Machine-Readable Analysis Manifest (JSON)
                                </h3>
                                {manifest && (
                                    <button
                                        onClick={() => {
                                            navigator.clipboard.writeText(JSON.stringify(manifest.manifest_data, null, 2));
                                            setCopiedManifestHash(true);
                                            setTimeout(() => setCopiedManifestHash(false), 2000);
                                        }}
                                        className="px-3 py-1.5 rounded text-xs font-mono bg-cyan-950 text-cyan-300 border border-cyan-500/30 hover:bg-cyan-900 transition-colors flex items-center gap-1"
                                    >
                                        {copiedManifestHash ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                                        {copiedManifestHash ? "COPIED MANIFEST!" : "COPY JSON MANIFEST"}
                                    </button>
                                )}
                            </div>

                            {!manifest ? (
                                <div className="text-center py-12 text-xs font-mono text-slate-500">
                                    No manifest generated yet. Run analysis job to construct cryptographic manifest.
                                </div>
                            ) : (
                                <div className="space-y-3">
                                    <div className="p-3 bg-cyan-950/30 border border-cyan-900/40 rounded-lg text-xs font-mono text-cyan-300">
                                        MANIFEST SHA-256 HASH: <span className="font-bold text-emerald-400">{manifest.sha256_hash}</span>
                                    </div>
                                    <pre className="p-4 rounded-xl border border-slate-800 bg-[#070A11] text-xs font-mono text-slate-300 overflow-x-auto max-h-96">
                                        {JSON.stringify(manifest.manifest_data, null, 2)}
                                    </pre>
                                </div>
                            )}
                        </div>
                    )}

                </div>
            </div>

            {/* Analysis Modal */}
            <AnalysisModal
                isOpen={isModalOpen}
                onClose={() => setIsModalOpen(false)}
                onStartAnalysis={handleStartAnalysis}
                isSubmitting={isSubmittingJob}
            />
        </DashboardLayout>
    );
}
