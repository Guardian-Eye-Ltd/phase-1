"use client";

import React, { useEffect, useState } from "react";
import {
    Bot, Search, ShieldCheck, CheckCircle2, AlertTriangle, XCircle,
    Play, Clock, Download, FileText, Cpu, Sparkles, ArrowRight, Layers,
    Hash
} from "lucide-react";
import { semanticService, AgentInvestigationResponse } from "../services/semanticService";

interface InvestigationWorkspaceProps {
    evidenceId: number;
    onSeekVideo?: (timestamp: number, trackId?: number) => void;
}

const SAMPLE_QUERIES = [
    "Find person wearing blue shirt carrying a bag",
    "Locate loitering events and sudden movement",
    "Find red vehicle near person track",
    "Show timeline of person carrying backpack"
];

export const InvestigationWorkspace: React.FC<InvestigationWorkspaceProps> = ({
    evidenceId,
    onSeekVideo
}) => {
    const [query, setQuery] = useState("");
    const [loading, setLoading] = useState(false);
    const [activeTab, setActiveTab] = useState<"result" | "findings" | "events" | "report">("findings");
    const [response, setResponse] = useState<AgentInvestigationResponse | null>(null);
    const [error, setError] = useState<string | null>(null);

    const intentType = response?.intent?.type;
    const isCountIntent = intentType === "COUNT" || intentType === "FRAME_COUNT";

    useEffect(() => {
        if (!response) return;
        // For deterministic aggregation intents, land on the Result tab first.
        if (isCountIntent) setActiveTab("result");
        else if (intentType === "EVENT_SEARCH") setActiveTab("events");
        else setActiveTab("findings");
    }, [response, isCountIntent, intentType]);

    const handleRunInvestigation = async (queryToRun?: string) => {
        const targetQuery = queryToRun || query;
        if (!targetQuery.trim()) return;

        setLoading(true);
        setError(null);
        try {
            const res = await semanticService.executeAgentInvestigation(evidenceId, targetQuery);
            setResponse(res);
        } catch (err: any) {
            setError(err?.response?.data?.detail || "Failed to execute agentic investigation.");
        } finally {
            setLoading(false);
        }
    };

    const downloadReport = () => {
        if (!response?.report_markdown) return;
        const blob = new Blob([response.report_markdown], { type: "text/markdown" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `GuardianEye_Forensic_Report_${response.investigation_id}.md`;
        a.click();
    };

    const renderBadge = (status: string) => {
        switch (status) {
            case "SUPPORTED":
            case "VERIFIED":
                return (
                    <span className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                        <CheckCircle2 className="w-3.5 h-3.5" /> SUPPORTED
                    </span>
                );
            case "PARTIALLY_SUPPORTED":
                return (
                    <span className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/30">
                        <AlertTriangle className="w-3.5 h-3.5" /> PARTIALLY SUPPORTED
                    </span>
                );
            default:
                return (
                    <span className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-0.5 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/30">
                        <XCircle className="w-3.5 h-3.5" /> UNVERIFIED
                    </span>
                );
        }
    };

    return (
        <div className="space-y-6">
            {/* Header Banner */}
            <div className="bg-gradient-to-r from-slate-900 via-cyan-950/40 to-slate-900 border border-cyan-500/20 rounded-2xl p-6 shadow-2xl relative overflow-hidden">
                <div className="absolute top-0 right-0 p-8 opacity-10 pointer-events-none">
                    <Bot className="w-64 h-64 text-cyan-400" />
                </div>
                <div className="relative z-10 max-w-3xl">
                    <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/30 text-xs font-semibold uppercase tracking-wider mb-3">
                        <Sparkles className="w-3.5 h-3.5" /> GuardianEye 2.0 Multi-Agent Engine
                    </div>
                    <h2 className="text-2xl font-bold text-white mb-2">
                        Agentic Forensic Investigation Workspace
                    </h2>
                    <p className="text-slate-400 text-sm leading-relaxed mb-6">
                        Enter natural language investigation requests. Our multi-agent system decomposes intent, executes controlled evidence queries over verified database tracks, correlates spatial events, and generates verified forensic reports.
                    </p>

                    {/* Input Box */}
                    <div className="flex gap-2">
                        <div className="relative flex-1">
                            <Search className="absolute left-4 top-3.5 w-5 h-5 text-slate-400" />
                            <input
                                type="text"
                                value={query}
                                onChange={(e) => setQuery(e.target.value)}
                                onKeyDown={(e) => e.key === "Enter" && handleRunInvestigation()}
                                placeholder="e.g. Find person in blue shirt carrying a bag near a car..."
                                className="w-full bg-slate-950/80 border border-slate-700/80 focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 text-white rounded-xl pl-12 pr-4 py-3 text-sm placeholder-slate-500 transition-all outline-none"
                            />
                        </div>
                        <button
                            onClick={() => handleRunInvestigation()}
                            disabled={loading || !query.trim()}
                            className="px-6 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 disabled:opacity-50 text-white font-semibold rounded-xl text-sm transition-all shadow-lg shadow-cyan-500/25 flex items-center gap-2"
                        >
                            {loading ? (
                                <>
                                    <Cpu className="w-4 h-4 animate-spin text-white" /> Investigating...
                                </>
                            ) : (
                                <>
                                    <Bot className="w-4 h-4" /> Run Investigation
                                </>
                            )}
                        </button>
                    </div>

                    {/* Sample Queries */}
                    <div className="flex flex-wrap items-center gap-2 mt-4">
                        <span className="text-xs text-slate-500 font-medium">Sample Queries:</span>
                        {SAMPLE_QUERIES.map((sq, i) => (
                            <button
                                key={i}
                                onClick={() => {
                                    setQuery(sq);
                                    handleRunInvestigation(sq);
                                }}
                                className="text-xs bg-slate-800/60 hover:bg-cyan-950/60 text-slate-300 hover:text-cyan-300 border border-slate-700/60 hover:border-cyan-500/40 px-3 py-1 rounded-lg transition-all"
                            >
                                {sq}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {error && (
                <div className="bg-rose-950/40 border border-rose-500/30 text-rose-300 p-4 rounded-xl text-sm flex items-center gap-3">
                    <AlertTriangle className="w-5 h-5 flex-shrink-0 text-rose-400" />
                    <span>{error}</span>
                </div>
            )}

            {/* Response Workspace */}
            {response && (
                <div className="space-y-6 animate-in fade-in duration-300">
                    {/* Agent Progress Bar */}
                    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4">
                        <div className="flex items-center justify-between mb-3">
                            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-2">
                                <Layers className="w-4 h-4 text-cyan-400" /> Multi-Agent Execution Pipeline
                            </span>
                            <span className="text-xs text-slate-400 font-mono">
                                ID: {response.investigation_id} | Execution Time: {response.execution_time_ms}ms
                            </span>
                        </div>
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                            {response.progress.map((step, idx) => (
                                <div
                                    key={idx}
                                    className="bg-slate-950/60 border border-slate-800 rounded-lg p-2.5 text-xs flex flex-col justify-between"
                                >
                                    <div className="flex items-center justify-between mb-1">
                                        <span className="font-semibold text-slate-300 uppercase text-[10px]">{step.stage}</span>
                                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
                                    </div>
                                    <p className="text-[11px] text-slate-400 line-clamp-2">{step.message}</p>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Navigation Tabs */}
                    <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                        <div className="flex gap-2">
                            {isCountIntent && (
                                <button
                                    onClick={() => setActiveTab("result")}
                                    className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all flex items-center gap-2 ${activeTab === "result"
                                            ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/30"
                                            : "text-slate-400 hover:text-slate-200"
                                        }`}
                                >
                                    <Hash className="w-4 h-4" /> Count Result
                                </button>
                            )}
                            <button
                                onClick={() => setActiveTab("findings")}
                                className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all flex items-center gap-2 ${activeTab === "findings"
                                        ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/30"
                                        : "text-slate-400 hover:text-slate-200"
                                    }`}
                            >
                                <ShieldCheck className="w-4 h-4" /> Verified Findings ({response.findings.length})
                            </button>
                            <button
                                onClick={() => setActiveTab("events")}
                                className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all flex items-center gap-2 ${activeTab === "events"
                                        ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/30"
                                        : "text-slate-400 hover:text-slate-200"
                                    }`}
                            >
                                <Clock className="w-4 h-4" /> Correlated Events ({response.events.length})
                            </button>
                            <button
                                onClick={() => setActiveTab("report")}
                                className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all flex items-center gap-2 ${activeTab === "report"
                                        ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/30"
                                        : "text-slate-400 hover:text-slate-200"
                                    }`}
                            >
                                <FileText className="w-4 h-4" /> Digital Forensics Report
                            </button>
                        </div>

                        {activeTab === "report" && (
                            <button
                                onClick={downloadReport}
                                className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg transition-all flex items-center gap-2 border border-slate-700"
                            >
                                <Download className="w-3.5 h-3.5 text-cyan-400" /> Export Report (.md)
                            </button>
                        )}
                    </div>

                    {/* TAB 0: Deterministic Count Result (COUNT / FRAME_COUNT) */}
                    {activeTab === "result" && isCountIntent && (
                        <div className="space-y-4">
                            {(() => {
                                const r = response.result || {};
                                const intent = response.intent!;
                                const total = (r.total ?? r.count ?? 0) as number;
                                const breakdown = r.breakdown || {};
                                const breakdownEntries = Object.entries(breakdown).sort(
                                    (a, b) => (b[1] as number) - (a[1] as number)
                                );
                                const headerLabel = intent.type === "FRAME_COUNT"
                                    ? `Active ${intent.entity ?? "entities"} at t=${(intent.timestamp ?? 0).toFixed(2)}s`
                                    : `Unique ${intent.entity ?? "entities"} detected`;
                                const verOk = response.verification?.status === "SUPPORTED";

                                return (
                                    <>
                                        {/* Headline card */}
                                        <div className="bg-gradient-to-br from-cyan-950/50 via-slate-900 to-slate-900 border border-cyan-500/30 rounded-2xl p-8 shadow-2xl">
                                            <div className="flex items-center justify-between gap-4 flex-wrap">
                                                <div>
                                                    <p className="text-xs uppercase tracking-widest text-cyan-400 font-semibold mb-2">
                                                        {headerLabel}
                                                    </p>
                                                    <div className="flex items-baseline gap-3">
                                                        <span className="text-6xl font-black text-white tabular-nums">
                                                            {total}
                                                        </span>
                                                        <span className="text-slate-400 text-lg capitalize">
                                                            {intent.entity}{total === 1 ? "" : "s"}
                                                        </span>
                                                    </div>
                                                    <p className="text-xs text-slate-500 mt-3 font-mono">
                                                        aggregation: {intent.aggregation} · scope: {intent.scope ?? "ENTIRE_VIDEO"}
                                                        {response.analysis_job_id != null && ` · job #${response.analysis_job_id}`}
                                                    </p>
                                                </div>
                                                <div>
                                                    {verOk ? (
                                                        <span className="inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                                                            <CheckCircle2 className="w-4 h-4" /> VERIFIED (DETERMINISTIC)
                                                        </span>
                                                    ) : (
                                                        <span className="inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/30">
                                                            <AlertTriangle className="w-4 h-4" /> {response.verification?.status ?? "UNVERIFIED"}
                                                        </span>
                                                    )}
                                                </div>
                                            </div>
                                        </div>

                                        {/* Breakdown table */}
                                        {breakdownEntries.length > 0 && (
                                            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5">
                                                <h4 className="text-sm font-bold text-slate-200 mb-3 flex items-center gap-2">
                                                    <Layers className="w-4 h-4 text-cyan-400" /> Breakdown by Class
                                                </h4>
                                                <div className="divide-y divide-slate-800">
                                                    {breakdownEntries.map(([cls, n]) => (
                                                        <div key={cls} className="flex items-center justify-between py-2 text-sm">
                                                            <span className="text-slate-300 capitalize">{cls}</span>
                                                            <span className="text-white font-mono font-bold tabular-nums">{n as number}</span>
                                                        </div>
                                                    ))}
                                                    <div className="flex items-center justify-between py-2 pt-3 text-sm">
                                                        <span className="text-cyan-400 font-semibold uppercase text-xs tracking-wider">Total</span>
                                                        <span className="text-cyan-400 font-mono font-bold tabular-nums text-base">{total}</span>
                                                    </div>
                                                </div>
                                            </div>
                                        )}

                                        {/* Track IDs */}
                                        {r.track_ids && r.track_ids.length > 0 && (
                                            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                                                <p className="text-xs text-slate-400 mb-2 font-semibold uppercase tracking-wider">
                                                    Counted Track IDs ({r.track_ids.length})
                                                </p>
                                                <div className="flex flex-wrap gap-1.5">
                                                    {r.track_ids.map((tid) => (
                                                        <span key={tid} className="text-xs font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                                                            #{tid}
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>
                                        )}

                                        {/* Methodology note */}
                                        <div className="bg-slate-950/60 border border-slate-800 rounded-xl p-4 text-xs text-slate-500 leading-relaxed">
                                            Each tracked entity is counted exactly once regardless of how many frames it appears in.
                                            Per-frame visual counts (VLM) are <span className="text-slate-300 font-semibold">not summed</span>.
                                            Event-engine findings (loitering, acceleration, etc.) are intentionally omitted from count queries.
                                        </div>
                                    </>
                                );
                            })()}
                        </div>
                    )}

                    {/* TAB 1: Verified Findings */}
                    {activeTab === "findings" && (
                        <div className="space-y-4">
                            {response.findings.length === 0 ? (
                                <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-8 text-center text-slate-400">
                                    <AlertTriangle className="w-8 h-8 text-amber-400 mx-auto mb-2" />
                                    {isCountIntent ? (
                                        <>
                                            <p className="font-semibold text-slate-300">COUNT INTENT — see the Count Result tab</p>
                                            <p className="text-xs text-slate-500 mt-1">
                                                Deterministic aggregation queries don&apos;t produce verified-finding cards.
                                                The authoritative total is shown under <span className="text-cyan-400">Count Result</span>.
                                            </p>
                                        </>
                                    ) : (
                                        <>
                                            <p className="font-semibold text-slate-300">NO SUPPORTED EVIDENCE FOUND</p>
                                            <p className="text-xs text-slate-500 mt-1">
                                                The requested entities or attributes did not meet the minimum verification threshold in this video.
                                            </p>
                                        </>
                                    )}
                                </div>
                            ) : (
                                response.findings.map((item, idx) => (
                                    <div
                                        key={idx}
                                        className="bg-slate-900/90 border border-slate-800 hover:border-cyan-500/40 rounded-xl p-5 transition-all shadow-lg space-y-3"
                                    >
                                        <div className="flex items-start justify-between gap-4">
                                            <div className="space-y-1">
                                                <div className="flex items-center gap-3">
                                                    <h4 className="font-bold text-slate-100 text-base">{item.title}</h4>
                                                    {renderBadge(item.verification_status)}
                                                </div>
                                                <p className="text-slate-300 text-sm leading-relaxed">{item.summary}</p>
                                            </div>

                                            {item.start_time !== undefined && (
                                                <button
                                                    onClick={() => onSeekVideo && onSeekVideo(item.start_time!, item.track_id)}
                                                    className="px-3 py-2 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 rounded-lg text-xs font-semibold transition-all flex items-center gap-1.5 flex-shrink-0"
                                                >
                                                    <Play className="w-3.5 h-3.5" /> Jump to {item.start_time.toFixed(2)}s
                                                </button>
                                            )}
                                        </div>

                                        <div className="bg-slate-950/60 border border-slate-800/80 rounded-lg p-3 text-xs text-slate-400 space-y-1 font-mono">
                                            <div className="flex justify-between">
                                                <span>Verification Detail:</span>
                                                <span className="text-slate-200">{item.confidence_reason}</span>
                                            </div>
                                            {item.track_id && (
                                                <div className="flex justify-between">
                                                    <span>Track Number:</span>
                                                    <span className="text-cyan-400 font-bold">Track #{item.track_id}</span>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                ))
                            )}
                        </div>
                    )}

                    {/* TAB 2: Correlated Events */}
                    {activeTab === "events" && (
                        <div className="space-y-3">
                            {response.events.length === 0 ? (
                                <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-8 text-center text-slate-400">
                                    <Clock className="w-8 h-8 text-cyan-400 mx-auto mb-2" />
                                    <p>No loitering or proximity events detected for query timeframe.</p>
                                </div>
                            ) : (
                                response.events.map((ev, idx) => (
                                    <div key={idx} className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 flex items-center justify-between">
                                        <div className="space-y-1">
                                            <div className="flex items-center gap-2">
                                                <span className="text-xs font-bold px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                                                    {ev.event_type}
                                                </span>
                                                <h5 className="font-semibold text-slate-200 text-sm">{ev.title}</h5>
                                            </div>
                                            <p className="text-slate-400 text-xs">{ev.description}</p>
                                        </div>

                                        <button
                                            onClick={() => onSeekVideo && onSeekVideo(ev.start_time, ev.involved_track_ids?.[0])}
                                            className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg text-xs font-semibold flex items-center gap-1 transition-all"
                                        >
                                            <Play className="w-3 h-3 text-cyan-400" /> {ev.start_time.toFixed(1)}s
                                        </button>
                                    </div>
                                ))
                            )}
                        </div>
                    )}

                    {/* TAB 3: Digital Forensics Report */}
                    {activeTab === "report" && (
                        <div className="bg-slate-950/90 border border-slate-800 rounded-2xl p-8 text-slate-200 text-sm leading-relaxed font-mono whitespace-pre-wrap shadow-2xl">
                            {response.report_markdown}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};
