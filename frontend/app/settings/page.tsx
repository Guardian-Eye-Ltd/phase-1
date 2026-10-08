"use client";

import React, { useEffect, useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { useAuth } from "@/contexts/AuthContext";
import { adminService, ResetPreview, ResetResult } from "@/services/adminService";
import { AlertTriangle, CheckCircle2, Loader2, Lock, RefreshCw, ShieldCheck, Trash2 } from "lucide-react";

const TABLE_LABELS: Record<string, string> = {
    evidence: "Evidence videos",
    analysis_jobs: "Analysis jobs",
    detections: "Detections",
    tracks: "Tracks",
    keyframes: "Keyframes",
    visual_attribute_observations: "Attribute observations",
    track_attribute_aggregates: "Attribute aggregates",
    forensic_documents: "Forensic documents",
    search_queries: "Search history",
};

function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
    return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

export default function SettingsPage() {
    const { user } = useAuth();
    const isAdmin = user?.role?.role_name === "Admin";

    const [preview, setPreview] = useState<ResetPreview | null>(null);
    const [loadingPreview, setLoadingPreview] = useState(false);
    const [confirmText, setConfirmText] = useState("");
    const [resetting, setResetting] = useState(false);
    const [result, setResult] = useState<ResetResult | null>(null);
    const [error, setError] = useState<string | null>(null);

    const loadPreview = async () => {
        setLoadingPreview(true);
        setError(null);
        try {
            setPreview(await adminService.getResetPreview());
        } catch (err: any) {
            setError(err?.response?.data?.detail || "Failed to load reset preview.");
        } finally {
            setLoadingPreview(false);
        }
    };

    useEffect(() => {
        if (isAdmin) loadPreview();
    }, [isAdmin]);

    const phrase = preview?.confirmation_phrase ?? "";
    const phraseMatches = phrase !== "" && confirmText === phrase;
    const canSubmit = !!preview?.can_reset && phraseMatches && !resetting;

    const handleReset = async () => {
        if (!canSubmit) return;
        setResetting(true);
        setError(null);
        setResult(null);
        try {
            const res = await adminService.resetSystem(confirmText);
            setResult(res);
            setConfirmText("");
            await loadPreview();
        } catch (err: any) {
            setError(err?.response?.data?.detail || "System reset failed.");
        } finally {
            setResetting(false);
        }
    };

    const totalFiles = preview?.storage.reduce((n, s) => n + s.files, 0) ?? 0;
    const totalBytes = preview?.storage.reduce((n, s) => n + s.bytes, 0) ?? 0;

    return (
        <DashboardLayout>
            <div className="space-y-6 max-w-4xl">
                <div>
                    <h1 className="text-2xl font-bold text-white tracking-tight">Settings</h1>
                    <p className="text-sm text-gray-400 font-mono mt-0.5">System administration and maintenance.</p>
                </div>

                {!isAdmin ? (
                    <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 flex items-start gap-3">
                        <Lock className="w-5 h-5 text-slate-500 mt-0.5 shrink-0" />
                        <div>
                            <p className="text-slate-200 font-semibold text-sm">Administrator access required</p>
                            <p className="text-slate-400 text-xs mt-1">
                                System maintenance actions are restricted to the Admin role.
                            </p>
                        </div>
                    </div>
                ) : (
                    <section className="border border-rose-500/40 rounded-xl overflow-hidden">
                        <div className="bg-rose-950/40 border-b border-rose-500/30 px-6 py-4 flex items-center justify-between gap-4">
                            <div className="flex items-center gap-2">
                                <AlertTriangle className="w-5 h-5 text-rose-400" />
                                <h2 className="text-base font-bold text-rose-300 tracking-wide">Danger Zone — Reset All Evidence</h2>
                            </div>
                            <button
                                onClick={loadPreview}
                                disabled={loadingPreview}
                                className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-mono border border-slate-700 flex items-center gap-1.5 disabled:opacity-50"
                            >
                                <RefreshCw className={`w-3.5 h-3.5 ${loadingPreview ? "animate-spin" : ""}`} /> Refresh
                            </button>
                        </div>

                        <div className="bg-slate-950/60 p-6 space-y-5">
                            <p className="text-sm text-slate-300 leading-relaxed">
                                Permanently deletes <span className="text-rose-300 font-semibold">every evidence video</span>,
                                all analysis results, keyframes, attribute observations, search history and the
                                vector index. <span className="text-rose-300 font-semibold">This cannot be undone.</span>
                            </p>

                            {preview && (
                                <>
                                    <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                                        {Object.entries(TABLE_LABELS).map(([key, label]) => (
                                            <div key={key} className="bg-slate-900/80 border border-slate-800 rounded-lg px-3 py-2">
                                                <div className="text-[10px] uppercase tracking-wider text-slate-500">{label}</div>
                                                <div className="text-lg font-bold text-white tabular-nums">
                                                    {(preview.tables[key] ?? 0).toLocaleString()}
                                                </div>
                                            </div>
                                        ))}
                                        <div className="bg-slate-900/80 border border-slate-800 rounded-lg px-3 py-2">
                                            <div className="text-[10px] uppercase tracking-wider text-slate-500">Stored files</div>
                                            <div className="text-lg font-bold text-white tabular-nums">
                                                {totalFiles} <span className="text-xs text-slate-400 font-normal">({formatBytes(totalBytes)})</span>
                                            </div>
                                        </div>
                                        <div className="bg-slate-900/80 border border-slate-800 rounded-lg px-3 py-2">
                                            <div className="text-[10px] uppercase tracking-wider text-slate-500">Vector collections</div>
                                            <div className="text-lg font-bold text-white tabular-nums">
                                                {preview.vector_collections ?? "?"}
                                            </div>
                                        </div>
                                    </div>

                                    <div className="flex items-start gap-2 bg-emerald-950/30 border border-emerald-500/20 rounded-lg px-4 py-3 text-xs text-emerald-200">
                                        <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                                        <span>
                                            Preserved: user accounts, roles, cameras, alerts, incidents, and the
                                            <span className="font-semibold"> audit log</span>. The reset itself is recorded
                                            in the audit log for chain of custody.
                                        </span>
                                    </div>

                                    {!preview.can_reset && (
                                        <div className="flex items-start gap-2 bg-amber-950/30 border border-amber-500/30 rounded-lg px-4 py-3 text-xs text-amber-200">
                                            <Loader2 className="w-4 h-4 text-amber-400 shrink-0 mt-0.5 animate-spin" />
                                            <span>
                                                Analysis job(s) #{preview.running_jobs.join(", #")} are running.
                                                Reset is disabled until they finish.
                                            </span>
                                        </div>
                                    )}

                                    <div className="space-y-2">
                                        <label className="block text-xs text-slate-400">
                                            To confirm, type{" "}
                                            <code className="px-1.5 py-0.5 bg-slate-800 text-rose-300 rounded font-mono">{phrase}</code>{" "}
                                            below:
                                        </label>
                                        <div className="flex flex-col sm:flex-row gap-2">
                                            <input
                                                type="text"
                                                value={confirmText}
                                                onChange={(e) => setConfirmText(e.target.value)}
                                                onKeyDown={(e) => e.key === "Enter" && handleReset()}
                                                placeholder={phrase}
                                                autoComplete="off"
                                                spellCheck={false}
                                                className="flex-1 bg-slate-950 border border-slate-700 focus:border-rose-500 focus:ring-1 focus:ring-rose-500 text-white rounded-lg px-3 py-2 text-sm font-mono outline-none"
                                            />
                                            <button
                                                onClick={handleReset}
                                                disabled={!canSubmit}
                                                className="px-5 py-2 bg-rose-600 hover:bg-rose-500 disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed text-white font-semibold rounded-lg text-sm flex items-center justify-center gap-2 transition-colors"
                                            >
                                                {resetting ? (
                                                    <><Loader2 className="w-4 h-4 animate-spin" /> Resetting…</>
                                                ) : (
                                                    <><Trash2 className="w-4 h-4" /> Reset all evidence</>
                                                )}
                                            </button>
                                        </div>
                                    </div>
                                </>
                            )}

                            {error && (
                                <div className="bg-rose-950/40 border border-rose-500/30 text-rose-300 px-4 py-3 rounded-lg text-sm flex items-center gap-2">
                                    <AlertTriangle className="w-4 h-4 shrink-0" /> {error}
                                </div>
                            )}

                            {result && (
                                <div className={`rounded-lg px-4 py-3 text-sm border ${result.errors.length
                                    ? "bg-amber-950/30 border-amber-500/30 text-amber-200"
                                    : "bg-emerald-950/30 border-emerald-500/30 text-emerald-200"}`}>
                                    <div className="flex items-center gap-2 font-semibold">
                                        <CheckCircle2 className="w-4 h-4" />
                                        {result.errors.length ? "Reset completed with warnings" : "Reset complete — fresh start"}
                                    </div>
                                    <p className="text-xs mt-1 opacity-90">
                                        Deleted {result.total_rows_deleted.toLocaleString()} database rows,{" "}
                                        {result.files_deleted} files, and {result.vector_collections_deleted} vector collections.
                                    </p>
                                    {result.errors.length > 0 && (
                                        <ul className="text-xs mt-2 list-disc list-inside space-y-0.5 font-mono">
                                            {result.errors.map((e, i) => <li key={i}>{e}</li>)}
                                        </ul>
                                    )}
                                </div>
                            )}
                        </div>
                    </section>
                )}
            </div>
        </DashboardLayout>
    );
}
