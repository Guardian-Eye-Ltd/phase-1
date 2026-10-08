"use client";

import React, { useEffect, useMemo, useState } from "react";
import { AlertTriangle, CheckCircle2, EyeOff, Loader2, Lock, Play, ScanFace, Search, Upload, UserX } from "lucide-react";
import { AuthImage } from "./AuthImage";
import { faceService, FaceListResponse, FaceSearchResponse, FaceTrackStatus } from "@/services/faceService";

interface FaceSearchPanelProps {
    evidenceId: number;
    onSeek: (timestamp: number) => void;
}

const fmt = (t: number) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;

const TRACK_BADGE: Record<FaceTrackStatus, { label: string; cls: string }> = {
    USABLE_FACE: { label: "Face usable", cls: "bg-emerald-500/10 text-emerald-300 border-emerald-500/30" },
    LOW_QUALITY: { label: "Face too unclear", cls: "bg-amber-500/10 text-amber-300 border-amber-500/30" },
    NO_FACE_VISIBLE: { label: "No face visible", cls: "bg-slate-800 text-slate-400 border-slate-700" },
};

const STATUS_STYLE: Record<FaceSearchResponse["status"], string> = {
    POSSIBLE_MATCH_FOUND: "bg-cyan-950/40 border-cyan-500/40 text-cyan-200",
    NO_MATCH_AMONG_USABLE_FACES: "bg-slate-900/80 border-slate-700 text-slate-200",
    NO_USABLE_FACES: "bg-amber-950/30 border-amber-500/40 text-amber-200",
    FACE_EXTRACTION_NOT_AVAILABLE: "bg-amber-950/30 border-amber-500/40 text-amber-200",
};

export const FaceSearchPanel: React.FC<FaceSearchPanelProps> = ({ evidenceId, onSeek }) => {
    const [faces, setFaces] = useState<FaceListResponse | null>(null);
    const [loadError, setLoadError] = useState<string | null>(null);
    const [forbidden, setForbidden] = useState(false);

    const [photo, setPhoto] = useState<File | null>(null);
    const [threshold, setThreshold] = useState(0.45);
    const [searching, setSearching] = useState(false);
    const [result, setResult] = useState<FaceSearchResponse | null>(null);
    const [searchError, setSearchError] = useState<string | null>(null);

    // Local preview only; the photo leaves the browser only when Search is pressed.
    const preview = useMemo(() => (photo ? URL.createObjectURL(photo) : null), [photo]);
    useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

    useEffect(() => {
        faceService.listFaces(evidenceId)
            .then((data) => { setFaces(data); setThreshold(data.match_threshold_default); })
            .catch((err) => {
                if (err?.response?.status === 403) setForbidden(true);
                else setLoadError(err?.response?.data?.detail || "Could not load faces for this evidence.");
            });
    }, [evidenceId]);

    const runSearch = async () => {
        if (!photo) return;
        setSearching(true);
        setSearchError(null);
        setResult(null);
        try {
            setResult(await faceService.searchByFace(evidenceId, photo, threshold));
        } catch (err: any) {
            setSearchError(err?.response?.data?.detail || "Face search failed.");
        } finally {
            setSearching(false);
        }
    };

    if (forbidden) {
        return (
            <div className="flex items-start gap-3 text-sm text-slate-300">
                <Lock className="w-5 h-5 text-slate-500 shrink-0" />
                Face recognition is restricted to Investigator, Operator and Admin roles.
            </div>
        );
    }
    if (loadError) {
        return <div className="text-sm text-rose-300 flex items-center gap-2"><AlertTriangle className="w-4 h-4" />{loadError}</div>;
    }
    if (!faces) {
        return <div className="text-xs font-mono text-slate-500 flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin" />Loading faces…</div>;
    }

    const ready = faces.face_stage === "COMPLETED";
    const counts = faces.tracks.reduce(
        (acc, t) => ({ ...acc, [t.face_status]: (acc[t.face_status] || 0) + 1 }),
        {} as Record<string, number>,
    );

    return (
        <div className="space-y-6">
            <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                    <h3 className="text-sm font-bold text-white flex items-center gap-2">
                        <ScanFace className="w-4 h-4 text-cyan-400" /> Search by Face
                    </h3>
                    <p className="text-xs text-slate-400 mt-1 max-w-2xl">
                        Upload a photo of one person to check whether they appear in analysis run #{faces.analysis_job_id}.
                        Results are <span className="text-cyan-300 font-semibold">possible matches for human review</span>, not identifications.
                    </p>
                </div>
                <span className="text-[10px] font-mono text-slate-500">
                    {faces.tracks.length} people · {counts.USABLE_FACE || 0} usable · {counts.LOW_QUALITY || 0} unclear · {counts.NO_FACE_VISIBLE || 0} no face
                </span>
            </div>

            {!ready ? (
                <div className="p-4 rounded-lg border border-amber-500/30 bg-amber-950/20 text-sm text-amber-200 flex gap-3">
                    <AlertTriangle className="w-5 h-5 shrink-0 text-amber-400" />
                    <div>
                        <div className="font-semibold">
                            {faces.face_stage === "NOT_RUN"
                                ? "Faces were not extracted for this analysis run."
                                : `Face extraction ${faces.face_stage.toLowerCase()} for this run.`}
                        </div>
                        <div className="text-xs text-amber-200/80 mt-1">
                            {faces.face_stage_detail || "This run predates face recognition."} Re-run analysis on this evidence to enable face search.
                        </div>
                    </div>
                </div>
            ) : (
                <div className="flex flex-col md:flex-row gap-4 p-4 rounded-xl border border-slate-800 bg-[#070A11]">
                    <label className="w-32 h-32 shrink-0 rounded-lg border border-dashed border-slate-600 hover:border-cyan-500 flex items-center justify-center cursor-pointer overflow-hidden bg-slate-950">
                        {preview ? (
                            <img src={preview} alt="Query" className="w-full h-full object-cover" />
                        ) : (
                            <span className="text-[10px] text-slate-500 font-mono text-center px-2 flex flex-col items-center gap-1">
                                <Upload className="w-5 h-5" /> Choose photo
                            </span>
                        )}
                        <input type="file" accept="image/*" className="hidden"
                            onChange={(e) => { setPhoto(e.target.files?.[0] ?? null); setResult(null); setSearchError(null); }} />
                    </label>
                    <div className="flex-1 space-y-3">
                        <p className="text-xs text-slate-400">
                            Use a clear, front-facing photo with exactly one face. The photo is compared in memory and is not stored;
                            only its SHA-256 fingerprint is written to the audit log.
                        </p>
                        <div className="flex items-center gap-3 text-xs font-mono text-slate-400">
                            <span>Match threshold</span>
                            <input type="range" min={0.3} max={0.9} step={0.01} value={threshold}
                                onChange={(e) => setThreshold(Number(e.target.value))} className="flex-1 max-w-xs accent-cyan-500" />
                            <span className="text-cyan-300 w-10">{threshold.toFixed(2)}</span>
                        </div>
                        <button onClick={runSearch} disabled={!photo || searching}
                            className="px-4 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-800 disabled:text-slate-500 text-white text-sm font-semibold flex items-center gap-2">
                            {searching ? <><Loader2 className="w-4 h-4 animate-spin" /> Searching…</> : <><Search className="w-4 h-4" /> Search this video</>}
                        </button>
                    </div>
                </div>
            )}

            {searchError && (
                <div className="p-3 rounded-lg border border-rose-500/30 bg-rose-950/30 text-sm text-rose-300 flex gap-2">
                    <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" /> {searchError}
                </div>
            )}

            {result && (
                <div className="space-y-4">
                    <div className={`p-4 rounded-lg border text-sm ${STATUS_STYLE[result.status]}`}>
                        <div className="font-semibold flex items-center gap-2">
                            {result.status === "POSSIBLE_MATCH_FOUND" ? <CheckCircle2 className="w-4 h-4" /> : <UserX className="w-4 h-4" />}
                            {result.status.replace(/_/g, " ")}
                        </div>
                        <div className="text-xs mt-1 opacity-90">{result.message}</div>
                    </div>

                    {result.matches.map((m) => (
                        <div key={m.track_id} className="p-4 rounded-xl border border-cyan-500/30 bg-slate-950/60 space-y-3">
                            <div className="flex flex-wrap items-center justify-between gap-3">
                                <div>
                                    <div className="text-white font-bold">Person Track #{m.track_id} — possible match</div>
                                    <div className="text-xs font-mono text-slate-400 mt-0.5">
                                        best similarity <span className="text-cyan-300 font-bold">{m.best_similarity.toFixed(3)}</span>
                                        {" "}(threshold {result.threshold.toFixed(2)}) · {m.supporting_observations}/{m.compared_observations} frames above threshold ·
                                        {" "}{fmt(m.first_seen)}–{fmt(m.last_seen)}
                                    </div>
                                </div>
                                <button onClick={() => onSeek(m.first_seen)}
                                    className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 flex items-center gap-1">
                                    <Play className="w-3 h-3 text-cyan-400" /> Jump to {fmt(m.first_seen)}
                                </button>
                            </div>
                            <div className="flex flex-wrap gap-3">
                                {m.observations.map((o) => (
                                    <button key={o.frame_number} onClick={() => onSeek(o.timestamp)} className="text-left group">
                                        <AuthImage src={o.crop_url} alt={`Track ${m.track_id} at ${fmt(o.timestamp)}`}
                                            className="w-20 h-24 object-cover rounded border border-slate-700 group-hover:border-cyan-500" />
                                        <div className="text-[10px] font-mono text-slate-400 mt-1">{fmt(o.timestamp)} · {o.similarity.toFixed(2)}</div>
                                    </button>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            )}

            <div className="space-y-3">
                <h4 className="text-xs font-bold text-slate-300 uppercase tracking-wider">People in this run</h4>
                {faces.tracks.length === 0 && <div className="text-xs text-slate-500 font-mono">No person tracks in this run.</div>}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {faces.tracks.map((t) => (
                        <div key={t.track_id} className="p-3 rounded-lg border border-slate-800 bg-[#070A11] space-y-2">
                            <div className="flex items-center justify-between">
                                <button onClick={() => onSeek(t.first_seen)} className="text-sm text-white font-semibold hover:text-cyan-300">
                                    Track #{t.track_id} <span className="text-[10px] font-mono text-slate-500">{fmt(t.first_seen)}–{fmt(t.last_seen)}</span>
                                </button>
                                <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${TRACK_BADGE[t.face_status].cls}`}>
                                    {TRACK_BADGE[t.face_status].label}
                                </span>
                            </div>
                            {t.faces.length > 0 ? (
                                <div className="flex gap-2">
                                    {t.faces.map((f) => (
                                        <div key={f.frame_number} title={f.quality_reasons.join("; ") || "usable"}>
                                            <AuthImage src={f.crop_url} alt={`Track ${t.track_id}`}
                                                className={`w-14 h-16 object-cover rounded border ${f.quality_status === "USABLE" ? "border-slate-700" : "border-amber-700/60 opacity-60"}`} />
                                        </div>
                                    ))}
                                </div>
                            ) : (
                                <div className="text-[10px] text-slate-500 flex items-center gap-1"><EyeOff className="w-3 h-3" /> face not visible in the sampled frames</div>
                            )}
                            {t.face_status === "LOW_QUALITY" && (
                                <div className="text-[10px] text-amber-300/80">{t.faces[0]?.quality_reasons.join("; ")}</div>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};
