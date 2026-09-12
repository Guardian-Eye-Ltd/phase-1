"use client";

import React, { useState, useEffect } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import ProtectedRoute from "@/components/ProtectedRoute";
import { SemanticSearchWorkspace } from "@/components/SemanticSearchWorkspace";
import { evidenceService } from "@/services/evidenceService";
import { Evidence } from "@/types/evidence";
import { Sparkles, FileVideo, Shield, Search } from "lucide-react";

export default function ForensicSearchPage() {
    const [evidenceList, setEvidenceList] = useState<Evidence[]>([]);
    const [selectedEvidenceId, setSelectedEvidenceId] = useState<number | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchEvidence = async () => {
            try {
                const res = await evidenceService.getEvidenceList();
                const items = res.items || [];
                setEvidenceList(items);
                if (items.length > 0) {
                    setSelectedEvidenceId(items[0].id);
                }
            } catch (err) {
                console.error("Failed to load evidence list:", err);
            } finally {
                setLoading(false);
            }
        };
        fetchEvidence();
    }, []);

    return (
        <ProtectedRoute>
            <DashboardLayout>
                <div className="space-y-6">
                    {/* Header */}
                    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                        <div>
                            <div className="flex items-center gap-2">
                                <Sparkles className="w-6 h-6 text-cyan-400" />
                                <h1 className="text-2xl font-bold text-white tracking-wide">
                                    Semantic Forensic Search Workspace
                                </h1>
                            </div>
                            <p className="text-sm text-slate-400 mt-1">
                                Natural language evidence retrieval grounded strictly in verified CCTV observations.
                            </p>
                        </div>

                        {/* Evidence File Selector */}
                        {evidenceList.length > 0 && (
                            <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 rounded-lg p-2">
                                <FileVideo className="w-4 h-4 text-cyan-400" />
                                <select
                                    value={selectedEvidenceId || ""}
                                    onChange={(e) => setSelectedEvidenceId(Number(e.target.value))}
                                    className="bg-slate-950 text-white text-xs font-semibold px-3 py-1.5 rounded border border-slate-700 focus:outline-none focus:border-cyan-500"
                                >
                                    {evidenceList.map((ev) => (
                                        <option key={ev.id} value={ev.id}>
                                            #{ev.id} — {ev.original_filename} ({ev.duration ? `${ev.duration.toFixed(0)}s` : "Video"})
                                        </option>
                                    ))}
                                </select>
                            </div>
                        )}
                    </div>

                    {/* Main Search Workspace Component */}
                    {loading ? (
                        <div className="p-12 text-center text-slate-400 text-sm animate-pulse">
                            Loading forensic evidence vault...
                        </div>
                    ) : selectedEvidenceId ? (
                        <SemanticSearchWorkspace evidenceId={selectedEvidenceId} />
                    ) : (
                        <div className="p-12 text-center bg-slate-900/60 border border-slate-800 rounded-xl space-y-3">
                            <Shield className="w-10 h-10 text-slate-600 mx-auto" />
                            <h3 className="text-base font-semibold text-slate-300">No Evidence Files Uploaded</h3>
                            <p className="text-xs text-slate-500">
                                Please upload CCTV footage to the Evidence Vault before performing semantic queries.
                            </p>
                        </div>
                    )}
                </div>
            </DashboardLayout>
        </ProtectedRoute>
    );
}
