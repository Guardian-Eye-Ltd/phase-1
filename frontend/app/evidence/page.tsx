"use client";

import React, { useEffect, useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { evidenceService } from "@/services/evidenceService";
import { Evidence } from "@/types/evidence";
import Link from "next/link";
import { Film, Search, Filter, UploadCloud, LayoutGrid, List as ListIcon, ShieldCheck } from "lucide-react";

export default function EvidencePage() {
    const [evidenceList, setEvidenceList] = useState<Evidence[]>([]);
    const [total, setTotal] = useState(0);
    const [page, setPage] = useState(1);
    const [statusFilter, setStatusFilter] = useState("ALL");
    const [searchQuery, setSearchQuery] = useState("");
    const [viewMode, setViewMode] = useState<"grid" | "list">("grid");
    const [loading, setLoading] = useState(true);

    const fetchEvidence = async () => {
        setLoading(true);
        try {
            const data = await evidenceService.getEvidenceList(page, 20, statusFilter, searchQuery);
            setEvidenceList(data.items);
            setTotal(data.total);
        } catch (err) {
            console.error("Failed to load evidence library", err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchEvidence();
    }, [page, statusFilter]);

    const handleSearchSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        setPage(1);
        fetchEvidence();
    };

    return (
        <DashboardLayout>
            <div className="space-y-6">
                {/* Header */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div>
                        <h1 className="text-2xl font-bold text-white tracking-tight">Evidence Vault & Library</h1>
                        <p className="text-sm text-gray-400 font-mono mt-0.5">SHA-256 integrity protected CCTV video repository for digital forensics.</p>
                    </div>
                    <Link
                        href="/evidence/upload"
                        className="px-4 py-2 bg-cyberCyan hover:bg-cyan-400 text-slate-950 font-semibold rounded-lg text-xs font-mono flex items-center shrink-0 shadow-lg shadow-cyberCyan/20 transition-all"
                    >
                        <UploadCloud className="w-4 h-4 mr-2" />
                        INGEST NEW EVIDENCE
                    </Link>
                </div>

                {/* Filter Controls Bar */}
                <div className="bg-[#0D1424] border border-slate-800 rounded-xl p-4 flex flex-col md:flex-row items-center justify-between gap-4">
                    <form onSubmit={handleSearchSubmit} className="relative w-full md:w-96">
                        <Search className="w-4 h-4 text-gray-500 absolute left-3 top-3" />
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder="Search filename or SHA-256 hash..."
                            className="w-full bg-[#070A11] border border-slate-700/80 rounded-lg pl-9 pr-4 py-2 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-cyberCyan font-mono"
                        />
                    </form>

                    <div className="flex items-center space-x-3 w-full md:w-auto justify-between md:justify-end">
                        <div className="flex items-center space-x-2">
                            <Filter className="w-3.5 h-3.5 text-gray-400" />
                            <select
                                value={statusFilter}
                                onChange={(e) => {
                                    setStatusFilter(e.target.value);
                                    setPage(1);
                                }}
                                className="bg-[#070A11] border border-slate-700/80 rounded-lg px-3 py-2 text-xs text-gray-200 focus:outline-none focus:border-cyberCyan font-mono"
                            >
                                <option value="ALL">All Statuses</option>
                                <option value="UPLOADED">Uploaded</option>
                                <option value="QUEUED">Queued</option>
                                <option value="PROCESSING">Processing</option>
                                <option value="COMPLETED">Completed</option>
                                <option value="FAILED">Failed</option>
                            </select>
                        </div>

                        <div className="flex items-center bg-[#070A11] border border-slate-800 rounded-lg p-1">
                            <button
                                onClick={() => setViewMode("grid")}
                                className={`p-1.5 rounded ${viewMode === "grid" ? "bg-cyberCyan/15 text-cyberCyan" : "text-gray-500 hover:text-gray-300"}`}
                            >
                                <LayoutGrid className="w-4 h-4" />
                            </button>
                            <button
                                onClick={() => setViewMode("list")}
                                className={`p-1.5 rounded ${viewMode === "list" ? "bg-cyberCyan/15 text-cyberCyan" : "text-gray-500 hover:text-gray-300"}`}
                            >
                                <ListIcon className="w-4 h-4" />
                            </button>
                        </div>
                    </div>
                </div>

                {/* Evidence Content Display */}
                {loading ? (
                    <div className="py-20 text-center text-xs font-mono text-gray-500 animate-pulse">Loading CCTV Evidence Vault...</div>
                ) : evidenceList.length === 0 ? (
                    <div className="py-20 text-center bg-[#0D1424] border border-dashed border-slate-800 rounded-xl">
                        <Film className="w-12 h-12 text-gray-600 mx-auto mb-3" />
                        <h3 className="text-sm font-semibold text-white">No Evidence Found</h3>
                        <p className="text-xs font-mono text-gray-400 mt-1 max-w-sm mx-auto">No CCTV video files match the current query filter or search term.</p>
                    </div>
                ) : viewMode === "grid" ? (
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
                        {evidenceList.map((ev) => (
                            <div key={ev.id} className="bg-[#0D1424] border border-slate-800 rounded-xl overflow-hidden hover:border-cyberCyan/50 transition-all flex flex-col group">
                                <div className="h-40 bg-[#070A11] border-b border-slate-800 flex items-center justify-center relative">
                                    <Film className="w-12 h-12 text-slate-700 group-hover:text-cyberCyan transition-colors" />
                                    <span className={`absolute top-3 right-3 px-2 py-0.5 rounded text-[10px] font-mono uppercase font-bold border ${ev.status === "COMPLETED" ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30" :
                                            ev.status === "PROCESSING" ? "bg-amber-500/10 text-amber-400 border-amber-500/30" :
                                                "bg-slate-800 text-gray-300 border-slate-700"
                                        }`}>
                                        {ev.status}
                                    </span>
                                    <div className="absolute bottom-2 left-3 flex items-center text-[10px] font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded">
                                        <ShieldCheck className="w-3 h-3 mr-1" /> SHA-256 VERIFIED
                                    </div>
                                </div>

                                <div className="p-4 flex-1 flex flex-col justify-between">
                                    <div>
                                        <h3 className="font-semibold text-white text-sm truncate" title={ev.original_filename}>
                                            {ev.original_filename}
                                        </h3>
                                        <p className="text-[11px] font-mono text-cyberCyan/80 mt-1 truncate" title={ev.sha256_hash}>
                                            HASH: {ev.sha256_hash}
                                        </p>
                                        <div className="grid grid-cols-2 gap-2 mt-3 text-[11px] font-mono text-gray-400 bg-slate-900/50 p-2 rounded border border-slate-800">
                                            <div>Size: {(ev.file_size / (1024 * 1024)).toFixed(2)} MB</div>
                                            <div>FPS: {ev.fps || 30}</div>
                                            <div>Res: {ev.resolution || "1920x1080"}</div>
                                            <div>Duration: {ev.duration ? `${ev.duration}s` : "N/A"}</div>
                                        </div>
                                    </div>

                                    <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between text-xs">
                                        <span className="text-[10px] font-mono text-gray-500">{new Date(ev.uploaded_at).toLocaleDateString()}</span>
                                        <Link
                                            href={`/evidence/${ev.id}`}
                                            className="px-3 py-1.5 bg-cyberCyan/10 hover:bg-cyberCyan/20 text-cyberCyan border border-cyberCyan/30 rounded font-mono font-medium text-xs transition-colors"
                                        >
                                            OPEN EVIDENCE
                                        </Link>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="bg-[#0D1424] border border-slate-800 rounded-xl overflow-hidden">
                        <table className="w-full text-left border-collapse text-xs">
                            <thead>
                                <tr className="border-b border-slate-800 text-gray-400 font-mono uppercase text-[10px] bg-slate-900/50">
                                    <th className="py-3 px-4">Filename</th>
                                    <th className="py-3 px-4">SHA-256 Hash</th>
                                    <th className="py-3 px-4">Size / Duration</th>
                                    <th className="py-3 px-4">Status</th>
                                    <th className="py-3 px-4">Uploaded At</th>
                                    <th className="py-3 px-4 text-right">Action</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-slate-800/60">
                                {evidenceList.map((ev) => (
                                    <tr key={ev.id} className="hover:bg-slate-900/50 transition-colors">
                                        <td className="py-3.5 px-4">
                                            <p className="font-semibold text-white max-w-[220px] truncate">{ev.original_filename}</p>
                                            <p className="text-[10px] text-gray-500 font-mono">{ev.mime_type}</p>
                                        </td>
                                        <td className="py-3.5 px-4 font-mono text-cyberCyan text-[11px]">
                                            {ev.sha256_hash.substring(0, 16)}...
                                        </td>
                                        <td className="py-3.5 px-4 font-mono text-gray-300 text-[11px]">
                                            {(ev.file_size / (1024 * 1024)).toFixed(2)} MB • {ev.duration ? `${ev.duration}s` : "N/A"}
                                        </td>
                                        <td className="py-3.5 px-4">
                                            <span className="px-2 py-0.5 bg-slate-800 text-cyberCyan border border-slate-700 rounded text-[10px] font-mono uppercase">
                                                {ev.status}
                                            </span>
                                        </td>
                                        <td className="py-3.5 px-4 font-mono text-gray-400 text-[11px]">
                                            {new Date(ev.uploaded_at).toLocaleString()}
                                        </td>
                                        <td className="py-3.5 px-4 text-right">
                                            <Link
                                                href={`/evidence/${ev.id}`}
                                                className="px-3 py-1.5 bg-cyberCyan/10 hover:bg-cyberCyan/20 text-cyberCyan border border-cyberCyan/30 rounded font-mono text-xs transition-colors"
                                            >
                                                INSPECT
                                            </Link>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </DashboardLayout>
    );
}
