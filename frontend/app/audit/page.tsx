"use client";

import React, { useEffect, useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { auditService } from "@/services/auditService";
import { AuditLog } from "@/types/audit";
import { FileText, Filter, RefreshCw, ShieldCheck, User } from "lucide-react";

export default function AuditPage() {
    const [logs, setLogs] = useState<AuditLog[]>([]);
    const [total, setTotal] = useState(0);
    const [page, setPage] = useState(1);
    const [actionFilter, setActionFilter] = useState("");
    const [loading, setLoading] = useState(true);

    const fetchAuditLogs = async () => {
        setLoading(true);
        try {
            const data = await auditService.getAuditLogs(page, 50, actionFilter || undefined);
            setLogs(data.items);
            setTotal(data.total);
        } catch (err) {
            console.error("Failed to load audit logs", err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchAuditLogs();
    }, [page, actionFilter]);

    return (
        <DashboardLayout>
            <div className="space-y-6">
                {/* Header */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                    <div>
                        <h1 className="text-2xl font-bold text-white tracking-tight">Chain-of-Custody Audit Log</h1>
                        <p className="text-sm text-gray-400 font-mono mt-0.5">Immutable record of all investigator actions, evidence access, and authentications.</p>
                    </div>
                    <button
                        onClick={fetchAuditLogs}
                        className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-gray-300 rounded-lg text-xs font-mono border border-slate-700 flex items-center transition-colors shrink-0"
                    >
                        <RefreshCw className={`w-3.5 h-3.5 mr-1.5 ${loading ? "animate-spin text-cyberCyan" : ""}`} />
                        REFRESH LOGS
                    </button>
                </div>

                {/* Filter Bar */}
                <div className="bg-[#0D1424] border border-slate-800 rounded-xl p-4 flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                        <Filter className="w-4 h-4 text-gray-400" />
                        <span className="text-xs font-mono text-gray-300">Filter by Action:</span>
                        <select
                            value={actionFilter}
                            onChange={(e) => {
                                setActionFilter(e.target.value);
                                setPage(1);
                            }}
                            className="bg-[#070A11] border border-slate-700 rounded-lg px-3 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-cyberCyan"
                        >
                            <option value="">All Forensic Actions</option>
                            <option value="LOGIN">LOGIN</option>
                            <option value="UPLOAD_EVIDENCE">UPLOAD_EVIDENCE</option>
                            <option value="UPLOAD_EVIDENCE_DUPLICATE">UPLOAD_EVIDENCE_DUPLICATE</option>
                            <option value="VIEW_EVIDENCE">VIEW_EVIDENCE</option>
                            <option value="SEARCH_EVIDENCE">SEARCH_EVIDENCE</option>
                        </select>
                    </div>

                    <div className="text-xs font-mono text-gray-400">
                        Total Log Entries: <span className="text-cyberCyan font-bold">{total}</span>
                    </div>
                </div>

                {/* Audit Log Table */}
                <div className="bg-[#0D1424] border border-slate-800 rounded-xl overflow-hidden shadow-xl">
                    {loading ? (
                        <div className="py-20 text-center text-xs font-mono text-gray-500 animate-pulse">Loading audit chain...</div>
                    ) : logs.length === 0 ? (
                        <div className="py-20 text-center text-xs font-mono text-gray-500">No audit entries found matching filter.</div>
                    ) : (
                        <div className="overflow-x-auto">
                            <table className="w-full text-left border-collapse text-xs">
                                <thead>
                                    <tr className="border-b border-slate-800 text-gray-400 font-mono uppercase text-[10px] bg-slate-900/60">
                                        <th className="py-3.5 px-4">Timestamp</th>
                                        <th className="py-3.5 px-4">Investigator / User</th>
                                        <th className="py-3.5 px-4">Action</th>
                                        <th className="py-3.5 px-4">Target Resource</th>
                                        <th className="py-3.5 px-4">Audit Metadata Payload</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-800/60 font-mono">
                                    {logs.map((log) => (
                                        <tr key={log.id} className="hover:bg-slate-900/50 transition-colors">
                                            <td className="py-3 px-4 text-gray-400 text-[11px] whitespace-nowrap">
                                                {new Date(log.timestamp).toLocaleString()}
                                            </td>
                                            <td className="py-3 px-4">
                                                <div className="flex items-center space-x-2">
                                                    <User className="w-3.5 h-3.5 text-cyberCyan" />
                                                    <div>
                                                        <p className="font-semibold text-white text-xs">{log.user_name}</p>
                                                        <p className="text-[10px] text-gray-500">@{log.username}</p>
                                                    </div>
                                                </div>
                                            </td>
                                            <td className="py-3 px-4">
                                                <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${log.action.includes("UPLOAD") ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" :
                                                        log.action === "LOGIN" ? "bg-cyberCyan/10 text-cyberCyan border border-cyberCyan/20" :
                                                            log.action.includes("VIEW") ? "bg-indigo-500/10 text-indigo-300 border border-indigo-500/20" :
                                                                "bg-slate-800 text-gray-300 border border-slate-700"
                                                    }`}>
                                                    {log.action}
                                                </span>
                                            </td>
                                            <td className="py-3 px-4 text-gray-300">
                                                {log.resource_type} {log.resource_id ? `#${log.resource_id}` : ""}
                                            </td>
                                            <td className="py-3 px-4 text-[11px] text-gray-400">
                                                {log.metadata ? (
                                                    <pre className="bg-[#070A11] p-1.5 rounded border border-slate-800 text-[10px] max-w-xs overflow-x-auto text-cyberCyan/90">
                                                        {JSON.stringify(log.metadata, null, 1)}
                                                    </pre>
                                                ) : (
                                                    <span className="text-gray-600">N/A</span>
                                                )}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            </div>
        </DashboardLayout>
    );
}
