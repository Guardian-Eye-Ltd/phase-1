"use client";

import React, { useEffect, useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { evidenceService } from "@/services/evidenceService";
import { auditService } from "@/services/auditService";
import { Evidence } from "@/types/evidence";
import { AuditLog } from "@/types/audit";
import Link from "next/link";
import { Film, CheckCircle2, Clock, AlertTriangle, UploadCloud, ArrowRight, Shield, RefreshCw } from "lucide-react";

export default function DashboardPage() {
  const [evidenceList, setEvidenceList] = useState<Evidence[]>([]);
  const [totalEvidence, setTotalEvidence] = useState(0);
  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchDashboardData = async () => {
    setLoading(true);
    try {
      const [evData, auditData] = await Promise.all([
        evidenceService.getEvidenceList(1, 10),
        auditService.getAuditLogs(1, 8)
      ]);

      setEvidenceList(evData.items);
      setTotalEvidence(evData.total);
      setAuditLogs(auditData.items);
    } catch (err) {
      console.error("Failed to load dashboard data", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const stats = {
    total: totalEvidence,
    processing: evidenceList.filter(e => e.status === "PROCESSING" || e.status === "QUEUED").length,
    completed: evidenceList.filter(e => e.status === "COMPLETED" || e.status === "UPLOADED").length,
    failed: evidenceList.filter(e => e.status === "FAILED").length,
  };

  return (
    <DashboardLayout>
      <div className="space-y-6">
        {/* Page Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white tracking-tight">Forensic Investigation Dashboard</h1>
            <p className="text-sm text-gray-400 font-mono mt-0.5">Overview of CCTV evidence vault, analysis queues, and chain-of-custody logs.</p>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={fetchDashboardData}
              className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-gray-300 rounded-lg text-xs font-mono border border-slate-700 flex items-center transition-colors"
            >
              <RefreshCw className={`w-3.5 h-3.5 mr-1.5 ${loading ? "animate-spin text-cyberCyan" : ""}`} />
              REFRESH
            </button>
            <Link
              href="/evidence/upload"
              className="px-4 py-2 bg-cyberCyan hover:bg-cyan-400 text-slate-950 font-semibold rounded-lg text-xs font-mono flex items-center shadow-lg shadow-cyberCyan/20 transition-all"
            >
              <UploadCloud className="w-4 h-4 mr-2" />
              INGEST EVIDENCE
            </Link>
          </div>
        </div>

        {/* Metric Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-[#0D1424] border border-slate-800 rounded-xl p-5 relative overflow-hidden">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-mono text-gray-400 uppercase tracking-wider">Total Ingested Evidence</p>
                <p className="text-3xl font-bold text-white font-mono mt-2">{stats.total}</p>
              </div>
              <div className="p-3 bg-cyberCyan/10 rounded-lg border border-cyberCyan/20 text-cyberCyan">
                <Film className="w-6 h-6" />
              </div>
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-center text-[11px] text-gray-400 font-mono">
              <span className="text-cyberCyan font-semibold mr-1">Protected</span> in SHA-256 Vault
            </div>
          </div>

          <div className="bg-[#0D1424] border border-slate-800 rounded-xl p-5 relative overflow-hidden">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-mono text-gray-400 uppercase tracking-wider">In Analysis Queue</p>
                <p className="text-3xl font-bold text-amber-400 font-mono mt-2">{stats.processing}</p>
              </div>
              <div className="p-3 bg-amber-500/10 rounded-lg border border-amber-500/20 text-amber-400">
                <Clock className="w-6 h-6" />
              </div>
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-center text-[11px] text-gray-400 font-mono">
              <span className="text-amber-400 font-semibold mr-1">Queued / Processing</span> background jobs
            </div>
          </div>

          <div className="bg-[#0D1424] border border-slate-800 rounded-xl p-5 relative overflow-hidden">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-mono text-gray-400 uppercase tracking-wider">Analysis Ready</p>
                <p className="text-3xl font-bold text-emerald-400 font-mono mt-2">{stats.completed}</p>
              </div>
              <div className="p-3 bg-emerald-500/10 rounded-lg border border-emerald-500/20 text-emerald-400">
                <CheckCircle2 className="w-6 h-6" />
              </div>
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-center text-[11px] text-gray-400 font-mono">
              <span className="text-emerald-400 font-semibold mr-1">Ready</span> for AI Modules
            </div>
          </div>

          <div className="bg-[#0D1424] border border-slate-800 rounded-xl p-5 relative overflow-hidden">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-mono text-gray-400 uppercase tracking-wider">Failed Pipeline Jobs</p>
                <p className="text-3xl font-bold text-rose-400 font-mono mt-2">{stats.failed}</p>
              </div>
              <div className="p-3 bg-rose-500/10 rounded-lg border border-rose-500/20 text-rose-400">
                <AlertTriangle className="w-6 h-6" />
              </div>
            </div>
            <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-center text-[11px] text-gray-400 font-mono">
              <span className="text-rose-400 font-semibold mr-1">0 Critical</span> File corruptions
            </div>
          </div>
        </div>

        {/* Dashboard Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent Uploaded Evidence (2 Columns) */}
          <div className="lg:col-span-2 bg-[#0D1424] border border-slate-800 rounded-xl p-5 flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Film className="w-5 h-5 text-cyberCyan" />
                <h2 className="font-semibold text-white">Recent Forensic Evidence</h2>
              </div>
              <Link href="/evidence" className="text-xs font-mono text-cyberCyan hover:underline flex items-center">
                VIEW VAULT <ArrowRight className="w-3.5 h-3.5 ml-1" />
              </Link>
            </div>

            {loading ? (
              <div className="py-12 text-center text-xs font-mono text-gray-500 animate-pulse">Loading evidence vault...</div>
            ) : evidenceList.length === 0 ? (
              <div className="py-12 text-center border border-dashed border-slate-800 rounded-lg">
                <Film className="w-8 h-8 text-gray-600 mx-auto mb-2" />
                <p className="text-xs font-mono text-gray-400">No CCTV evidence files uploaded yet.</p>
                <Link href="/evidence/upload" className="mt-3 inline-block px-3 py-1.5 bg-cyberCyan/10 text-cyberCyan text-xs font-mono border border-cyberCyan/30 rounded hover:bg-cyberCyan/20">
                  Ingest First Evidence Video
                </Link>
              </div>
            ) : (
              <div className="overflow-x-auto flex-1">
                <table className="w-full text-left border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-gray-400 font-mono uppercase text-[10px]">
                      <th className="py-2.5 px-3">Evidence File</th>
                      <th className="py-2.5 px-3">SHA-256 Hash</th>
                      <th className="py-2.5 px-3">Status</th>
                      <th className="py-2.5 px-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60">
                    {evidenceList.map((ev) => (
                      <tr key={ev.id} className="hover:bg-slate-900/50 transition-colors">
                        <td className="py-3 px-3">
                          <p className="font-medium text-white max-w-[200px] truncate">{ev.original_filename}</p>
                          <p className="text-[10px] text-gray-500 font-mono">{ev.resolution || "1920x1080"} • {ev.fps || 30} FPS</p>
                        </td>
                        <td className="py-3 px-3 font-mono text-cyberCyan/80 text-[11px]">
                          {ev.sha256_hash.substring(0, 12)}...
                        </td>
                        <td className="py-3 px-3">
                          <span className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-mono uppercase font-semibold ${ev.status === "COMPLETED" ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20" :
                              ev.status === "PROCESSING" ? "bg-amber-500/10 text-amber-400 border border-amber-500/20" :
                                ev.status === "QUEUED" ? "bg-blue-500/10 text-blue-400 border border-blue-500/20" :
                                  "bg-slate-800 text-gray-300 border border-slate-700"
                            }`}>
                            {ev.status}
                          </span>
                        </td>
                        <td className="py-3 px-3 text-right">
                          <Link
                            href={`/evidence/${ev.id}`}
                            className="px-2.5 py-1 bg-cyberCyan/10 hover:bg-cyberCyan/20 text-cyberCyan border border-cyberCyan/30 rounded text-[11px] font-mono transition-colors"
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

          {/* Investigator Activity Feed (1 Column) */}
          <div className="bg-[#0D1424] border border-slate-800 rounded-xl p-5 flex flex-col">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <Shield className="w-5 h-5 text-cyberCyan" />
                <h2 className="font-semibold text-white">Chain of Custody</h2>
              </div>
              <Link href="/audit" className="text-xs font-mono text-cyberCyan hover:underline">
                AUDIT LOG
              </Link>
            </div>

            {loading ? (
              <div className="py-12 text-center text-xs font-mono text-gray-500 animate-pulse">Loading activity logs...</div>
            ) : auditLogs.length === 0 ? (
              <div className="py-12 text-center text-xs font-mono text-gray-500">No activity recorded yet.</div>
            ) : (
              <div className="space-y-3 flex-1 overflow-y-auto max-h-[360px] pr-1">
                {auditLogs.map((log) => (
                  <div key={log.id} className="p-2.5 rounded-lg bg-slate-900/60 border border-slate-800 text-xs">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-cyberCyan font-semibold">{log.action}</span>
                      <span className="text-[10px] font-mono text-gray-500">{new Date(log.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <p className="text-gray-300 text-[11px] mt-1">
                      By <span className="text-white font-medium">{log.user_name}</span> ({log.username})
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
