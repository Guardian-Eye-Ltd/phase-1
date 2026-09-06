"use client";

import React, { useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { evidenceService } from "@/services/evidenceService";
import { EvidenceUploadResponse } from "@/types/evidence";
import Link from "next/link";
import { UploadCloud, FileVideo, ShieldCheck, AlertCircle, ArrowLeft, Loader2, CheckCircle2 } from "lucide-react";

export default function EvidenceUploadPage() {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [result, setResult] = useState<EvidenceUploadResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [dragActive, setDragActive] = useState(false);

    const handleFileSelect = (file: File) => {
        setError(null);
        setResult(null);

        // Check format
        const ext = file.name.split(".").pop()?.toLowerCase();
        const allowed = ["mp4", "webm", "avi", "mov", "mkv"];
        if (!ext || !allowed.includes(ext)) {
            setError(`Invalid file format '.${ext}'. Allowed video formats: ${allowed.join(", ")}`);
            return;
        }

        // Check size (500MB limit)
        if (file.size > 500 * 1024 * 1024) {
            setError("File size exceeds 500MB max upload limit.");
            return;
        }

        setSelectedFile(file);
    };

    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    };

    const handleUploadSubmit = async () => {
        if (!selectedFile) return;

        setUploading(true);
        setProgress(0);
        setError(null);

        try {
            const res = await evidenceService.uploadEvidence(selectedFile, (percent) => {
                setProgress(percent);
            });
            setResult(res);
        } catch (err: any) {
            if (err.response && err.response.data && err.response.data.detail) {
                setError(err.response.data.detail);
            } else {
                setError("Failed to upload evidence video. Please check connection and permissions.");
            }
        } finally {
            setUploading(false);
        }
    };

    return (
        <DashboardLayout>
            <div className="max-w-3xl mx-auto space-y-6">
                {/* Header */}
                <div className="flex items-center justify-between">
                    <div>
                        <Link href="/evidence" className="text-xs font-mono text-cyberCyan hover:underline flex items-center mb-2">
                            <ArrowLeft className="w-3.5 h-3.5 mr-1" /> BACK TO EVIDENCE VAULT
                        </Link>
                        <h1 className="text-2xl font-bold text-white tracking-tight">Ingest CCTV Evidence</h1>
                        <p className="text-sm text-gray-400 font-mono mt-0.5">Upload surveillance footage into the SHA-256 chain-of-custody vault.</p>
                    </div>
                </div>

                {/* Upload Card */}
                <div className="bg-[#0D1424] border border-slate-800 rounded-xl p-6 shadow-xl">
                    {error && (
                        <div className="mb-6 p-4 bg-red-500/10 border border-red-500/40 rounded-lg flex items-start text-red-400 text-xs font-mono">
                            <AlertCircle className="w-4 h-4 mr-2.5 shrink-0 mt-0.5" />
                            <div>
                                <p className="font-bold">Ingestion Error</p>
                                <p className="mt-0.5 text-red-300">{error}</p>
                            </div>
                        </div>
                    )}

                    {!result ? (
                        <div className="space-y-6">
                            {/* Drag & Drop Zone */}
                            <div
                                onDragEnter={handleDrag}
                                onDragOver={handleDrag}
                                onDragLeave={handleDrag}
                                onDrop={handleDrop}
                                className={`border-2 border-dashed rounded-xl p-10 text-center transition-all cursor-pointer ${dragActive ? "border-cyberCyan bg-cyberCyan/5" : "border-slate-700/80 hover:border-cyberCyan/60 bg-[#070A11]"
                                    }`}
                                onClick={() => {
                                    const input = document.getElementById("file-input") as HTMLInputElement;
                                    if (input) input.click();
                                }}
                            >
                                <input
                                    id="file-input"
                                    type="file"
                                    accept="video/mp4,video/webm,video/avi,video/quicktime,video/x-matroska"
                                    className="hidden"
                                    onChange={(e) => {
                                        if (e.target.files && e.target.files[0]) {
                                            handleFileSelect(e.target.files[0]);
                                        }
                                    }}
                                />

                                <UploadCloud className="w-14 h-14 text-cyberCyan/70 mx-auto mb-3" />
                                <h3 className="text-base font-semibold text-white">Drag & drop CCTV video evidence here</h3>
                                <p className="text-xs text-gray-400 font-mono mt-1">Supports MP4, WebM, AVI, MOV, MKV up to 500MB</p>
                                <button
                                    type="button"
                                    className="mt-4 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-cyberCyan border border-slate-700 rounded-lg text-xs font-mono transition-colors"
                                >
                                    Browse Files
                                </button>
                            </div>

                            {/* Selected File Details & Upload Action */}
                            {selectedFile && (
                                <div className="bg-slate-900/80 border border-slate-800 rounded-lg p-4 space-y-4">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center space-x-3 truncate">
                                            <FileVideo className="w-8 h-8 text-cyberCyan shrink-0" />
                                            <div className="truncate">
                                                <p className="text-sm font-semibold text-white truncate">{selectedFile.name}</p>
                                                <p className="text-xs font-mono text-gray-400">{(selectedFile.size / (1024 * 1024)).toFixed(2)} MB</p>
                                            </div>
                                        </div>
                                        <button
                                            onClick={() => setSelectedFile(null)}
                                            disabled={uploading}
                                            className="text-xs font-mono text-red-400 hover:underline"
                                        >
                                            Remove
                                        </button>
                                    </div>

                                    {uploading && (
                                        <div className="space-y-2">
                                            <div className="flex justify-between text-xs font-mono text-gray-300">
                                                <span>Streaming & Calculating SHA-256 Hash...</span>
                                                <span>{progress}%</span>
                                            </div>
                                            <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                                                <div
                                                    className="bg-cyberCyan h-full transition-all duration-200"
                                                    style={{ width: `${progress}%` }}
                                                ></div>
                                            </div>
                                        </div>
                                    )}

                                    <button
                                        onClick={handleUploadSubmit}
                                        disabled={uploading}
                                        className="w-full bg-cyberCyan hover:bg-cyan-400 text-slate-950 font-bold py-3 rounded-lg text-xs font-mono tracking-wider flex items-center justify-center shadow-lg shadow-cyberCyan/20 transition-all"
                                    >
                                        {uploading ? (
                                            <>
                                                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                                PROCESSING FORENSIC INGESTION...
                                            </>
                                        ) : (
                                            "INGEST & COMPUTE SHA-256 HASH"
                                        )}
                                    </button>
                                </div>
                            )}
                        </div>
                    ) : (
                        /* Upload Success Result View */
                        <div className="space-y-6">
                            <div className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg flex items-start text-emerald-400 text-xs font-mono">
                                <CheckCircle2 className="w-5 h-5 mr-3 shrink-0 mt-0.5" />
                                <div>
                                    <h3 className="font-bold text-sm text-emerald-400">
                                        {result.is_duplicate ? "Duplicate Evidence Ingested" : "Evidence Successfully Ingested"}
                                    </h3>
                                    <p className="mt-1 text-gray-300">{result.message}</p>
                                </div>
                            </div>

                            {/* Forensic Hash Card */}
                            <div className="bg-[#070A11] border border-cyberCyan/40 rounded-xl p-5 space-y-4">
                                <div className="flex items-center space-x-2 text-cyberCyan text-xs font-mono font-bold uppercase tracking-wider">
                                    <ShieldCheck className="w-4 h-4 text-emerald-400" />
                                    Forensic Chain of Custody Integrity Record
                                </div>

                                <div className="space-y-2 text-xs font-mono">
                                    <div>
                                        <span className="text-gray-500 block text-[10px]">EVIDENCE ID</span>
                                        <span className="text-white font-bold">{result.evidence.id}</span>
                                    </div>
                                    <div>
                                        <span className="text-gray-500 block text-[10px]">ORIGINAL FILENAME</span>
                                        <span className="text-white">{result.evidence.original_filename}</span>
                                    </div>
                                    <div>
                                        <span className="text-gray-500 block text-[10px]">SHA-256 INTEGRITY HASH</span>
                                        <div className="p-2.5 bg-slate-900 rounded border border-slate-800 text-cyberCyan text-[11px] break-all font-mono">
                                            {result.evidence.sha256_hash}
                                        </div>
                                    </div>
                                    <div className="grid grid-cols-2 gap-4 pt-2">
                                        <div>
                                            <span className="text-gray-500 block text-[10px]">ANALYSIS STATUS</span>
                                            <span className="px-2 py-0.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded uppercase text-[10px] font-bold">
                                                {result.evidence.status}
                                            </span>
                                        </div>
                                        <div>
                                            <span className="text-gray-500 block text-[10px]">FILE SIZE</span>
                                            <span className="text-gray-300">{(result.evidence.file_size / (1024 * 1024)).toFixed(2)} MB</span>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* Action Buttons */}
                            <div className="flex items-center space-x-3 pt-2">
                                <Link
                                    href={`/evidence/${result.evidence.id}`}
                                    className="flex-1 bg-cyberCyan hover:bg-cyan-400 text-slate-950 font-bold py-2.5 rounded-lg text-xs font-mono text-center shadow-lg shadow-cyberCyan/20 transition-all"
                                >
                                    INSPECT EVIDENCE DETAILS & PLAYER
                                </Link>
                                <button
                                    onClick={() => {
                                        setResult(null);
                                        setSelectedFile(null);
                                    }}
                                    className="px-4 py-2.5 bg-slate-800 hover:bg-slate-700 text-gray-300 rounded-lg text-xs font-mono border border-slate-700 transition-colors"
                                >
                                    Upload Another File
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </DashboardLayout>
    );
}
