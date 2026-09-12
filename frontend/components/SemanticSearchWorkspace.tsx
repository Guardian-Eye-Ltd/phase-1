"use client";

import React, { useState, useEffect } from "react";
import {
    Search, Shield, AlertTriangle, CheckCircle, Sparkles, Clock,
    HelpCircle, ChevronDown, ChevronUp, Play, Eye, FileText, Cpu, Database, RefreshCw
} from "lucide-react";
import { semanticService, SearchResponse, SearchResultItem, SemanticStatus } from "../services/semanticService";

interface SemanticSearchWorkspaceProps {
    evidenceId: number;
    onJumpToTimestamp?: (seconds: number) => void;
}

const EXAMPLE_QUERIES = [
    "Find a person carrying a backpack",
    "Show people near vehicles",
    "What happened around 01:30?",
    "Show Track 17",
    "Find a person wearing a red shirt",
    "Show all people near the entrance"
];

export const SemanticSearchWorkspace: React.FC<SemanticSearchWorkspaceProps> = ({
    evidenceId,
    onJumpToTimestamp
}) => {
    const [query, setQuery] = useState("");
    const [isSearching, setIsSearching] = useState(false);
    const [searchResponse, setSearchResponse] = useState<SearchResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [expandedExplanation, setExpandedExplanation] = useState<number | null>(null);

    // Indexing State
    const [indexingStatus, setIndexingStatus] = useState<SemanticStatus | null>(null);
    const [isTriggeringIndex, setIsTriggeringIndex] = useState(false);

    const checkStatus = async () => {
        try {
            const status = await semanticService.getIndexingStatus(evidenceId);
            setIndexingStatus(status);
        } catch (err) {
            console.error("Failed to fetch indexing status:", err);
        }
    };

    useEffect(() => {
        if (evidenceId) {
            checkStatus();
        }
    }, [evidenceId]);

    const handleTriggerIndex = async () => {
        setIsTriggeringIndex(true);
        try {
            const status = await semanticService.triggerIndexing(evidenceId);
            setIndexingStatus(status);
            // Poll every 3s
            const interval = setInterval(async () => {
                const updated = await semanticService.getIndexingStatus(evidenceId);
                setIndexingStatus(updated);
                if (updated.status === "COMPLETED" || updated.status === "FAILED") {
                    clearInterval(interval);
                }
            }, 3000);
        } catch (err: any) {
            setError(err.response?.data?.detail || "Failed to trigger semantic indexing.");
        } finally {
            setIsTriggeringIndex(false);
        }
    };

    const handleSearch = async (queryToRun?: string) => {
        const q = queryToRun || query;
        if (!q.trim()) return;

        setIsSearching(true);
        setError(null);

        try {
            const res = await semanticService.searchEvidence(evidenceId, q);
            setSearchResponse(res);
        } catch (err: any) {
            setError(err.response?.data?.detail || "Error performing semantic search.");
            setSearchResponse(null);
        } finally {
            setIsSearching(false);
        }
    };

    return (
        <div className="space-y-6 bg-slate-900/60 border border-slate-800 rounded-xl p-6 backdrop-blur-md">
            {/* Header & Indexing Banner */}
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-800">
                <div>
                    <div className="flex items-center gap-2">
                        <Sparkles className="w-5 h-5 text-cyan-400" />
                        <h2 className="text-xl font-bold text-white tracking-wide">
                            Semantic Forensic Intelligence
                        </h2>
                    </div>
                    <p className="text-sm text-slate-400 mt-1">
                        Grounded CCTV search powered by natural language, vector similarity & VLM keyframe analysis.
                    </p>
                </div>

                {/* Indexing Action / Status Badge */}
                <div className="flex items-center gap-3">
                    {indexingStatus?.status === "COMPLETED" ? (
                        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                            <CheckCircle className="w-3.5 h-3.5" />
                            Indexed ({indexingStatus.total_documents || 0} Docs)
                        </span>
                    ) : indexingStatus?.status === "PROCESSING" ? (
                        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 animate-pulse">
                            <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                            Indexing Stage: {indexingStatus.stage} ({indexingStatus.progress}%)
                        </span>
                    ) : (
                        <button
                            onClick={handleTriggerIndex}
                            disabled={isTriggeringIndex}
                            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold bg-cyan-600 hover:bg-cyan-500 text-white transition-all shadow-md shadow-cyan-900/20"
                        >
                            <Database className="w-3.5 h-3.5" />
                            {isTriggeringIndex ? "Queueing Indexing..." : "Generate Semantic Index"}
                        </button>
                    )}
                </div>
            </div>

            {/* Search Bar Input */}
            <div className="space-y-3">
                <form
                    onSubmit={(e) => {
                        e.preventDefault();
                        handleSearch();
                    }}
                    className="relative flex items-center"
                >
                    <Search className="absolute left-4 w-5 h-5 text-slate-400" />
                    <input
                        type="text"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Ask GuardianEye about this evidence... (e.g. Find a person carrying a backpack)"
                        className="w-full pl-12 pr-28 py-3.5 bg-slate-950/80 border border-slate-700/80 rounded-xl text-white placeholder-slate-500 text-sm focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all shadow-inner"
                    />
                    <button
                        type="submit"
                        disabled={isSearching || !query.trim()}
                        className="absolute right-2 px-4 py-2 bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white text-xs font-semibold rounded-lg transition-all shadow-md shadow-cyan-950/50 disabled:opacity-50"
                    >
                        {isSearching ? "Analyzing..." : "Search Evidence"}
                    </button>
                </form>

                {/* Quick Example Chips */}
                <div className="flex flex-wrap items-center gap-2 pt-1">
                    <span className="text-xs text-slate-400 font-medium mr-1">Examples:</span>
                    {EXAMPLE_QUERIES.map((example, idx) => (
                        <button
                            key={idx}
                            onClick={() => {
                                setQuery(example);
                                handleSearch(example);
                            }}
                            className="text-xs px-2.5 py-1 rounded-md bg-slate-800/80 hover:bg-slate-700 text-slate-300 border border-slate-700/60 transition-colors"
                        >
                            {example}
                        </button>
                    ))}
                </div>
            </div>

            {/* Error Banner */}
            {error && (
                <div className="flex items-center gap-3 p-4 bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-lg text-sm">
                    <AlertTriangle className="w-5 h-5 flex-shrink-0" />
                    <span>{error}</span>
                </div>
            )}

            {/* Search Results Display */}
            {searchResponse && (
                <div className="space-y-6 pt-2">
                    {/* Grounded Summary Banner */}
                    <div className="p-4 bg-slate-950/90 border border-cyan-500/30 rounded-xl space-y-2">
                        <div className="flex items-center justify-between text-xs text-slate-400">
                            <span className="flex items-center gap-1.5 text-cyan-400 font-semibold uppercase tracking-wider">
                                <Shield className="w-4 h-4" /> Evidence Grounded Response
                            </span>
                            <span>Execution Time: {searchResponse.execution_time_ms}ms</span>
                        </div>
                        <p className="text-sm font-medium text-slate-200 leading-relaxed">
                            {searchResponse.answer}
                        </p>
                    </div>

                    {/* Results Cards List */}
                    <div className="space-y-4">
                        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                            Retrieved Evidence Records ({searchResponse.total_results})
                        </h3>

                        {searchResponse.results.length === 0 ? (
                            <div className="p-8 text-center bg-slate-950/40 border border-slate-800/80 rounded-xl space-y-2">
                                <AlertTriangle className="w-8 h-8 text-amber-400 mx-auto opacity-80" />
                                <h4 className="text-sm font-semibold text-slate-200">No Supported Evidence Found</h4>
                                <p className="text-xs text-slate-400 max-w-md mx-auto">
                                    GuardianEye strictly enforces anti-hallucination policies. No underlying computer vision or VLM observations matched this query.
                                </p>
                            </div>
                        ) : (
                            searchResponse.results.map((item, idx) => (
                                <div
                                    key={idx}
                                    className="p-5 bg-slate-950/80 border border-slate-800/90 hover:border-cyan-500/40 rounded-xl space-y-4 transition-all"
                                >
                                    {/* Top Bar: Title & Separate Confidence Metrics Badges */}
                                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                                        <div className="space-y-1.5">
                                            <div className="flex flex-wrap items-center gap-2">
                                                <span className="text-sm font-bold text-white">{item.title}</span>

                                                {/* Query Relevance Metric */}
                                                <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-md text-[10px] font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                                                    Relevance: {item.query_relevance !== undefined ? `${item.query_relevance.toFixed(0)}%` : `${(item.score * 100).toFixed(0)}%`}
                                                </span>

                                                {/* Evidence Support Metric */}
                                                <span
                                                    className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-md text-[10px] font-bold border ${(item.evidence_support || item.confidence_level) === "HIGH"
                                                            ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30"
                                                            : (item.evidence_support || item.confidence_level) === "MEDIUM"
                                                                ? "bg-amber-500/10 text-amber-400 border-amber-500/30"
                                                                : "bg-slate-500/10 text-slate-400 border-slate-500/30"
                                                        }`}
                                                >
                                                    Support: {item.evidence_support || item.confidence_level}
                                                </span>

                                                {/* Model Detection Confidence Metric */}
                                                {item.model_confidence !== undefined && item.model_confidence !== null && (
                                                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[10px] font-bold bg-indigo-500/10 text-indigo-300 border border-indigo-500/30">
                                                        <Cpu className="w-3 h-3 text-indigo-400" />
                                                        Det Conf: {(item.model_confidence * 100).toFixed(0)}%
                                                    </span>
                                                )}
                                            </div>
                                            <p className="text-xs text-slate-400">{item.confidence_reason}</p>
                                        </div>

                                        {/* Timestamp & Video Jump Action */}
                                        {item.start_time !== undefined && item.start_time !== null && (
                                            <button
                                                onClick={() => onJumpToTimestamp && onJumpToTimestamp(item.start_time!)}
                                                className="inline-flex items-center gap-2 px-3 py-1.5 bg-cyan-600/20 hover:bg-cyan-600/40 text-cyan-300 border border-cyan-500/30 text-xs font-semibold rounded-lg transition-all self-start sm:self-auto"
                                            >
                                                <Play className="w-3.5 h-3.5 fill-current" />
                                                Jump to {item.start_time.toFixed(1)}s
                                            </button>
                                        )}
                                    </div>

                                    {/* Summary Text Content */}
                                    <p className="text-xs text-slate-300 bg-slate-900/90 p-3 rounded-lg border border-slate-800/80 leading-relaxed font-mono">
                                        {item.summary}
                                    </p>

                                    {/* "Why This Result?" Transparent Forensic Explanation Accordion */}
                                    <div>
                                        <button
                                            onClick={() =>
                                                setExpandedExplanation(expandedExplanation === idx ? null : idx)
                                            }
                                            className="flex items-center gap-1.5 text-xs text-cyan-400 font-semibold hover:text-cyan-300 transition-colors"
                                        >
                                            <HelpCircle className="w-3.5 h-3.5" />
                                            <span>Why this result?</span>
                                            {expandedExplanation === idx ? (
                                                <ChevronUp className="w-3.5 h-3.5" />
                                            ) : (
                                                <ChevronDown className="w-3.5 h-3.5" />
                                            )}
                                        </button>

                                        {expandedExplanation === idx && (
                                            <div className="mt-3 p-4 bg-slate-900/80 border border-slate-800 rounded-lg text-xs space-y-3 font-mono">
                                                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 text-slate-300 border-b border-slate-800 pb-3">
                                                    <div>
                                                        <span className="text-slate-500 block">Query Relevance:</span>
                                                        <span className="text-cyan-400 font-bold text-sm">
                                                            {item.why_explanation.query_relevance_formatted ||
                                                                `${(item.score * 100).toFixed(0)}%`}
                                                        </span>
                                                    </div>
                                                    <div>
                                                        <span className="text-slate-500 block">Semantic Embedding Similarity:</span>
                                                        <span className="text-slate-200 font-medium">
                                                            {item.why_explanation.semantic_match_percentage ||
                                                                `${(item.why_explanation.semantic_match_score * 100).toFixed(1)}%`}
                                                        </span>
                                                    </div>
                                                    <div>
                                                        <span className="text-slate-500 block">Evidence Support:</span>
                                                        <span className="text-emerald-400 font-semibold">
                                                            {item.why_explanation.evidence_support || item.evidence_support || "HIGH"}
                                                        </span>
                                                    </div>
                                                    <div>
                                                        <span className="text-slate-500 block">Document Type:</span>
                                                        <span className="text-slate-200">{item.why_explanation.document_type}</span>
                                                    </div>
                                                    <div>
                                                        <span className="text-slate-500 block">Source Type:</span>
                                                        <span className="text-slate-200">{item.why_explanation.source_type}</span>
                                                    </div>
                                                    <div>
                                                        <span className="text-slate-500 block">Temporal Range:</span>
                                                        <span className="text-slate-200">{item.why_explanation.temporal_range}</span>
                                                    </div>
                                                </div>

                                                {/* Verification & Attribute Breakdown */}
                                                <div className="space-y-2">
                                                    <div className="flex items-center gap-2">
                                                        <span className="text-slate-400">Entity Match:</span>
                                                        <span className="text-emerald-400 font-bold">
                                                            {item.why_explanation.entity_match_details || "✓ VERIFIED"}
                                                        </span>
                                                    </div>

                                                    {item.why_explanation.visual_attribute_matches &&
                                                        item.why_explanation.visual_attribute_matches.length > 0 && (
                                                            <div className="space-y-1">
                                                                <span className="text-slate-400 block">Visual Attributes:</span>
                                                                <div className="flex flex-wrap gap-2">
                                                                    {item.why_explanation.visual_attribute_matches.map((attr, aIdx) => (
                                                                        <span
                                                                            key={aIdx}
                                                                            className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${attr.status === "VERIFIED"
                                                                                    ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/30"
                                                                                    : "bg-amber-500/10 text-amber-400 border-amber-500/30"
                                                                                }`}
                                                                        >
                                                                            {attr.formatted}
                                                                        </span>
                                                                    ))}
                                                                </div>
                                                            </div>
                                                        )}

                                                    <div className="flex flex-wrap gap-4 pt-1 text-slate-400 text-[11px]">
                                                        {item.why_explanation.supporting_track && (
                                                            <span>Supporting Track: #{item.why_explanation.supporting_track}</span>
                                                        )}
                                                        {item.why_explanation.supporting_frames !== undefined && (
                                                            <span>Supporting Frames: {item.why_explanation.supporting_frames}</span>
                                                        )}
                                                        {item.why_explanation.detection_confidence_formatted && (
                                                            <span>CV Model Confidence: {item.why_explanation.detection_confidence_formatted}</span>
                                                        )}
                                                    </div>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))
                        )}

                        {/* Search Suggestions for No-Result State */}
                        {searchResponse.results.length === 0 && searchResponse.suggestions && searchResponse.suggestions.length > 0 && (
                            <div className="p-4 bg-slate-950/60 border border-slate-800 rounded-xl space-y-3">
                                <h4 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                                    Suggested Forensic Queries:
                                </h4>
                                <div className="flex flex-wrap gap-2">
                                    {searchResponse.suggestions.map((sug, sIdx) => (
                                        <button
                                            key={sIdx}
                                            onClick={() => {
                                                setQuery(sug);
                                                handleSearch(sug);
                                            }}
                                            className="text-xs px-3 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-cyan-300 border border-cyan-500/30 transition-all font-medium"
                                        >
                                            {sug}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};
