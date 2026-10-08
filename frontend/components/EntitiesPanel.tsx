"use client";

import React, { useEffect, useState } from "react";
import { AlertTriangle, Car, HelpCircle, Loader2, Play, User } from "lucide-react";
import { analysisService } from "@/services/analysisService";
import { EntitiesResponse, Entity, EntityAttribute } from "@/types/analysis";

interface EntitiesPanelProps {
    evidenceId: number;
    onSeek: (timestamp: number) => void;
}

const LABELS: Record<string, string> = {
    upper_garment_presence: "Upper body",
    upper_garment_color: "Top colour",
    upper_garment_type: "Top type",
    lower_garment_color: "Lower colour",
    headwear: "Headwear",
    carried_item: "Carrying",
    vehicle_color: "Colour",
    vehicle_body_style: "Body type",
    license_plate_text: "Plate",
};

const NOT_DETERMINED_LABELS: Record<string, string> = {
    upper_garment: "Upper body",
    lower_garment: "Lower body",
    clothing: "Clothing",
    carried_item: "Carrying",
    license_plate_text: "Plate",
    vehicle_make_model: "Make / model",
};

const SWATCH: Record<string, string> = {
    black: "#111", white: "#f5f5f5", grey: "#8a8f98", red: "#d33", orange: "#f39433", yellow: "#f2d33a",
    green: "#3a9d4a", blue: "#3b6fd6", purple: "#8a4fd0", pink: "#f08fc0", brown: "#7a4e2d", beige: "#d9c49a",
};

const fmt = (t: number) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;

function displayValue(attr: string, value: string): string {
    if (attr === "upper_garment_presence") return value === "absent" ? "no top (bare torso)" : "clothed";
    return value;
}

const AttributeRow: React.FC<{ name: string; a: EntityAttribute }> = ({ name, a }) => {
    const withheld = a.status !== "OBSERVED";
    const isColour = name.endsWith("_color");
    const isPlate = name === "license_plate_text";
    return (
        <div className={`flex items-center justify-between gap-2 text-xs ${withheld ? "opacity-60" : ""}`}
            title={withheld ? `Not asserted: ${a.withheld_reason ?? "insufficient agreement"}` : undefined}>
            <span className="text-slate-400 w-24 shrink-0">{LABELS[name] ?? name}</span>
            <span className="flex-1 flex items-center gap-1.5 text-slate-100 min-w-0">
                {isColour && <span className="w-3 h-3 rounded-sm border border-slate-600 shrink-0" style={{ background: SWATCH[a.value] ?? "transparent" }} />}
                <span className={`truncate ${isPlate ? "font-mono font-bold tracking-wider text-cyan-200" : ""}`}>
                    {withheld ? `${displayValue(name, a.value)}?` : displayValue(name, a.value)}
                </span>
                {withheld && (
                    <span className="text-[9px] font-mono px-1 rounded bg-slate-800 text-slate-400">
                        {a.status === "CONTRADICTED" ? "contradicted" : "uncertain"}
                    </span>
                )}
            </span>
            <span className="text-[10px] font-mono text-slate-500 shrink-0">
                {a.confidence.toFixed(2)} · {a.supporting_count}/{a.observation_count}
            </span>
        </div>
    );
};

const EntityCard: React.FC<{ e: Entity; onSeek: (t: number) => void }> = ({ e, onSeek }) => {
    const Icon = e.entity_type === "PERSON" ? User : Car;
    const attrs = Object.entries(e.attributes).filter(([k]) => !(k === "carries_bag" && e.attributes.carried_item));
    const showColour = e.attributes.upper_garment_presence?.value !== "absent";
    return (
        <div className="p-4 rounded-xl border border-slate-800 bg-[#070A11] space-y-3">
            <div className="flex items-center justify-between">
                <span className="text-xs font-bold font-mono text-emerald-400 uppercase flex items-center gap-1.5">
                    <Icon className="w-3.5 h-3.5" /> {e.class_name} #{e.track_id}
                </span>
                <button onClick={() => onSeek(e.first_seen)}
                    className="text-[10px] font-mono text-slate-300 bg-slate-900 hover:bg-slate-800 px-2 py-0.5 rounded flex items-center gap-1">
                    <Play className="w-3 h-3 text-emerald-400" /> {fmt(e.first_seen)}–{fmt(e.last_seen)}
                </button>
            </div>
            <div className="space-y-1.5">
                {attrs.filter(([k]) => showColour || k !== "upper_garment_color").map(([k, a]) => (
                    <AttributeRow key={k} name={k} a={a} />
                ))}
                {attrs.length === 0 && Object.keys(e.not_determined).length === 0 && (
                    <div className="text-[11px] text-slate-500">No attributes extracted.</div>
                )}
            </div>
            {Object.keys(e.not_determined).length > 0 && (
                <div className="pt-2 border-t border-slate-800/80 space-y-1">
                    {Object.entries(e.not_determined).map(([k, reason]) => (
                        <div key={k} className="flex gap-2 text-[10px] text-slate-500" title={reason}>
                            <HelpCircle className="w-3 h-3 shrink-0 mt-0.5" />
                            <span className="text-slate-400 shrink-0">{NOT_DETERMINED_LABELS[k] ?? k}:</span>
                            <span className="line-clamp-2">{reason}</span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
};

export const EntitiesPanel: React.FC<EntitiesPanelProps> = ({ evidenceId, onSeek }) => {
    const [data, setData] = useState<EntitiesResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [filter, setFilter] = useState<"ALL" | "PERSON" | "VEHICLE">("ALL");

    useEffect(() => {
        analysisService.getEntities(evidenceId)
            .then(setData)
            .catch((err) => setError(err?.response?.data?.detail || "Could not load tracked entities."));
    }, [evidenceId]);

    if (error) return <div className="text-sm text-slate-400 flex items-center gap-2"><AlertTriangle className="w-4 h-4 text-amber-400" />{error}</div>;
    if (!data) return <div className="text-xs font-mono text-slate-500 flex items-center gap-2"><Loader2 className="w-4 h-4 animate-spin" />Loading…</div>;

    const shown = data.entities.filter((e) => e.entity_type !== "OBJECT" && (filter === "ALL" || e.entity_type === filter));
    const count = (t: string) => data.entities.filter((e) => e.entity_type === t).length;

    return (
        <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
                <div className="flex gap-2">
                    {(["ALL", "PERSON", "VEHICLE"] as const).map((f) => (
                        <button key={f} onClick={() => setFilter(f)}
                            className={`px-3 py-1 rounded-lg text-xs font-mono border ${filter === f ? "bg-emerald-950 text-emerald-300 border-emerald-500/40" : "text-slate-400 border-slate-800 hover:text-slate-200"}`}>
                            {f === "ALL" ? `All (${count("PERSON") + count("VEHICLE")})` : f === "PERSON" ? `People (${count("PERSON")})` : `Vehicles (${count("VEHICLE")})`}
                        </button>
                    ))}
                </div>
                <span className="text-[10px] font-mono text-slate-500">
                    analysis run #{data.analysis_job_id} · value · confidence · frames agreeing / frames observed
                </span>
            </div>

            {!data.attributes_extracted && (
                <div className="p-3 rounded-lg border border-amber-500/30 bg-amber-950/20 text-xs text-amber-200 flex gap-2">
                    <AlertTriangle className="w-4 h-4 shrink-0 text-amber-400" />
                    This analysis run predates attribute extraction. Use Reprocess to extract clothing, vehicle colour and plates.
                </div>
            )}

            {shown.length === 0 ? (
                <div className="text-center py-12 text-xs font-mono text-slate-500">No tracked people or vehicles.</div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {shown.map((e) => <EntityCard key={e.track_id} e={e} onSeek={onSeek} />)}
                </div>
            )}
        </div>
    );
};
