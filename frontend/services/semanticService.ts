import apiClient from "./api";

export interface SemanticStatus {
    evidence_id: number;
    status: string;
    progress: number;
    stage: string;
    total_documents?: number;
    total_vlm_observations?: number;
    total_embeddings?: number;
    error?: string;
}

export interface VisualAttributeMatch {
    name: string;
    status: "VERIFIED" | "NOT_VERIFIED";
    formatted: string;
}

export interface SearchResultItem {
    doc_id: number;
    title: string;
    summary: string;
    score: number;
    confidence_level: "HIGH" | "MEDIUM" | "LOW";
    confidence_reason: string;
    query_relevance?: number;
    evidence_support?: "HIGH" | "MEDIUM" | "LOW";
    model_confidence?: number | null;
    track_id?: number;
    keyframe_id?: number;
    start_time?: number;
    end_time?: number;
    why_explanation: {
        semantic_match_score: number;
        semantic_match_percentage?: string;
        query_relevance?: number;
        query_relevance_formatted?: string;
        entity_match?: boolean;
        entity_match_details?: string;
        visual_attribute_matches?: VisualAttributeMatch[];
        supporting_frames?: number;
        supporting_keyframes?: number;
        supporting_track?: number;
        detection_confidence?: number | null;
        detection_confidence_formatted?: string;
        evidence_support?: string;
        document_type: string;
        source_type: string;
        temporal_range: string;
    };
    verification_status: string;
}

export interface SearchResponse {
    search_query_id: number;
    evidence_id: number;
    query: string;
    extracted_intent: Record<string, any>;
    answer: string;
    execution_time_ms: number;
    total_results: number;
    suggestions?: string[];
    results: SearchResultItem[];
}

export interface EnhancedTimelineEvent {
    id: string;
    timestamp: number;
    timestamp_str: string;
    event_type: string;
    title: string;
    description: string;
    track_id?: number;
    keyframe_id?: number;
    image_path?: string;
    confidence_score: number;
    source_type: string;
    is_clickable: boolean;
}

export interface EnhancedTimelineResponse {
    evidence_id: number;
    total_events: number;
    timeline: EnhancedTimelineEvent[];
}

export const semanticService = {
    triggerIndexing: async (evidenceId: number): Promise<SemanticStatus> => {
        const res = await apiClient.post(`/evidence/${evidenceId}/semantic-index`, {});
        return res.data;
    },

    getIndexingStatus: async (evidenceId: number): Promise<SemanticStatus> => {
        const res = await apiClient.get(`/evidence/${evidenceId}/semantic-status`);
        return res.data;
    },

    searchEvidence: async (evidenceId: number, query: string, topK: number = 5): Promise<SearchResponse> => {
        const res = await apiClient.post(`/evidence/${evidenceId}/search`, {
            query,
            top_k: topK
        });
        return res.data;
    },

    getEnhancedTimeline: async (evidenceId: number): Promise<EnhancedTimelineResponse> => {
        const res = await apiClient.get(`/evidence/${evidenceId}/timeline/enhanced`);
        return res.data;
    },

    getForensicDocuments: async (evidenceId: number): Promise<any[]> => {
        const res = await apiClient.get(`/evidence/${evidenceId}/forensic-documents`);
        return res.data;
    },

    getAIObservations: async (evidenceId: number): Promise<any[]> => {
        const res = await apiClient.get(`/evidence/${evidenceId}/ai-observations`);
        return res.data;
    }
};
