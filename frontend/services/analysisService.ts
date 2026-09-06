import apiClient from "./api";
import {
    AnalysisJob,
    Detection,
    Track,
    Keyframe,
    ActivityInterval,
    TimelineEvent,
    ManifestResponse
} from "../types/analysis";

export const analysisService = {
    // Start analysis job
    startAnalysis: async (
        evidenceId: number,
        config: { sampling_fps: number; confidence_threshold: number; tracking_enabled: boolean }
    ): Promise<AnalysisJob> => {
        const res = await apiClient.post(`/evidence/${evidenceId}/analysis`, config);
        return res.data;
    },

    // Poll analysis job status
    getAnalysisJob: async (jobId: number): Promise<AnalysisJob> => {
        const res = await apiClient.get(`/evidence/analysis/${jobId}`);
        return res.data;
    },

    // Cancel analysis job
    cancelAnalysisJob: async (jobId: number): Promise<{ message: string }> => {
        const res = await apiClient.post(`/evidence/analysis/${jobId}/cancel`);
        return res.data;
    },

    // Detections
    getDetections: async (evidenceId: number, className?: string, trackId?: number): Promise<Detection[]> => {
        const params: Record<string, any> = {};
        if (className) params.class_name = className;
        if (trackId) params.track_id = trackId;
        const res = await apiClient.get(`/evidence/${evidenceId}/detections`, { params });
        return res.data;
    },

    // Tracks
    getTracks: async (evidenceId: number, className?: string): Promise<Track[]> => {
        const params: Record<string, any> = {};
        if (className) params.class_name = className;
        const res = await apiClient.get(`/evidence/${evidenceId}/tracks`, { params });
        return res.data;
    },

    // Keyframes
    getKeyframes: async (evidenceId: number): Promise<Keyframe[]> => {
        const res = await apiClient.get(`/evidence/${evidenceId}/keyframes`);
        return res.data;
    },

    // Activity Intervals
    getActivityIntervals: async (evidenceId: number): Promise<ActivityInterval[]> => {
        const res = await apiClient.get(`/evidence/${evidenceId}/activity`);
        return res.data;
    },

    // Timeline
    getTimeline: async (evidenceId: number): Promise<TimelineEvent[]> => {
        const res = await apiClient.get(`/evidence/${evidenceId}/timeline`);
        return res.data;
    },

    // Manifest
    getManifest: async (evidenceId: number): Promise<ManifestResponse> => {
        const res = await apiClient.get(`/evidence/${evidenceId}/manifest`);
        return res.data;
    }
};
