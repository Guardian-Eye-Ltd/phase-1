import apiClient from "./api";

export type FaceTrackStatus = "USABLE_FACE" | "LOW_QUALITY" | "NO_FACE_VISIBLE";

export interface FaceObservationView {
    frame_number: number;
    timestamp: number;
    quality_status: "USABLE" | "LOW_QUALITY";
    quality_reasons: string[];
    face_size_px: [number, number];
    det_score: number;
    sharpness: number;
    crop_url: string;
    crop_sha256: string;
}

export interface FaceTrackView {
    track_id: number;
    first_seen: number;
    last_seen: number;
    face_status: FaceTrackStatus;
    faces: FaceObservationView[];
}

export interface FaceListResponse {
    evidence_id: number;
    analysis_job_id: number;
    face_stage: "NOT_RUN" | "COMPLETED" | "UNAVAILABLE" | "FAILED";
    face_stage_detail: string | null;
    match_threshold_default: number;
    tracks: FaceTrackView[];
}

export interface FaceMatch {
    track_id: number;
    status: "POSSIBLE_MATCH";
    best_similarity: number;
    margin_over_threshold: number;
    supporting_observations: number;
    compared_observations: number;
    first_seen: number;
    last_seen: number;
    observations: { frame_number: number; timestamp: number; similarity: number; crop_url: string; face_width_px: number }[];
}

export interface FaceSearchResponse {
    evidence_id: number;
    analysis_job_id: number;
    threshold: number;
    status: "POSSIBLE_MATCH_FOUND" | "NO_MATCH_AMONG_USABLE_FACES" | "NO_USABLE_FACES" | "FACE_EXTRACTION_NOT_AVAILABLE";
    message: string;
    coverage: {
        person_tracks: number;
        tracks_with_usable_face: number;
        tracks_with_low_quality_face_only: number;
        tracks_with_no_face_visible: number;
    } | null;
    matches: FaceMatch[];
}

// crop_url is an absolute API path; apiClient's baseURL already ends in /api/v1.
const toClientPath = (url: string) => url.replace(/^\/api\/v1/, "");

export const faceService = {
    async listFaces(evidenceId: number): Promise<FaceListResponse> {
        const res = await apiClient.get(`/evidence/${evidenceId}/faces`);
        return res.data;
    },

    async searchByFace(evidenceId: number, photo: File, threshold?: number): Promise<FaceSearchResponse> {
        const form = new FormData();
        form.append("photo", photo);
        if (threshold !== undefined) form.append("threshold", String(threshold));
        const res = await apiClient.post(`/evidence/${evidenceId}/faces/search`, form, {
            headers: { "Content-Type": "multipart/form-data" },
            timeout: 120000, // first search may load the face model
        });
        return res.data;
    },

    // Face crops are biometric data served only to authenticated users, so they
    // are fetched with the bearer token rather than via a plain <img src>.
    async fetchCropObjectUrl(cropUrl: string): Promise<string> {
        const res = await apiClient.get(toClientPath(cropUrl), { responseType: "blob" });
        return URL.createObjectURL(res.data);
    },
};
