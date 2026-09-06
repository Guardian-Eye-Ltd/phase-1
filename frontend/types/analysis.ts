export type JobStatus = 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';

export type JobStage =
    | 'VALIDATING'
    | 'EXTRACTING_METADATA'
    | 'SAMPLING_FRAMES'
    | 'MOTION_ANALYSIS'
    | 'OBJECT_DETECTION'
    | 'TRACKING'
    | 'KEYFRAME_EXTRACTION'
    | 'FINALIZING';

export interface AnalysisJob {
    job_id: number;
    evidence_id: number;
    status: JobStatus;
    current_stage: JobStage;
    progress: number;
    sampling_fps: number;
    confidence_threshold: number;
    started_at?: string;
    completed_at?: string;
    error_message?: string;
    model_name?: string;
    manifest_hash?: string;
}

export interface Detection {
    id: number;
    frame_number: number;
    timestamp: number;
    class_name: string;
    confidence: number;
    bbox: [number, number, number, number]; // [x1, y1, x2, y2] normalized 0..1
    track_id?: number;
}

export interface Track {
    id: number;
    track_number: number;
    class_name: string;
    first_seen_timestamp: number;
    last_seen_timestamp: number;
    first_seen_frame: number;
    last_seen_frame: number;
    duration: number;
    observation_count: number;
}

export interface Keyframe {
    id: number;
    frame_number: number;
    timestamp: number;
    selection_reason: string;
    image_path: string;
    sha256_hash: string;
    track_ids?: number[];
}

export interface ActivityInterval {
    id: number;
    start_time: number;
    end_time: number;
    start_frame: number;
    end_frame: number;
    activity_level: string;
    motion_score: number;
}

export interface TimelineEvent {
    timestamp: number;
    frame_number: number;
    type: 'TRACK_APPEARANCE' | 'KEYFRAME_SELECTED' | 'POSSIBLE_INTERACTION';
    title: string;
    description: string;
    track_id?: number;
    class_name?: string;
    image_path?: string;
    sha256_hash?: string;
    confidence?: number;
}

export interface ManifestResponse {
    manifest_data: Record<string, any>;
    sha256_hash: string;
}
