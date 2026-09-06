export type EvidenceStatus = "UPLOADED" | "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED";

export interface Evidence {
  id: number;
  original_filename: string;
  stored_filename: string;
  file_path: string;
  file_size: number;
  mime_type: string;
  duration: number | null;
  resolution: string | null;
  fps: number | null;
  status: EvidenceStatus;
  sha256_hash: string;
  uploaded_by: number;
  uploaded_at: string;
  created_at: string;
  updated_at: string;
}

export interface EvidenceListResponse {
  items: Evidence[];
  total: number;
  page: number;
  size: number;
}

export interface EvidenceUploadResponse {
  evidence: Evidence;
  is_duplicate: boolean;
  message: string;
}
