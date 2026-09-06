import apiClient from "./api";
import { Evidence, EvidenceListResponse, EvidenceUploadResponse } from "@/types/evidence";

export const evidenceService = {
    async uploadEvidence(file: File, onProgress?: (percent: number) => void): Promise<EvidenceUploadResponse> {
        const formData = new FormData();
        formData.append("file", file);

        const response = await apiClient.post<EvidenceUploadResponse>("/evidence/upload", formData, {
            headers: {
                "Content-Type": "multipart/form-data",
            },
            onUploadProgress: (progressEvent) => {
                if (progressEvent.total && onProgress) {
                    const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    onProgress(percent);
                }
            },
        });
        return response.data;
    },

    async getEvidenceList(page = 1, size = 20, status?: string, search?: string): Promise<EvidenceListResponse> {
        const params: Record<string, any> = { page, size };
        if (status && status !== "ALL") params.status = status;
        if (search) params.search = search;

        const response = await apiClient.get<EvidenceListResponse>("/evidence", { params });
        return response.data;
    },

    async getEvidenceById(id: number): Promise<Evidence> {
        const response = await apiClient.get<Evidence>(`/evidence/${id}`);
        return response.data;
    },

    async reprocessEvidence(id: number): Promise<Evidence> {
        const response = await apiClient.post<Evidence>(`/evidence/${id}/reprocess`);
        return response.data;
    },

    getStreamUrl(id: number): string {
        // Use the same-origin Next.js API proxy at /api/video/{id}.
        // The proxy injects the Authorization: Bearer header server-side,
        // avoiding CORS issues and eliminating the fragile ?token= query-param
        // approach that breaks when the HTML <video> element makes direct requests.
        const token = typeof window !== "undefined" ? localStorage.getItem("access_token") : null;
        if (!token) {
            console.warn("[evidenceService] No access_token in localStorage – video will fail auth.");
        }
        // Pass token as query param to the proxy — it runs on same-origin (Next.js),
        // so there are no CORS restrictions. The proxy then forwards it as a header.
        return `/api/video/${id}${token ? `?token=${encodeURIComponent(token)}` : ""}`;
    }
};
