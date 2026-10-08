import apiClient from "./api";

export interface StorageStats {
    path: string;
    files: number;
    bytes: number;
}

export interface ResetPreview {
    tables: Record<string, number>;
    total_rows: number;
    storage: StorageStats[];
    vector_collections: number | null;
    running_jobs: number[];
    can_reset: boolean;
    confirmation_phrase: string;
    preserved: string[];
}

export interface ResetResult {
    status: "COMPLETED" | "COMPLETED_WITH_ERRORS";
    deleted_rows: Record<string, number>;
    total_rows_deleted: number;
    vector_collections_deleted: number;
    files_deleted: number;
    errors: string[];
}

export const adminService = {
    async getResetPreview(): Promise<ResetPreview> {
        const res = await apiClient.get<ResetPreview>("/admin/reset/preview");
        return res.data;
    },

    async resetSystem(confirmation: string): Promise<ResetResult> {
        const res = await apiClient.post<ResetResult>("/admin/reset", { confirmation });
        return res.data;
    },
};
