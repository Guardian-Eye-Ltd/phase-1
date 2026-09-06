import apiClient from "./api";
import { AuditLogListResponse } from "@/types/audit";

export const auditService = {
    async getAuditLogs(page = 1, size = 50, action?: string, userId?: number): Promise<AuditLogListResponse> {
        const params: Record<string, any> = { page, size };
        if (action) params.action = action;
        if (userId) params.user_id = userId;

        const response = await apiClient.get<AuditLogListResponse>("/audit", { params });
        return response.data;
    }
};
