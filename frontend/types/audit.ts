export interface AuditLog {
    id: number;
    user_id: number | null;
    user_name: string;
    username: string;
    action: string;
    resource_type: string;
    resource_id: string | null;
    timestamp: string;
    metadata: Record<string, any> | null;
}

export interface AuditLogListResponse {
    items: AuditLog[];
    total: number;
    page: number;
    size: number;
}
