"use client";

import React, { useEffect, useState } from "react";
import { faceService } from "@/services/faceService";

interface AuthImageProps {
    src: string;
    alt: string;
    className?: string;
}

/** Image loaded with the user's bearer token (for protected crops). */
export const AuthImage: React.FC<AuthImageProps> = ({ src, alt, className }) => {
    const [objectUrl, setObjectUrl] = useState<string | null>(null);
    const [failed, setFailed] = useState(false);

    useEffect(() => {
        let revoked = false;
        let url: string | null = null;
        setFailed(false);
        faceService
            .fetchCropObjectUrl(src)
            .then((u) => {
                url = u;
                if (!revoked) setObjectUrl(u);
                else URL.revokeObjectURL(u);
            })
            .catch(() => !revoked && setFailed(true));
        return () => {
            revoked = true;
            if (url) URL.revokeObjectURL(url);
        };
    }, [src]);

    if (failed) {
        return <div className={`${className ?? ""} flex items-center justify-center bg-slate-900 text-[9px] text-slate-500 font-mono`}>unavailable</div>;
    }
    if (!objectUrl) {
        return <div className={`${className ?? ""} bg-slate-900 animate-pulse`} />;
    }
    return <img src={objectUrl} alt={alt} className={className} />;
};
