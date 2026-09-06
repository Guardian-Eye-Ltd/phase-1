import { NextRequest, NextResponse } from "next/server";

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

/**
 * Next.js API Proxy for CCTV evidence video streaming.
 *
 * Motivation: The HTML <video> element makes direct browser requests that
 * bypass Axios interceptors, so the Authorization header cannot be injected.
 * Passing the token as a query-param (?token=) is fragile and broken by
 * stale/malformed tokens. This proxy forwards the request to the FastAPI
 * backend with the token in the Authorization header (extracted from the
 * cookie or the Authorization header forwarded by the browser), then streams
 * the response back — completely eliminating the token-in-URL problem.
 *
 * The frontend now points <video src> to /api/video/{id} (same-origin),
 * so cookies and Authorization headers work natively without CORS concerns.
 */
export async function GET(
    request: NextRequest,
    { params }: { params: { id: string } }
) {
    const evidenceId = params.id;

    // Extract token from Authorization header or from query param as fallback
    const authHeader = request.headers.get("authorization");
    const queryToken = request.nextUrl.searchParams.get("token");
    const token = authHeader?.replace("Bearer ", "") || queryToken;

    if (!token) {
        return new NextResponse(
            JSON.stringify({ detail: "Authentication token is required." }),
            { status: 401, headers: { "Content-Type": "application/json" } }
        );
    }

    // Forward the Range header for HTTP range streaming support
    const rangeHeader = request.headers.get("range");

    const backendUrl = `${BACKEND_URL}/evidence/${evidenceId}/stream`;

    const backendHeaders: HeadersInit = {
        Authorization: `Bearer ${token}`,
    };
    if (rangeHeader) {
        backendHeaders["Range"] = rangeHeader;
    }

    try {
        const backendResponse = await fetch(backendUrl, {
            headers: backendHeaders,
        });

        if (!backendResponse.ok && backendResponse.status !== 206) {
            // Pass through auth errors etc. with a clean JSON body
            const errorBody = await backendResponse.text();
            return new NextResponse(errorBody, {
                status: backendResponse.status,
                headers: { "Content-Type": "application/json" },
            });
        }

        // Stream the video back to the browser
        const responseHeaders: Record<string, string> = {
            "Content-Type": backendResponse.headers.get("Content-Type") || "video/mp4",
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
        };

        const contentRange = backendResponse.headers.get("Content-Range");
        const contentLength = backendResponse.headers.get("Content-Length");
        if (contentRange) responseHeaders["Content-Range"] = contentRange;
        if (contentLength) responseHeaders["Content-Length"] = contentLength;

        return new NextResponse(backendResponse.body, {
            status: backendResponse.status,
            headers: responseHeaders,
        });
    } catch (err) {
        console.error("[video-proxy] Backend fetch error:", err);
        return new NextResponse(
            JSON.stringify({ detail: "Failed to connect to evidence backend." }),
            { status: 502, headers: { "Content-Type": "application/json" } }
        );
    }
}
