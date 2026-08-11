"use client";

import React, { useEffect, useRef, useState } from "react";
import { Camera, Activity, AlertCircle, WifiOff } from "lucide-react";

interface CameraFeedCardProps {
  cameraId: number;
  cameraName: string;
  location: string;
  isActive?: boolean;
}

export default function CameraFeedCard({ cameraId, cameraName, location, isActive = true }: CameraFeedCardProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<string>("CONNECTING");
  const [fps, setFps] = useState<number>(0);
  const [cameraActive, setCameraActive] = useState<boolean>(isActive);
  const [reconnectAttempts, setReconnectAttempts] = useState<number>(0);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    let isMounted = true;
    let reconnectDelay = 1000;

    const connect = () => {
      if (!isMounted) return;
      
      if (!cameraActive) {
        setStatus("OFFLINE");
        return;
      }

      const token = localStorage.getItem("access_token");
      if (!token) {
        setStatus("OFFLINE");
        return;
      }

      // Determine WS URL based on current host or environment variable
      const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const host = process.env.NEXT_PUBLIC_API_URL 
        ? new URL(process.env.NEXT_PUBLIC_API_URL).host 
        : window.location.hostname + ":8000";
      
      const wsUrl = `${wsProtocol}//${host}/api/v1/streams/${cameraId}?token=${token}`;
      
      console.log(`Connecting to WS: ${wsUrl}`);
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        if (!isMounted) return;
        setStatus("LIVE");
        setReconnectAttempts(0);
        reconnectDelay = 1000; // reset delay
      };

      ws.onmessage = async (event) => {
        if (!isMounted) return;
        
        if (typeof event.data === "string") {
          try {
            const data = JSON.parse(event.data);
            if (data.type === "status") {
              setStatus(data.status);
              setFps(data.fps);
            } else if (data.type === "error") {
              setStatus("ERROR");
              console.error("Camera stream error:", data.message);
            }
          } catch (e) {
            console.error("Failed to parse WS message", e);
          }
        } else if (event.data instanceof Blob) {
          // It's a binary frame (JPEG)
          const canvas = canvasRef.current;
          if (canvas) {
            const ctx = canvas.getContext("2d");
            if (ctx) {
              const img = new Image();
              const objectUrl = URL.createObjectURL(event.data);
              img.onload = () => {
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                URL.revokeObjectURL(objectUrl);
              };
              img.src = objectUrl;
            }
          }
        }
      };

      ws.onclose = () => {
        if (!isMounted) return;
        setStatus("RECONNECTING");
        setReconnectAttempts(prev => prev + 1);
        setFps(0);
        // Exponential backoff
        reconnectTimeoutRef.current = setTimeout(connect, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 2, 30000);
      };

      ws.onerror = (error) => {
        console.error(`WebSocket error on camera ${cameraId}`, error);
        ws.close();
      };
    };

    connect();

    return () => {
      isMounted = false;
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [cameraId, cameraActive]);

  const toggleStream = async () => {
    try {
      // Opt-in UI change
      setCameraActive(!cameraActive);
      if (cameraActive && wsRef.current) {
        wsRef.current.close();
      }
      
      const response = await apiClient.post(`/cameras/${cameraId}/toggle`);
      setCameraActive(response.data.is_active);
    } catch (error) {
      console.error("Failed to toggle stream", error);
      // Revert if failed
      setCameraActive(cameraActive);
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case "LIVE": return "text-green-500";
      case "CONNECTING": return "text-yellow-500";
      case "RECONNECTING": return "text-orange-500";
      case "OFFLINE":
      case "ERROR": return "text-red-500";
      default: return "text-gray-500";
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case "LIVE": return <Activity className="w-4 h-4 mr-1 animate-pulse" />;
      case "CONNECTING":
      case "RECONNECTING": return <AlertCircle className="w-4 h-4 mr-1 animate-spin" />;
      case "OFFLINE":
      case "ERROR": return <WifiOff className="w-4 h-4 mr-1" />;
      default: return <Camera className="w-4 h-4 mr-1" />;
    }
  };

  return (
    <div className="bg-cyberDarker border border-cyberDark rounded-xl overflow-hidden flex flex-col shadow-lg transition-all hover:border-cyberMuted">
      {/* Header */}
      <div className="px-4 py-3 flex justify-between items-center border-b border-cyberDark bg-cyberPanel/50">
        <div>
          <h3 className="text-cyberText font-semibold flex items-center">
            <Camera className="w-4 h-4 mr-2 text-cyberCyan" />
            {cameraName}
          </h3>
          <p className="text-cyberMuted text-xs mt-1">{location}</p>
        </div>
        <div className="flex space-x-2">
          <button 
            onClick={toggleStream}
            className={`px-2 py-1 text-xs rounded border transition-colors ${
              cameraActive ? "border-red-500/50 text-red-500 hover:bg-red-500/10" : "border-green-500/50 text-green-500 hover:bg-green-500/10"
            }`}
          >
            {cameraActive ? "PAUSE" : "RESUME"}
          </button>
          <div className={`flex items-center text-xs font-bold px-2 py-1 rounded bg-black/40 ${getStatusColor()}`}>
            {getStatusIcon()}
            {status}
          </div>
        </div>
      </div>
      
      {/* Video Area */}
      <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden group">
        <canvas 
          ref={canvasRef} 
          width={1280} 
          height={720} 
          className="w-full h-full object-contain"
        />
        
        {status !== "LIVE" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/60 backdrop-blur-sm text-center">
            {getStatusIcon()}
            <span className={`mt-2 font-mono text-sm tracking-widest uppercase ${getStatusColor()}`}>
              {status}
            </span>
            {status === "RECONNECTING" && reconnectAttempts > 0 && (
              <span className="mt-1 font-mono text-xs text-cyberMuted">
                Attempt {reconnectAttempts}
              </span>
            )}
          </div>
        )}

        {/* Overlay Stats */}
        <div className="absolute bottom-0 left-0 right-0 p-2 bg-gradient-to-t from-black/80 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
          <div className="flex justify-between items-center text-xs font-mono">
            <span className="text-cyberMuted">FPS: <span className="text-cyberCyan">{fps.toFixed(1)}</span></span>
            <span className="text-cyberMuted">Latency: <span className="text-cyberCyan">&lt;1s</span></span>
          </div>
        </div>
      </div>
    </div>
  );
}
