"use client";

import React, { useEffect, useState } from "react";
import apiClient from "@/services/api";
import CameraFeedCard from "@/components/CameraFeedCard";
import DashboardLayout from "@/components/DashboardLayout";
import { ServerCrash, Loader2, Video } from "lucide-react";

interface CameraData {
  id: number;
  name: string;
  camera_code: string;
  location: string;
  status: string;
  is_active: boolean;
}

export default function RealTimeMonitoringPage() {
  const [cameras, setCameras] = useState<CameraData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [gridCols, setGridCols] = useState(2);

  useEffect(() => {
    const fetchCameras = async () => {
      try {
        const response = await apiClient.get("/cameras");
        const activeCameras = response.data.filter((c: CameraData) => c.is_active);
        setCameras(activeCameras);
        if (activeCameras.length > 4) {
          setGridCols(3);
        } else {
          setGridCols(2);
        }
        setError(null);
      } catch (err) {
        console.error("Failed to fetch cameras:", err);
        setError("Failed to load camera configuration. Please check your connection or permissions.");
      } finally {
        setLoading(false);
      }
    };

    fetchCameras();
  }, []);

  return (
    <DashboardLayout>
      <div className="p-8 h-full flex flex-col">
        <div className="mb-6 flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold text-cyberCyan mb-2 flex items-center">
            <Video className="w-8 h-8 mr-3" />
            Live Monitoring
          </h1>
          <p className="text-cyberText text-sm">Real-time low-latency RTSPS camera feeds</p>
        </div>
        <div className="flex items-center space-x-4 text-xs font-mono text-cyberMuted">
          <div className="flex space-x-1 border border-cyberDark rounded p-1">
            <button 
              onClick={() => setGridCols(1)} 
              className={`px-2 py-1 rounded transition-colors ${gridCols === 1 ? 'bg-cyberCyan/20 text-cyberCyan' : 'hover:bg-cyberDark'}`}
            >
              1x1
            </button>
            <button 
              onClick={() => setGridCols(2)} 
              className={`px-2 py-1 rounded transition-colors ${gridCols === 2 ? 'bg-cyberCyan/20 text-cyberCyan' : 'hover:bg-cyberDark'}`}
            >
              2x2
            </button>
            <button 
              onClick={() => setGridCols(3)} 
              className={`px-2 py-1 rounded transition-colors ${gridCols === 3 ? 'bg-cyberCyan/20 text-cyberCyan' : 'hover:bg-cyberDark'}`}
            >
              3x3
            </button>
          </div>
          <span className="px-2 py-1 bg-cyberPanel rounded border border-cyberDark">
            ACTIVE STREAMS: <span className="text-cyberCyan">{cameras.length}</span>
          </span>
        </div>
      </div>

      {loading ? (
        <div className="flex-1 flex flex-col items-center justify-center text-cyberMuted">
          <Loader2 className="w-12 h-12 animate-spin mb-4 text-cyberCyan" />
          <p>Initializing Control Room...</p>
        </div>
      ) : error ? (
        <div className="flex-1 flex flex-col items-center justify-center text-red-400">
          <ServerCrash className="w-16 h-16 mb-4 opacity-50" />
          <p className="max-w-md text-center">{error}</p>
        </div>
      ) : cameras.length === 0 ? (
        <div className="flex-1 flex flex-col items-center justify-center text-cyberMuted">
          <Video className="w-16 h-16 mb-4 opacity-20" />
          <p>No active cameras found in your zone.</p>
        </div>
      ) : (
        <div className={`grid gap-6 overflow-y-auto pb-8 ${gridCols === 1 ? 'grid-cols-1' : gridCols === 2 ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3'}`}>
          {cameras.map((camera) => (
            <CameraFeedCard 
              key={camera.id}
              cameraId={camera.id}
              cameraName={`${camera.name} (${camera.camera_code})`}
              location={camera.location}
              isActive={camera.is_active}
            />
          ))}
        </div>
        )}
      </div>
    </DashboardLayout>
  );
}
