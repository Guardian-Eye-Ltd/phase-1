"use client";

import React, { useEffect, useState } from "react";
import DashboardLayout from "@/components/DashboardLayout";
import apiClient from "@/services/api";
import { Camera, Server, AlertTriangle, ShieldAlert, Activity, CheckCircle, XCircle } from "lucide-react";

export default function SystemStatusPage() {
  const [stats, setStats] = useState({ total: 0, active: 0, offline: 0 });
  const [health, setHealth] = useState({ cpu_usage: 0, memory_usage: 0, disk_usage: 0 });
  const [alerts, setAlerts] = useState<any[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsRes, healthRes, alertsRes] = await Promise.all([
          apiClient.get("/cameras/stats"),
          apiClient.get("/health/system"),
          apiClient.get("/alerts?limit=5")
        ]);
        setStats(statsRes.data);
        setHealth(healthRes.data);
        setAlerts(alertsRes.data);
      } catch (error) {
        console.error("Failed to fetch dashboard data", error);
      }
    };
    
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <DashboardLayout>
      <div className="p-8 space-y-6">
        <h1 className="text-3xl font-bold text-cyberCyan mb-6 flex items-center">
          <Activity className="w-8 h-8 mr-3" /> System Status
        </h1>

        {/* Top Widgets Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          
          {/* Camera Network Stats */}
          <div className="glass-panel p-6 rounded-lg relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4 opacity-10">
              <Camera className="w-24 h-24" />
            </div>
            <h2 className="text-lg font-bold text-cyberText mb-4 font-mono uppercase tracking-widest flex items-center">
              <Camera className="w-5 h-5 mr-2 text-cyberCyan" /> Camera Network
            </h2>
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-cyberDarker p-4 rounded text-center border border-cyberDark">
                <p className="text-3xl font-bold text-cyberCyan">{stats.total}</p>
                <p className="text-xs text-cyberMuted mt-1">TOTAL</p>
              </div>
              <div className="bg-cyberDarker p-4 rounded text-center border border-green-900/30">
                <p className="text-3xl font-bold text-green-500">{stats.active}</p>
                <p className="text-xs text-green-500/70 mt-1 flex items-center justify-center"><CheckCircle className="w-3 h-3 mr-1"/> ACTIVE</p>
              </div>
              <div className="bg-cyberDarker p-4 rounded text-center border border-red-900/30">
                <p className="text-3xl font-bold text-red-500">{stats.offline}</p>
                <p className="text-xs text-red-500/70 mt-1 flex items-center justify-center"><XCircle className="w-3 h-3 mr-1"/> OFFLINE</p>
              </div>
            </div>
          </div>

          {/* System Health */}
          <div className="glass-panel p-6 rounded-lg relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4 opacity-10">
              <Server className="w-24 h-24" />
            </div>
            <h2 className="text-lg font-bold text-cyberText mb-4 font-mono uppercase tracking-widest flex items-center">
              <Server className="w-5 h-5 mr-2 text-cyberCyan" /> Core Systems
            </h2>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-cyberMuted">CPU Usage</span>
                  <span className="text-cyberCyan font-bold">{health.cpu_usage}%</span>
                </div>
                <div className="w-full bg-cyberDark rounded-full h-2">
                  <div className="bg-cyberCyan h-2 rounded-full transition-all duration-500" style={{ width: `${health.cpu_usage}%` }}></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-cyberMuted">Memory Allocation</span>
                  <span className="text-cyberCyan font-bold">{health.memory_usage}%</span>
                </div>
                <div className="w-full bg-cyberDark rounded-full h-2">
                  <div className="bg-cyberCyan h-2 rounded-full transition-all duration-500" style={{ width: `${health.memory_usage}%` }}></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-cyberMuted">Storage Array</span>
                  <span className="text-cyberCyan font-bold">{health.disk_usage}%</span>
                </div>
                <div className="w-full bg-cyberDark rounded-full h-2">
                  <div className="bg-cyberCyan h-2 rounded-full transition-all duration-500" style={{ width: `${health.disk_usage}%` }}></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Alerts */}
        <div className="glass-panel p-6 rounded-lg mt-6">
          <h2 className="text-lg font-bold text-cyberText mb-4 font-mono uppercase tracking-widest flex items-center">
            <ShieldAlert className="w-5 h-5 mr-2 text-cyberCyan" /> Recent Alerts
          </h2>
          {alerts.length === 0 ? (
            <p className="text-cyberMuted text-sm py-4">No recent security alerts detected.</p>
          ) : (
            <div className="space-y-3">
              {alerts.map((alert) => (
                <div key={alert.id} className="flex items-center justify-between bg-cyberDarker p-4 rounded border-l-4 border-cyberCyan">
                  <div className="flex items-center">
                    <AlertTriangle className={`w-5 h-5 mr-4 ${alert.severity === 'HIGH' ? 'text-red-500' : 'text-yellow-500'}`} />
                    <div>
                      <p className="text-sm font-bold text-cyberText">{alert.alert_type.replace(/_/g, ' ')}</p>
                      <p className="text-xs text-cyberMuted mt-1">{alert.description}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <span className="text-xs font-mono text-cyberMuted bg-cyberDark px-2 py-1 rounded">{alert.status}</span>
                    <p className="text-[10px] text-cyberMuted mt-2">{new Date(alert.timestamp).toLocaleString()}</p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
}
