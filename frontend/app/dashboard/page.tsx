"use client";

import React from "react";
import DashboardLayout from "@/components/DashboardLayout";
import { 
  Monitor, Brain, Bell, ShieldAlert, Play, 
  Search, UserSquare2, FileText, ShieldCheck, 
  Lock, Shield, Cloud, Zap, Camera
} from "lucide-react";
import { useRouter } from "next/navigation";

export default function DashboardPage() {
  const router = useRouter();

  return (
    <DashboardLayout>
      <div className="p-8 h-full flex flex-col items-center overflow-y-auto">
        
        {/* Header Section */}
        <div className="text-center mb-10 mt-4">
          <h1 className="text-4xl font-bold text-white mb-3">
            Welcome to <span className="text-cyberCyan">GuardianEye</span>
          </h1>
          <p className="text-cyberMuted text-sm max-w-2xl mx-auto">
            AI-Powered Smart Surveillance & Forensic Intelligence System.<br/>
            Choose a module to get started.
          </p>
        </div>

        {/* Main Cards Container */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 w-full max-w-6xl mb-12">
          
          {/* REAL-TIME MONITORING CARD */}
          <div className="relative group rounded-2xl border border-cyberCyan/30 bg-cyberDarker/50 overflow-hidden flex flex-col hover:border-cyberCyan/60 transition-colors duration-500 shadow-[0_0_30px_rgba(0,195,255,0.05)] hover:shadow-[0_0_40px_rgba(0,195,255,0.1)]">
            
            {/* Top Image area placeholder (dark gradient with some lines) */}
            <div className="h-48 w-full bg-gradient-to-b from-cyberCyan/10 to-transparent relative border-b border-cyberCyan/10">
              <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-20 mix-blend-overlay"></div>
              {/* Center floating icon */}
              <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 w-16 h-16 rounded-full bg-cyberDarker border-2 border-cyberCyan flex items-center justify-center shadow-[0_0_20px_rgba(0,195,255,0.4)] z-10">
                <Camera className="w-8 h-8 text-cyberCyan" />
              </div>
            </div>

            <div className="pt-12 pb-8 px-8 flex flex-col flex-grow text-center">
              <h2 className="text-2xl font-bold text-white mb-2 tracking-wide">REAL-TIME <br/> MONITORING SYSTEM</h2>
              <p className="text-cyberCyan text-sm font-semibold mb-8">Live Detection. Instant Response. Safer Cities.</p>
              
              {/* Feature Icons */}
              <div className="grid grid-cols-4 gap-2 mb-8">
                <div className="flex flex-col items-center text-cyberMuted">
                  <Monitor className="w-6 h-6 mb-2 text-white/70" />
                  <span className="text-[10px] leading-tight text-center">Live Feed<br/>Monitoring</span>
                </div>
                <div className="flex flex-col items-center text-cyberMuted">
                  <Brain className="w-6 h-6 mb-2 text-white/70" />
                  <span className="text-[10px] leading-tight text-center">AI-Powered<br/>Incident Detection</span>
                </div>
                <div className="flex flex-col items-center text-cyberMuted">
                  <Bell className="w-6 h-6 mb-2 text-white/70" />
                  <span className="text-[10px] leading-tight text-center">Automated Alerts<br/>& Dispatch</span>
                </div>
                <div className="flex flex-col items-center text-cyberMuted">
                  <ShieldAlert className="w-6 h-6 mb-2 text-white/70" />
                  <span className="text-[10px] leading-tight text-center">Emergency<br/>Response</span>
                </div>
              </div>

              <div className="mt-auto">
                <button 
                  onClick={() => router.push('/realtime-monitoring')}
                  className="w-full py-4 rounded-lg bg-gradient-to-r from-blue-700 to-cyberCyan text-white font-bold flex items-center justify-center hover:opacity-90 transition-opacity shadow-[0_0_15px_rgba(0,195,255,0.4)]"
                >
                  <Play className="w-5 h-5 mr-2 fill-white" />
                  Enter Real-Time Monitoring
                </button>
              </div>
            </div>
          </div>

          {/* POST-INCIDENT MONITORING CARD */}
          <div className="relative group rounded-2xl border border-red-500/30 bg-cyberDarker/50 overflow-hidden flex flex-col hover:border-red-500/60 transition-colors duration-500 shadow-[0_0_30px_rgba(239,68,68,0.05)] hover:shadow-[0_0_40px_rgba(239,68,68,0.1)]">
            
            {/* Top Image area placeholder */}
            <div className="h-48 w-full bg-gradient-to-b from-red-500/10 to-transparent relative border-b border-red-500/10">
              <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-20 mix-blend-overlay"></div>
              {/* Center floating icon */}
              <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 w-16 h-16 rounded-full bg-cyberDarker border-2 border-red-500 flex items-center justify-center shadow-[0_0_20px_rgba(239,68,68,0.4)] z-10">
                <FileText className="w-8 h-8 text-red-500" />
              </div>
            </div>

            <div className="pt-12 pb-8 px-8 flex flex-col flex-grow text-center">
              <h2 className="text-2xl font-bold text-white mb-2 tracking-wide">POST-INCIDENT <br/> MONITORING SYSTEM</h2>
              <p className="text-red-400 text-sm font-semibold mb-8">Search Smarter. Analyze Faster. Build Stronger Cases.</p>
              
              {/* Feature Icons */}
              <div className="grid grid-cols-4 gap-2 mb-8">
                <div className="flex flex-col items-center text-cyberMuted">
                  <Search className="w-6 h-6 mb-2 text-white/70" />
                  <span className="text-[10px] leading-tight text-center">Semantic Search<br/>(Natural Language)</span>
                </div>
                <div className="flex flex-col items-center text-cyberMuted">
                  <UserSquare2 className="w-6 h-6 mb-2 text-white/70" />
                  <span className="text-[10px] leading-tight text-center">Face Recognition<br/>& Matching</span>
                </div>
                <div className="flex flex-col items-center text-cyberMuted">
                  <FileText className="w-6 h-6 mb-2 text-white/70" />
                  <span className="text-[10px] leading-tight text-center">Forensic Brief<br/>Generation</span>
                </div>
                <div className="flex flex-col items-center text-cyberMuted">
                  <ShieldCheck className="w-6 h-6 mb-2 text-white/70" />
                  <span className="text-[10px] leading-tight text-center">Evidence Integrity<br/>& Chain of Custody</span>
                </div>
              </div>

              <div className="mt-auto">
                <button 
                  onClick={() => router.push('/forensics')}
                  className="w-full py-4 rounded-lg bg-gradient-to-r from-red-800 to-red-500 text-white font-bold flex items-center justify-center hover:opacity-90 transition-opacity shadow-[0_0_15px_rgba(239,68,68,0.4)]"
                >
                  <Search className="w-5 h-5 mr-2" />
                  Enter Post-Incident Monitoring
                </button>
              </div>
            </div>
          </div>

        </div>

        {/* Footer Features Row */}
        <div className="w-full max-w-6xl grid grid-cols-2 md:grid-cols-4 gap-4 mt-auto pt-8 border-t border-cyberDark text-cyberMuted">
          <div className="flex items-center justify-center space-x-3">
            <Lock className="w-5 h-5" />
            <div className="text-left">
              <p className="text-xs font-semibold text-white">Data Integrity</p>
              <p className="text-[10px]">SHA-256 Encrypted</p>
            </div>
          </div>
          <div className="flex items-center justify-center space-x-3">
            <Shield className="w-5 h-5" />
            <div className="text-left">
              <p className="text-xs font-semibold text-white">System Security</p>
              <p className="text-[10px]">Multi-Layer Protected</p>
            </div>
          </div>
          <div className="flex items-center justify-center space-x-3">
            <Cloud className="w-5 h-5" />
            <div className="text-left">
              <p className="text-xs font-semibold text-white">AI Orchestrator</p>
              <p className="text-[10px]">Multi-Agent Framework</p>
            </div>
          </div>
          <div className="flex items-center justify-center space-x-3">
            <Zap className="w-5 h-5" />
            <div className="text-left">
              <p className="text-xs font-semibold text-white">Low Latency</p>
              <p className="text-[10px]">Real-time Processing</p>
            </div>
          </div>
        </div>

      </div>
    </DashboardLayout>
  );
}
