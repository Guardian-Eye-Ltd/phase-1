"use client";

import React from "react";
import { useAuth } from "@/contexts/AuthContext";
import ProtectedRoute from "./ProtectedRoute";
import { LogOut, LayoutDashboard, Video, ShieldAlert, Activity } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuth();
  const pathname = usePathname();

  return (
    <ProtectedRoute>
      <div className="flex h-screen bg-background overflow-hidden">
        {/* Sidebar */}
        <aside className="w-64 bg-cyberDarker border-r border-cyberDark flex flex-col">
          <div className="p-6 border-b border-cyberDark flex items-center">
            <ShieldAlert className="w-8 h-8 text-cyberCyan mr-3" />
            <div>
              <h2 className="font-bold text-cyberCyan tracking-wide">GuardianEye</h2>
              <p className="text-[10px] text-cyberMuted font-mono">SECURE SYSCOM</p>
            </div>
          </div>
          
          <nav className="flex-1 py-6 px-4 space-y-2">
            <Link 
              href="/dashboard"
              className={`flex items-center px-4 py-3 rounded-lg transition-colors ${
                pathname === "/dashboard" 
                  ? "bg-cyberCyan/10 text-cyberCyan border border-cyberCyan/30" 
                  : "text-cyberText hover:bg-cyberDark hover:text-cyberCyan"
              }`}
            >
              <LayoutDashboard className="w-5 h-5 mr-3" />
              <span>Dashboard</span>
            </Link>
            <Link 
              href="/realtime-monitoring"
              className={`flex items-center px-4 py-3 rounded-lg transition-colors ${
                pathname === "/realtime-monitoring" 
                  ? "bg-cyberCyan/10 text-cyberCyan border border-cyberCyan/30" 
                  : "text-cyberText hover:bg-cyberDark hover:text-cyberCyan"
              }`}
            >
              <Video className="w-5 h-5 mr-3" />
              <span>Live Monitoring</span>
            </Link>
            <Link 
              href="/system-status"
              className={`flex items-center px-4 py-3 rounded-lg transition-colors ${
                pathname === "/system-status" 
                  ? "bg-cyberCyan/10 text-cyberCyan border border-cyberCyan/30" 
                  : "text-cyberText hover:bg-cyberDark hover:text-cyberCyan"
              }`}
            >
              <Activity className="w-5 h-5 mr-3" />
              <span>System Status</span>
            </Link>
          </nav>

          {/* User Profile */}
          <div className="p-4 border-t border-cyberDark bg-cyberPanel/30">
            <div className="flex items-center justify-between mb-4">
              <div className="truncate">
                <p className="text-sm font-semibold text-cyberText truncate">{user?.full_name}</p>
                <div className="flex items-center mt-1">
                  <span className="w-2 h-2 rounded-full bg-green-500 mr-2 animate-pulse"></span>
                  <p className="text-xs text-cyberMuted font-mono truncate uppercase">{user?.role?.role_name || "Operator"}</p>
                </div>
              </div>
            </div>
            <button
              onClick={() => logout()}
              className="w-full flex items-center justify-center px-4 py-2 text-sm text-red-400 bg-red-400/10 hover:bg-red-400/20 border border-red-400/20 rounded transition-colors"
            >
              <LogOut className="w-4 h-4 mr-2" />
              Disconnect
            </button>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 overflow-auto bg-background">
          {children}
        </main>
      </div>
    </ProtectedRoute>
  );
}
