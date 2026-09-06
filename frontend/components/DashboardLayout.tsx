"use client";

import React from "react";
import { useAuth } from "@/contexts/AuthContext";
import ProtectedRoute from "./ProtectedRoute";
import { LogOut, LayoutDashboard, Film, UploadCloud, ShieldCheck, Activity, FileText } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const { user, logout } = useAuth();
  const pathname = usePathname();

  const navItems = [
    { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
    { href: "/evidence", label: "Evidence Vault", icon: Film },
    { href: "/evidence/upload", label: "Upload Evidence", icon: UploadCloud },
    { href: "/audit", label: "Audit Log", icon: FileText },
    { href: "/system-status", label: "System Status", icon: Activity },
  ];

  return (
    <ProtectedRoute>
      <div className="flex h-screen bg-background text-cyberText overflow-hidden">
        {/* Sidebar */}
        <aside className="w-64 bg-cyberDarker border-r border-cyberDark flex flex-col z-20">
          <div className="p-5 border-b border-cyberDark flex items-center">
            <ShieldCheck className="w-8 h-8 text-cyberCyan mr-3 shrink-0" />
            <div>
              <h2 className="font-bold text-cyberCyan tracking-wide text-base">GuardianEye</h2>
              <p className="text-[10px] text-cyberMuted font-mono">DIGITAL FORENSICS v1.0</p>
            </div>
          </div>

          <nav className="flex-1 py-6 px-3 space-y-1.5 overflow-y-auto">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = pathname === item.href || (item.href !== "/dashboard" && pathname?.startsWith(item.href));
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`flex items-center px-4 py-3 rounded-lg text-sm font-medium transition-all ${isActive
                      ? "bg-cyberCyan/15 text-cyberCyan border border-cyberCyan/40 shadow-sm shadow-cyberCyan/10 font-semibold"
                      : "text-gray-400 hover:bg-cyberDark hover:text-cyberCyan"
                    }`}
                >
                  <Icon className="w-5 h-5 mr-3 shrink-0" />
                  <span>{item.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* User Profile Footer */}
          <div className="p-4 border-t border-cyberDark bg-cyberPanel/40">
            <div className="flex items-center justify-between mb-3">
              <div className="truncate">
                <p className="text-sm font-semibold text-white truncate">{user?.full_name || "Investigator"}</p>
                <div className="flex items-center mt-1">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 mr-2 animate-pulse"></span>
                  <p className="text-[11px] text-cyberCyan font-mono truncate uppercase tracking-wider">
                    {user?.role?.role_name || "INVESTIGATOR"}
                  </p>
                </div>
              </div>
            </div>
            <button
              onClick={() => logout()}
              className="w-full flex items-center justify-center px-3 py-2 text-xs font-mono font-medium text-red-400 bg-red-500/10 hover:bg-red-500/20 border border-red-500/30 rounded transition-colors"
            >
              <LogOut className="w-3.5 h-3.5 mr-2" />
              TERMINATE SESSION
            </button>
          </div>
        </aside>

        {/* Main Content Viewport */}
        <main className="flex-1 overflow-auto bg-[#090D16] relative flex flex-col">
          {/* Top Bar */}
          <header className="h-14 border-b border-cyberDark bg-cyberDarker/70 backdrop-blur px-6 flex items-center justify-between shrink-0 sticky top-0 z-10">
            <div className="flex items-center space-x-2 text-xs font-mono text-cyberMuted">
              <span className="text-cyberCyan font-semibold">GUARDIANEYE</span>
              <span>/</span>
              <span className="text-gray-300 capitalize">{pathname?.split("/")[1] || "dashboard"}</span>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center text-xs font-mono px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <span className="w-2 h-2 rounded-full bg-emerald-400 mr-2"></span>
                SYSTEM ONLINE
              </div>
            </div>
          </header>

          <div className="flex-1 p-6">
            {children}
          </div>
        </main>
      </div>
    </ProtectedRoute>
  );
}
