"use client";

import React, { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import apiClient from "@/services/api";
import { useRouter } from "next/navigation";
import { ShieldCheck, Loader2, KeyRound, UserCheck } from "lucide-react";

export default function LoginPage() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const response = await apiClient.post("/auth/login", {
        username,
        password,
      });

      if (response.data && response.data.access_token) {
        await login(response.data.access_token, response.data.refresh_token);
        router.push("/dashboard");
      }
    } catch (err: any) {
      if (err.response && err.response.data && err.response.data.detail) {
        setError(err.response.data.detail);
      } else {
        setError("An unexpected error occurred during authorization.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleQuickLogin = (user: string, pass: string) => {
    setUsername(user);
    setPassword(pass);
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-[#070A11] p-4 relative overflow-hidden">
      {/* Background grid accent */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#1f293710_1px,transparent_1px),linear-gradient(to_bottom,#1f293710_1px,transparent_1px)] bg-[size:4rem_4rem]"></div>

      <div className="relative z-10 w-full max-w-md bg-[#0F172A]/90 backdrop-blur border border-cyberCyan/30 rounded-xl p-8 shadow-2xl shadow-cyberCyan/5">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center p-3 bg-cyberCyan/10 rounded-full border border-cyberCyan/30 mb-4">
            <ShieldCheck className="w-10 h-10 text-cyberCyan" />
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight">GuardianEye</h1>
          <p className="text-cyberCyan text-xs mt-1 font-mono tracking-wider uppercase">INTELLIGENT VIDEO DIGITAL FORENSICS</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-5">
          {error && (
            <div className="p-3 bg-red-500/10 border border-red-500/40 rounded-lg text-red-400 text-xs font-mono text-center">
              {error}
            </div>
          )}

          <div>
            <label className="block text-gray-400 text-xs font-mono mb-1.5 uppercase tracking-wider">Investigator ID / Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full bg-[#090D16] border border-cyberDark rounded-lg px-4 py-2.5 text-white placeholder-gray-600 focus:outline-none focus:border-cyberCyan transition-colors text-sm"
              placeholder="e.g. investigator"
              required
              disabled={loading}
            />
          </div>

          <div>
            <label className="block text-gray-400 text-xs font-mono mb-1.5 uppercase tracking-wider">Secure Passcode</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-[#090D16] border border-cyberDark rounded-lg px-4 py-2.5 text-white placeholder-gray-600 focus:outline-none focus:border-cyberCyan transition-colors text-sm"
              placeholder="Enter passcode"
              required
              disabled={loading}
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-cyberCyan hover:bg-cyan-400 text-gray-950 rounded-lg py-3 font-mono font-bold tracking-widest transition-all flex justify-center items-center shadow-lg shadow-cyberCyan/20 text-sm mt-2"
          >
            {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : "AUTHENTICATE & LOG IN"}
          </button>
        </form>

        {/* Preset quick login shortcuts */}
        <div className="mt-8 pt-6 border-t border-slate-800">
          <p className="text-[11px] font-mono text-gray-500 text-center uppercase tracking-wider mb-3">Quick Demo Credentials</p>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => handleQuickLogin("investigator", "Investigator123!")}
              className="flex items-center justify-center p-2 rounded-lg bg-slate-900 border border-slate-800 hover:border-cyberCyan/50 text-xs font-mono text-gray-300 transition-colors"
            >
              <UserCheck className="w-3.5 h-3.5 mr-1.5 text-cyberCyan" />
              Investigator
            </button>
            <button
              type="button"
              onClick={() => handleQuickLogin("admin", "Admin123!")}
              className="flex items-center justify-center p-2 rounded-lg bg-slate-900 border border-slate-800 hover:border-cyberCyan/50 text-xs font-mono text-gray-300 transition-colors"
            >
              <KeyRound className="w-3.5 h-3.5 mr-1.5 text-amber-400" />
              Admin
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
