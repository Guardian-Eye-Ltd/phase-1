"use client";

import React, { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import apiClient from "@/services/api";
import { useRouter } from "next/navigation";
import { ShieldAlert, Loader2 } from "lucide-react";

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
        router.push("/realtime-monitoring");
      }
    } catch (err: any) {
      if (err.response && err.response.data && err.response.data.detail) {
        setError(err.response.data.detail);
      } else {
        setError("An unexpected error occurred during authentication.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <div className="glass-panel p-8 rounded-lg max-w-md w-full">
        <div className="text-center mb-8">
          <ShieldAlert className="w-12 h-12 mx-auto text-cyberCyan mb-4" />
          <h1 className="text-2xl font-bold text-cyberCyan">GuardianEye</h1>
          <p className="text-cyberText text-sm mt-2 font-mono">SECURE ACCESS TERMINAL</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {error && (
            <div className="p-3 bg-red-500/10 border border-red-500/50 rounded text-red-500 text-sm text-center">
              {error}
            </div>
          )}

          <div>
            <label className="block text-cyberMuted text-xs font-mono mb-2 uppercase">Operator ID</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full bg-cyberDarker border border-cyberDark rounded px-4 py-2 text-cyberText focus:outline-none focus:border-cyberCyan transition-colors"
              placeholder="Enter your username"
              required
              disabled={loading}
            />
          </div>

          <div>
            <label className="block text-cyberMuted text-xs font-mono mb-2 uppercase">Passcode</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-cyberDarker border border-cyberDark rounded px-4 py-2 text-cyberText focus:outline-none focus:border-cyberCyan transition-colors"
              placeholder="Enter your password"
              required
              disabled={loading}
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full bg-cyberCyan/10 hover:bg-cyberCyan/20 text-cyberCyan border border-cyberCyan/50 rounded py-3 font-mono font-bold tracking-widest transition-all flex justify-center items-center"
          >
            {loading ? <Loader2 className="w-5 h-5 animate-spin" /> : "AUTHENTICATE"}
          </button>
        </form>
      </div>
    </div>
  );
}
