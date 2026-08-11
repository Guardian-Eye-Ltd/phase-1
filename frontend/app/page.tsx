"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function RootPage() {
  const router = useRouter();

  useEffect(() => {
    // Redirect to login page by default
    router.push("/dashboard");
  }, [router]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <div className="text-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-cyberCyan border-t-transparent mx-auto"></div>
        <p className="mt-4 text-cyberMuted font-mono text-sm tracking-wider">INITIALIZING GUARDIANEYE SECURITY PROTOCOLS...</p>
      </div>
    </div>
  );
}
