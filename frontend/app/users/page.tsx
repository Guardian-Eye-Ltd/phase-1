"use client";

import React from "react";

export default function UsersPagePlaceholder() {
  return (
    <div className="p-8">
      <div className="glass-panel p-6 rounded-lg">
        <h1 className="text-3xl font-bold text-cyberSuccess mb-2">User Administration Portal</h1>
        <p className="text-cyberText text-sm mb-4">Manage control room operators and system privileges here.</p>
        <div className="text-xs text-cyberMuted font-mono">STATUS: PENDING_USER_MODULE</div>
      </div>
    </div>
  );
}
