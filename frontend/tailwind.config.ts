import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#080b11",
        cardBg: "rgba(16, 22, 35, 0.6)",
        cardBorder: "rgba(255, 255, 255, 0.08)",
        cyberCyan: "#00f0ff",
        cyberBlue: "#3b82f6",
        cyberDarkBlue: "#1e3a8a",
        cyberGlow: "rgba(0, 240, 255, 0.15)",
        cyberAlert: "#ef4444",
        cyberWarning: "#f59e0b",
        cyberSuccess: "#10b981",
        cyberText: "#e2e8f0",
        cyberMuted: "#64748b",
      },
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
      },
      boxShadow: {
        "cyan-glow": "0 0 15px rgba(0, 240, 255, 0.25)",
        "blue-glow": "0 0 15px rgba(59, 130, 246, 0.25)",
      },
    },
  },
  plugins: [],
};
export default config;
