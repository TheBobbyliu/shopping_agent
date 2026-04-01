import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        vd: {
          bg:          "#0b0f0d",
          surface:     "#141c17",
          panel:       "#0f1512",
          hover:       "#1c2820",
          border:      "#1f2d24",
          muted:       "#2a3d30",
          fg:          "#e8ede9",
          "fg-2":      "#8faa96",
          "fg-3":      "#4d6655",
          accent:      "#6ee7a0",
          "accent-dim":"#3d9e65",
          "accent-bg": "rgba(110,231,160,0.08)",
        },
      },
      fontFamily: {
        display: ["Fraunces", "serif"],
        sans:    ["Outfit", "sans-serif"],
        mono:    ["JetBrains Mono", "monospace"],
      },
      keyframes: {
        "fade-up": {
          "0%":   { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-dot": {
          "0%, 100%": { opacity: "0.3", transform: "scale(0.75)" },
          "50%":      { opacity: "1",   transform: "scale(1)" },
        },
        "cursor-blink": {
          "0%, 100%": { opacity: "1" },
          "50%":      { opacity: "0" },
        },
        "blink": {
          "0%, 80%, 100%": { opacity: "0.15" },
          "40%":           { opacity: "1" },
        },
        "shimmer": {
          "0%":   { backgroundPosition: "-200% center" },
          "100%": { backgroundPosition: "200% center" },
        },
      },
      animation: {
        "fade-up":      "fade-up 0.35s ease forwards",
        "pulse-dot":    "pulse-dot 1.6s ease-in-out infinite",
        "cursor-blink": "cursor-blink 0.9s step-end infinite",
        "blink":        "blink 1.4s ease-in-out infinite",
        "blink-2":      "blink 1.4s ease-in-out infinite 0.2s",
        "blink-3":      "blink 1.4s ease-in-out infinite 0.4s",
        "shimmer":      "shimmer 2s linear infinite",
      },
      backgroundImage: {
        "accent-shimmer": "linear-gradient(90deg, transparent 0%, rgba(110,231,160,0.4) 50%, transparent 100%)",
      },
      boxShadow: {
        "accent-glow": "0 0 20px rgba(110,231,160,0.12), 0 0 40px rgba(110,231,160,0.04)",
        "card":        "0 1px 3px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.04)",
        "card-hover":  "0 4px 16px rgba(0,0,0,0.5), 0 0 0 1px rgba(110,231,160,0.15)",
      },
    },
  },
  plugins: [],
};

export default config;
