import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Verdure — Shopping Assistant",
  description: "AI-powered product discovery — bge-visualized-m3 · Elasticsearch · Claude",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
