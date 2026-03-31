import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Shopping Assistant",
  description: "AI-powered product search — bge-visualized-m3 · Elasticsearch · Claude",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="h-screen flex flex-col">{children}</body>
    </html>
  );
}
