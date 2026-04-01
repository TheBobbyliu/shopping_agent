"use client";

import { useState, useEffect, useCallback } from "react";
import ChatWindow from "@/components/ChatWindow";
import ChatSidebar from "@/components/ChatSidebar";
import { checkReady } from "@/lib/api";

export default function Home() {
  const [ready, setReady]                 = useState(false);
  const [activeSession, setActiveSession] = useState<string | undefined>();

  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      while (!cancelled) {
        const ok = await checkReady();
        if (ok) { if (!cancelled) setReady(true); return; }
        await new Promise((r) => setTimeout(r, 2000));
      }
    };
    poll();
    return () => { cancelled = true; };
  }, []);

  const [mountKey, setMountKey] = useState(0);

  // Called when user explicitly picks a session from sidebar → remount ChatWindow
  const handleSelectSession = useCallback((id: string) => {
    setActiveSession(id);
    setMountKey((k) => k + 1);
  }, []);

  // Called when user clicks "New chat" → remount ChatWindow with blank state
  const handleNewChat = useCallback(() => {
    setActiveSession(undefined);
    setMountKey((k) => k + 1);
  }, []);

  // Called when a brand-new session is created mid-chat → update sidebar only, no remount
  const handleSessionCreated = useCallback((id: string) => setActiveSession(id), []);

  if (!ready) {
    return (
      <main className="flex items-center justify-center h-full bg-vd-bg">
        <div className="flex flex-col items-center gap-5 animate-fade-up">
          {/* Brand mark */}
          <div className="relative">
            <div className="w-16 h-16 rounded-2xl bg-vd-surface border border-vd-border flex items-center justify-center shadow-card">
              <span className="font-display italic text-2xl text-vd-accent font-light tracking-tight">V</span>
            </div>
            {/* Glow ring */}
            <div className="absolute inset-0 rounded-2xl shadow-accent-glow pointer-events-none" />
          </div>

          <div className="text-center">
            <p className="font-display italic text-vd-fg text-lg font-light tracking-wide mb-1">
              Verdure
            </p>
            <p className="text-vd-fg-3 text-xs font-light tracking-widest uppercase">
              Loading models
            </p>
          </div>

          {/* Shimmer bar */}
          <div className="relative w-32 h-px bg-vd-border overflow-hidden rounded-full">
            <div className="absolute inset-0 rounded-full bg-vd-accent opacity-20" />
            <div
              className="absolute inset-0 rounded-full bg-accent-shimmer bg-[length:200%_100%] animate-shimmer"
            />
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="flex h-full overflow-hidden bg-vd-bg">
      <ChatSidebar
        activeSessionId={activeSession}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
      />
      <div className="flex-1 min-w-0 flex flex-col border-l border-vd-border">
        <ChatWindow
          key={mountKey}
          sessionId={activeSession}
          onSessionCreated={handleSessionCreated}
        />
      </div>
    </main>
  );
}
