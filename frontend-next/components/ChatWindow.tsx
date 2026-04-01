"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { streamChat } from "@/lib/api";
import ChatMessage, { parseItemIds } from "@/components/ChatMessage";
import ChatInput from "@/components/ChatInput";
import StatusIndicator from "@/components/StatusIndicator";
import { saveSession } from "@/components/ChatSidebar";

export interface Message {
  role: "user" | "assistant";
  content: string;
  imagePreview?: string;
  products?: string[];
  toolCalls?: { tool: string }[];
  streaming?: boolean;
}

const SUGGESTIONS = [
  "Show me a comfortable chair",
  "Red leather sofa under $500",
  "Running shoes for men",
  "Modern kitchen accessories",
];

interface ChatWindowProps {
  sessionId?: string;
  onSessionCreated?: (id: string) => void;
}

export default function ChatWindow({ sessionId: initialSessionId, onSessionCreated }: ChatWindowProps) {
  const [messages, setMessages]     = useState<Message[]>(() => {
    // Restore messages for an existing session from localStorage (Bug 2)
    if (!initialSessionId || typeof window === "undefined") return [];
    try {
      const stored = localStorage.getItem(`messages_${initialSessionId}`);
      return stored ? JSON.parse(stored) : [];
    } catch { return []; }
  });
  const [input, setInput]           = useState("");
  const [loading, setLoading]       = useState(false);
  const [status, setStatus]         = useState("");
  const [sessionId, setSessionId]   = useState<string | undefined>(initialSessionId);
  const [pendingImg, setPendingImg] = useState<string | null>(null);
  const bottomRef    = useRef<HTMLDivElement>(null);
  const abortRef     = useRef<AbortController | null>(null);
  // Ref tracks the latest sessionId so effects can read it without stale closures
  const sessionIdRef = useRef<string | undefined>(initialSessionId);

  // Note: session switching is handled by the `key` prop on this component in page.tsx,
  // which causes a full remount — no effect needed here for external session changes.

  // Keep sessionIdRef in sync — must be declared BEFORE the messages-save effect
  // so React runs it first when both sessionId and messages change in the same render.
  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);

  // Persist messages to localStorage whenever streaming finishes.
  // Keep user image previews so reopening a chat shows the original upload.
  // Using an effect (not inside the setMessages updater) because the updater
  // runs during the React render cycle, not synchronously on setState call.
  useEffect(() => {
    const sid = sessionIdRef.current;
    if (!sid || messages.length === 0) return;
    if (messages.some((m) => m.streaming)) return; // still streaming — wait
    try {
      localStorage.setItem(`messages_${sid}`, JSON.stringify(messages));
    } catch { /* ignore quota errors */ }
  }, [messages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, status]);

  const send = useCallback(async (text?: string) => {
    const msg = (text ?? input).trim();
    if (!msg && !pendingImg) return;
    if (loading) return;

    const userMsg: Message = {
      role:         "user",
      content:      msg || "(image search)",
      imagePreview: pendingImg ? `data:image/jpeg;base64,${pendingImg}` : undefined,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    const imgToSend = pendingImg;
    setPendingImg(null);
    setLoading(true);
    setStatus("Connecting...");

    // Note: we do NOT add a placeholder streaming message here.
    // The StatusIndicator shows "Connecting..." while we wait for the first token.
    // The streaming message is created on the first onToken call (Bug 4 fix).

    abortRef.current = streamChat(
      {
        message:    msg || "Find products similar to this image.",
        image_b64:  imgToSend ?? undefined,
        session_id: sessionId,
      },
      {
        onStatus: (text) => setStatus(text),
        onToken:  (token) => {
          setMessages((prev) => {
            const last = prev[prev.length - 1];
            if (last?.role === "assistant" && last.streaming) {
              // Append to existing streaming message
              const updated = [...prev];
              updated[updated.length - 1] = { ...last, content: last.content + token };
              return updated;
            }
            // First token — create the streaming assistant message
            return [...prev, { role: "assistant", content: token, streaming: true }];
          });
        },
        onDone: (res) => {
          const newSessionId = res.session_id;

          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last?.role === "assistant" && last.streaming) {
              const finalContent = res.reply || last.content;
              updated[updated.length - 1] = {
                ...last,
                content:   finalContent,
                products:  parseItemIds(finalContent),
                toolCalls: res.tool_calls,
                streaming: false,
              };
            } else {
              // No streaming tokens received — add final message directly
              const finalContent = res.reply;
              if (finalContent) {
                updated.push({
                  role:      "assistant",
                  content:   finalContent,
                  products:  parseItemIds(finalContent),
                  toolCalls: res.tool_calls,
                  streaming: false,
                });
              }
            }
            return updated;
            // localStorage persistence is handled by the useEffect([messages]) above
          });

          // Only save session metadata and notify parent on first message (Bug 5)
          if (!sessionId) {
            setSessionId(newSessionId);
            onSessionCreated?.(newSessionId);
            const title = msg.slice(0, 60) || "Image search";
            saveSession({ session_id: newSessionId, title, created_at: Date.now() });
          }
          setLoading(false);
          setStatus("");
        },
        onError: (detail) => {
          setMessages((prev) => {
            const updated = [...prev];
            const last = updated[updated.length - 1];
            if (last?.role === "assistant" && last.streaming) {
              updated[updated.length - 1] = {
                ...last,
                content:   detail || "Something went wrong. Please try again.",
                streaming: false,
              };
            } else {
              updated.push({ role: "assistant", content: detail || "Something went wrong." });
            }
            return updated;
          });
          setLoading(false);
          setStatus("");
        },
      }
    );
  }, [input, pendingImg, sessionId, loading, onSessionCreated]);

  const askAbout = useCallback((itemId: string) => send(`Tell me more about product ${itemId}`), [send]);

  return (
    <div className="flex flex-col h-full w-full bg-vd-bg">
      {/* Header */}
      <header className="border-b border-vd-border px-5 py-3.5 flex items-center gap-3 flex-shrink-0 bg-vd-panel/50">
        <div className="flex items-center gap-2.5 flex-1 min-w-0">
          <div className="w-6 h-6 rounded-lg bg-vd-surface border border-vd-border flex items-center justify-center flex-shrink-0">
            <span className="font-display italic text-xs text-vd-accent font-light leading-none">V</span>
          </div>
          <span className="font-display italic text-vd-fg text-sm font-light tracking-wide">
            {messages.length > 0
              ? messages.find((m) => m.role === "user")?.content.slice(0, 40) || "Conversation"
              : "New conversation"}
          </span>
        </div>
        {sessionId && (
          <span className="text-[10px] text-vd-fg-3 font-mono tracking-widest hidden sm:block">
            {sessionId.slice(0, 8)}
          </span>
        )}
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          /* Welcome / centered search state */
          <div className="flex flex-col items-center justify-center h-full px-6 pb-16">
            <div className="relative mb-6 animate-fade-up">
              <div className="w-16 h-16 rounded-2xl bg-vd-surface border border-vd-border flex items-center justify-center shadow-card">
                <span className="font-display italic text-2xl text-vd-accent font-light tracking-tight">V</span>
              </div>
              <div className="absolute inset-0 rounded-2xl shadow-accent-glow pointer-events-none" />
            </div>

            <div className="text-center mb-8 animate-fade-up" style={{ animationDelay: "80ms" }}>
              <h2 className="font-display italic text-vd-fg text-2xl font-light tracking-wide mb-2">
                What are you looking for?
              </h2>
              <p className="text-vd-fg-3 text-sm font-light">
                Describe a product or upload an image to get started
              </p>
            </div>

            <div className="flex flex-wrap gap-2 justify-center max-w-md animate-fade-up" style={{ animationDelay: "160ms" }}>
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="px-4 py-2 rounded-full bg-vd-surface border border-vd-border text-vd-fg-2 text-sm font-light hover:border-vd-accent/30 hover:text-vd-fg hover:bg-vd-hover transition-all duration-200 tracking-wide"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto w-full px-5 py-7 space-y-7">
            {messages.map((m, i) => (
              <ChatMessage key={i} message={m} onAsk={askAbout} />
            ))}

            {/* Live status indicator */}
            {loading && status && (
              <div className="flex gap-3">
                <div className="w-6 h-6 rounded-lg bg-vd-surface border border-vd-border flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="font-display italic text-xs text-vd-accent font-light leading-none">V</span>
                </div>
                <StatusIndicator text={status} />
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Separator */}
      <div className="border-t border-vd-border flex-shrink-0">
        <ChatInput
          value={input}
          onChange={setInput}
          onSend={() => send()}
          onImageSelect={setPendingImg}
          disabled={loading}
          pendingImg={pendingImg}
          onClearImg={() => setPendingImg(null)}
        />
      </div>
    </div>
  );
}
