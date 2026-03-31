"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { sendChat, productImageUrl, getProduct } from "@/lib/api";

interface Message {
  role: "user" | "assistant";
  content: string;
  imagePreview?: string;
  products?: string[];   // item_ids parsed from reply
  toolCalls?: { tool: string }[];
}

const ITEM_ID_RE = /\b(B[A-Z0-9]{9})\b/g;

function parseItemIds(text: string): string[] {
  return [...new Set([...text.matchAll(ITEM_ID_RE)].map((m) => m[1]))];
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function formatReply(text: string): string {
  return escapeHtml(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, '<code class="bg-gray-100 px-1 rounded text-xs font-mono">$1</code>')
    .replace(/\n\n/g, "</p><p class='mt-2'>")
    .replace(/\n/g, "<br/>");
}

function ProductCard({ itemId, onAsk }: { itemId: string; onAsk: (id: string) => void }) {
  const imgUrl = productImageUrl(itemId);
  return (
    <div className="bg-white border border-gray-200 rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-shadow">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={imgUrl}
        alt="product"
        className="w-full h-28 object-contain bg-gray-50 p-2"
        onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
      />
      <div className="px-3 py-2">
        <p className="text-xs text-gray-400 font-mono truncate">{itemId}</p>
        <button
          onClick={() => onAsk(itemId)}
          className="mt-1 text-xs text-indigo-600 hover:underline"
        >
          Tell me more →
        </button>
      </div>
    </div>
  );
}

export default function ChatWindow() {
  const [messages, setMessages]     = useState<Message[]>([]);
  const [input, setInput]           = useState("");
  const [loading, setLoading]       = useState(false);
  const [sessionId, setSessionId]   = useState<string | undefined>();
  const [pendingImg, setPendingImg] = useState<string | null>(null); // base64
  const fileRef    = useRef<HTMLInputElement>(null);
  const bottomRef  = useRef<HTMLDivElement>(null);
  const textRef    = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      const data = (ev.target?.result as string).split(",")[1];
      setPendingImg(data);
    };
    reader.readAsDataURL(file);
  };

  const send = useCallback(async (text?: string) => {
    const msg = (text ?? input).trim();
    if (!msg && !pendingImg) return;

    const userMsg: Message = {
      role: "user",
      content: msg || "🖼 (image search)",
      imagePreview: pendingImg ? `data:image/jpeg;base64,${pendingImg}` : undefined,
    };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setPendingImg(null);
    if (fileRef.current) fileRef.current.value = "";
    setLoading(true);

    try {
      const res = await sendChat({
        message:    msg || "What product is in this image? Search for similar ones.",
        image_b64:  pendingImg ?? undefined,
        session_id: sessionId,
      });
      if (!sessionId) setSessionId(res.session_id);

      const botMsg: Message = {
        role:      "assistant",
        content:   res.reply,
        products:  parseItemIds(res.reply),
        toolCalls: res.tool_calls,
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${(err as Error).message}` },
      ]);
    } finally {
      setLoading(false);
    }
  }, [input, pendingImg, sessionId]);

  const askAbout = (itemId: string) => send(`Tell me more about product ${itemId}`);

  return (
    <div className="flex flex-col h-full max-w-3xl mx-auto w-full">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-4 py-3 flex items-center gap-3 shadow-sm">
        <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white text-sm font-bold">S</div>
        <div>
          <h1 className="font-semibold text-sm">Shopping Assistant</h1>
          <p className="text-xs text-gray-400">bge-visualized-m3 · Elasticsearch · LLM</p>
        </div>
        {sessionId && (
          <span className="ml-auto text-xs text-gray-300 font-mono hidden sm:block truncate max-w-xs">
            session: {sessionId.slice(0, 8)}…
          </span>
        )}
      </header>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex gap-3">
            <div className="w-7 h-7 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-600 text-xs font-bold flex-shrink-0 mt-1">S</div>
            <div className="bubble-bot bg-white border border-gray-200 px-4 py-3 text-sm text-gray-700 shadow-sm max-w-lg">
              Hi! I can find products by description or image.{" "}
              {["Show me a comfortable chair", "Red leather sofa", "Upload an image"].map((s) => (
                <button key={s} onClick={() => send(s)} className="text-indigo-600 hover:underline mx-1">
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`flex gap-3 ${m.role === "user" ? "justify-end" : ""}`}>
            {m.role === "assistant" && (
              <div className="w-7 h-7 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-600 text-xs font-bold flex-shrink-0 mt-1">S</div>
            )}
            <div className="max-w-lg">
              {m.role === "user" ? (
                <div className="bubble-user bg-indigo-600 text-white px-4 py-2 text-sm shadow-sm">
                  {m.imagePreview && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={m.imagePreview} alt="upload" className="max-h-28 rounded-lg mb-2 block" />
                  )}
                  {m.content}
                </div>
              ) : (
                <>
                  <div className="bubble-bot bg-white border border-gray-200 px-4 py-3 text-sm text-gray-700 shadow-sm">
                    <p dangerouslySetInnerHTML={{ __html: "<p>" + formatReply(m.content) + "</p>" }} />
                    {m.toolCalls && m.toolCalls.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1">
                        {m.toolCalls.map((tc, j) => (
                          <span key={j} className="bg-gray-100 text-gray-500 px-2 py-0.5 rounded text-xs">{tc.tool}</span>
                        ))}
                      </div>
                    )}
                  </div>
                  {m.products && m.products.length > 0 && (
                    <div className="mt-2 grid grid-cols-2 sm:grid-cols-3 gap-2">
                      {m.products.map((id) => (
                        <ProductCard key={id} itemId={id} onAsk={askAbout} />
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="flex gap-3">
            <div className="w-7 h-7 rounded-full bg-indigo-100 flex items-center justify-center text-indigo-600 text-xs font-bold flex-shrink-0 mt-1">S</div>
            <div className="bubble-bot bg-white border border-gray-200 px-4 py-3 shadow-sm flex gap-1 items-center">
              <span className="typing-dot w-2 h-2 bg-gray-400 rounded-full" />
              <span className="typing-dot w-2 h-2 bg-gray-400 rounded-full" />
              <span className="typing-dot w-2 h-2 bg-gray-400 rounded-full" />
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Image preview */}
      {pendingImg && (
        <div className="bg-white border-t border-gray-100 px-4 py-2 flex items-center gap-3">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={`data:image/jpeg;base64,${pendingImg}`} alt="preview" className="h-14 w-14 object-cover rounded-lg border" />
          <span className="text-xs text-gray-500 flex-1">Image ready</span>
          <button onClick={() => { setPendingImg(null); if (fileRef.current) fileRef.current.value = ""; }} className="text-xs text-red-400 hover:text-red-600">✕</button>
        </div>
      )}

      {/* Input bar */}
      <div className="bg-white border-t border-gray-200 px-4 py-3">
        <div className="flex gap-2 items-end">
          <label className="cursor-pointer w-9 h-9 rounded-lg border border-gray-300 flex items-center justify-center hover:bg-gray-50 text-gray-400 flex-shrink-0">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            <input ref={fileRef} type="file" accept="image/*" className="hidden" onChange={handleFile} />
          </label>

          <textarea
            ref={textRef}
            rows={1}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
            placeholder="Ask about products or upload an image…"
            className="flex-1 resize-none border border-gray-300 rounded-xl px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
            style={{ maxHeight: 120 }}
          />

          <button
            onClick={() => send()}
            disabled={loading || (!input.trim() && !pendingImg)}
            className="w-9 h-9 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white flex items-center justify-center disabled:opacity-40 disabled:cursor-not-allowed flex-shrink-0"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
