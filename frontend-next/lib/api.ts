const BASE = process.env.NEXT_PUBLIC_API_URL
  ? process.env.NEXT_PUBLIC_API_URL
  : "/api";   // proxied via next.config.ts rewrites in dev

export interface ChatRequest {
  message:    string;
  image_b64?: string;
  session_id?: string;
  history?:   { role: "user" | "assistant"; content: string }[];
}

export interface ChatResponse {
  reply:      string;
  session_id: string;
  tool_calls: { tool: string; args: Record<string, unknown> }[];
}

export async function sendChat(req: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Unknown error");
  }
  return res.json();
}

export async function getProduct(itemId: string) {
  const res = await fetch(`${BASE}/products/${itemId}`);
  if (!res.ok) return null;
  return res.json();
}

export function productImageUrl(itemId: string) {
  return `${BASE}/image/${itemId}`;
}
