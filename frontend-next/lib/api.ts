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

export async function checkReady(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/ready`);
    if (!res.ok) return false;
    const data = await res.json();
    return data.status === "ready";
  } catch {
    return false;
  }
}

export interface StreamCallbacks {
  onStatus: (text: string) => void;
  onToken:  (text: string) => void;
  onDone:   (response: ChatResponse) => void;
  onError:  (detail: string) => void;
}

/**
 * Stream a chat request via SSE. Returns an AbortController to cancel.
 * Uses fetch + ReadableStream (not EventSource, which only supports GET).
 */
export function streamChat(req: ChatRequest, callbacks: StreamCallbacks): AbortController {
  const controller = new AbortController();

  (async () => {
    let res: Response;
    try {
      res = await fetch(`${BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req),
        signal: controller.signal,
      });
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        callbacks.onError((e as Error).message);
      }
      return;
    }

    if (!res.ok || !res.body) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      callbacks.onError(err.detail ?? "Stream failed");
      return;
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = "";

    while (true) {
      let done: boolean, value: Uint8Array | undefined;
      try {
        ({ done, value } = await reader.read());
      } catch {
        break;
      }
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE: events are separated by "\n\n"
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";   // keep incomplete tail

      for (const block of events) {
        if (!block.trim()) continue;
        let eventName = "message";
        let dataLine  = "";

        for (const line of block.split("\n")) {
          if (line.startsWith("event: ")) {
            eventName = line.slice(7).trim();
          } else if (line.startsWith("data: ")) {
            dataLine = line.slice(6).trim();
          }
        }

        if (!dataLine) continue;
        let payload: Record<string, unknown>;
        try {
          payload = JSON.parse(dataLine);
        } catch {
          continue;
        }

        if (eventName === "status") {
          callbacks.onStatus(payload.text as string);
        } else if (eventName === "token") {
          callbacks.onToken(payload.text as string);
        } else if (eventName === "done") {
          callbacks.onDone(payload as unknown as ChatResponse);
        } else if (eventName === "error") {
          callbacks.onError(payload.detail as string);
        }
      }
    }
  })();

  return controller;
}

export async function getProduct(itemId: string) {
  const res = await fetch(`${BASE}/products/${itemId}`);
  if (!res.ok) return null;
  return res.json();
}

export function productImageUrl(itemId: string) {
  return `${BASE}/image/${itemId}`;
}
