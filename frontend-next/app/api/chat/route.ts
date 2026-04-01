export const maxDuration = 120; // seconds — search + reranking can take ~45s

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function POST(req: Request) {
  const { pathname } = new URL(req.url);
  const isStream = pathname.endsWith("/stream");

  const body = await req.text();

  const upstream = await fetch(`${API_URL}/${isStream ? "chat/stream" : "chat"}`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body,
    // @ts-ignore — Node 18 fetch AbortSignal
    signal: isStream ? undefined : AbortSignal.timeout(115_000),
  });

  if (isStream) {
    // Pass the SSE stream straight through — no buffering, no timeout
    return new Response(upstream.body, {
      status:  upstream.status,
      headers: {
        "Content-Type":  "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection":    "keep-alive",
      },
    });
  }

  return new Response(upstream.body, {
    status:  upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
