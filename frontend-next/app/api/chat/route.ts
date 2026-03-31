export const maxDuration = 120; // seconds — search + reranking can take ~45s

export async function POST(req: Request) {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const body = await req.text();

  const upstream = await fetch(`${apiUrl}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body,
    // @ts-ignore — Node 18 fetch signal for explicit timeout
    signal: AbortSignal.timeout(115_000),
  });

  return new Response(upstream.body, {
    status: upstream.status,
    headers: { "Content-Type": "application/json" },
  });
}
