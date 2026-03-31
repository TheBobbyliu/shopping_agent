"""Run all 4 E2E conversation scenarios and print rich traces."""
import re
import requests
import time

API = "http://localhost:8000"


def chat(message, session_id=None):
    t0 = time.perf_counter()
    r = requests.post(
        f"{API}/chat",
        json={"message": message, "session_id": session_id},
        timeout=120,
    )
    elapsed = time.perf_counter() - t0
    d = r.json()
    d["_latency"] = round(elapsed, 2)
    return d


def turn(n, msg, resp):
    tools = [tc["tool"] for tc in resp.get("tool_calls", [])]
    print(f"  Turn {n}  ({resp['_latency']}s)")
    print(f"  User : {msg}")
    print(f"  Tools: {tools if tools else '(none)'}")
    print(f"  Agent: {resp['reply'][:280].strip()}")
    print()


# ---------------------------------------------------------------------------
print("=" * 68)
print("SCENARIO 1: Text search -> refinement -> product detail")
print("=" * 68)
r1 = chat("Show me some chairs")
sid = r1["session_id"]
print(f"  session: {sid[:8]}...\n")
turn(1, "Show me some chairs", r1)

r2 = chat("Actually I want something leather and black", session_id=sid)
turn(2, "Actually I want something leather and black", r2)

r3 = chat("Tell me more about the first one you mentioned", session_id=sid)
turn(3, "Tell me more about the first one you mentioned", r3)

# ---------------------------------------------------------------------------
print("=" * 68)
print("SCENARIO 2: Category filtering + attribute refinement")
print("=" * 68)
r1 = chat("Show me sofas and couches")
sid = r1["session_id"]
print(f"  session: {sid[:8]}...\n")
turn(1, "Show me sofas and couches", r1)

r2 = chat("I want wooden material ones", session_id=sid)
turn(2, "I want wooden material ones", r2)

r3 = chat("Which would work well in a modern minimalist home?", session_id=sid)
turn(3, "Which would work well in a modern minimalist home?", r3)

# ---------------------------------------------------------------------------
print("=" * 68)
print("SCENARIO 3: Session isolation (two independent sessions)")
print("=" * 68)
a1 = chat("I'm looking for running shoes")
b1 = chat("I need wall art for my bedroom")
sid_a, sid_b = a1["session_id"], b1["session_id"]
print(f"  session A: {sid_a[:8]}...  [shoes]")
print(f"  session B: {sid_b[:8]}...  [art]\n")
turn(1, "[A] I'm looking for running shoes", a1)
turn(1, "[B] I need wall art for my bedroom", b1)

a2 = chat("Men's size, preferably blue", session_id=sid_a)
b2 = chat("Something abstract and colorful", session_id=sid_b)
turn(2, "[A] Men's size, preferably blue", a2)
turn(2, "[B] Something abstract and colorful", b2)

# Verify no bleed
a2_lower = a2["reply"].lower()
b2_lower = b2["reply"].lower()
shoe_ok = any(w in a2_lower for w in ["shoe", "running", "sneaker", "blue", "men", "size", "sport"])
art_clean = "running shoe" not in b2_lower and "sneaker" not in b2_lower
print(f"  [check] Session A reply is about shoes: {shoe_ok}")
print(f"  [check] Session B reply has no shoe bleed: {art_clean}\n")

# ---------------------------------------------------------------------------
print("=" * 68)
print("SCENARIO 4: Conversation memory (recall item_id from prior turn)")
print("=" * 68)
r1 = chat("Show me a dining table")
sid = r1["session_id"]
print(f"  session: {sid[:8]}...\n")
turn(1, "Show me a dining table", r1)

ids = re.findall(r'\b(B[A-Z0-9]{9})\b', r1["reply"])
target = ids[0] if ids else None
if target:
    print(f"  item_id captured: {target}\n")
    r2 = chat(f"Tell me more about {target}", session_id=sid)
    turn(2, f"Tell me more about {target}", r2)
    r3 = chat("What material is it made of?", session_id=sid)
    turn(3, "What material is it made of?", r3)
    tools_r2 = [tc["tool"] for tc in r2.get("tool_calls", [])]
    print(f"  [check] get_product_info called on turn 2: {'get_product_info' in tools_r2}")
else:
    print("  [warn] No item_id found in turn 1 reply — skipping turns 2-3")

print()
print("=" * 68)
print("All scenarios completed.")
print("=" * 68)
