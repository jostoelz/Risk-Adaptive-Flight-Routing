"""
test_live_api.py — Live Ground Control API smoke test.

Simulates a Raspberry Pi pushing drone GPS + livestock telemetry to the deployed
server, then pulls the next 3-5 RHC waypoints, and reports whether the full
POST -> GET communication loop is functional and stable.

Usage:
    python test_live_api.py                       # tests the deployed URL
    python test_live_api.py http://localhost:8000 # tests a local API
"""

import io
import sys
import time

import requests

# Force UTF-8 stdout so status glyphs render on any console (e.g. Windows cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DEFAULT_BASE = "https://risk-adaptive-flight-routing.streamlit.app"
BASE = (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE).rstrip("/")

# A realistic packet: drone south of the pasture with a detected livestock cluster.
TELEMETRY = {
    "drone_lat": 46.79910,
    "drone_lon": 9.84980,
    "drone_alt": 25.0,
    "livestock": [
        [46.79950, 9.84950], [46.79952, 9.84958], [46.79948, 9.84962],
        [46.80000, 9.85050], [46.80005, 9.85060], [46.79970, 9.84990],
    ],
    "n_waypoints": 5,
}

SEP = "=" * 70


def _classify(resp: requests.Response):
    """Return ('json', obj) or ('html'/'text', preview-string)."""
    ctype = resp.headers.get("Content-Type", "")
    if "application/json" in ctype:
        try:
            return "json", resp.json()
        except ValueError:
            pass
    body = resp.text or ""
    looks_html = ("<html" in body[:600].lower()) or ("text/html" in ctype)
    return ("html" if looks_html else "text"), body[:200].replace("\n", " ").strip()


def probe(method: str, path: str, **kw):
    url = f"{BASE}{path}"
    print(f"\n--- {method} {url}")
    try:
        resp = requests.request(method, url, timeout=20, **kw)
    except requests.RequestException as exc:
        print(f"    ✗ request failed: {type(exc).__name__}: {exc}")
        return None, None
    kind, payload = _classify(resp)
    print(f"    status={resp.status_code}  "
          f"content-type={resp.headers.get('Content-Type', '?')}  kind={kind}")
    if kind == "json":
        print(f"    JSON: {payload}")
    else:
        print(f"    {kind.upper()} preview: {payload!r}")
    return resp, (payload if kind == "json" else None)


def main():
    print(SEP)
    print(f"LIVE GROUND CONTROL API TEST  —  base: {BASE}")
    print(SEP)

    results = {}

    # 1) Liveness / banner.
    r, _ = probe("GET", "/health")
    results["health"] = bool(r is not None and r.status_code == 200)
    probe("GET", "/")

    # 2) POST telemetry (simulate the Raspberry Pi).
    print("\n" + SEP)
    print("STEP 1 — POST /update_herd  (Raspberry Pi pushes telemetry)")
    print(SEP)
    r, post_json = probe("POST", "/update_herd", json=TELEMETRY)
    post_ok = bool(post_json and "waypoints" in post_json)
    results["post"] = post_ok

    # 3) GET the next waypoints (simulate the flight controller).
    print("\n" + SEP)
    print("STEP 2 — GET /next_waypoints  (flight controller pulls RHC plan)")
    print(SEP)
    time.sleep(0.5)
    r, get_json = probe("GET", "/next_waypoints")
    get_ok = bool(get_json and isinstance(get_json.get("waypoints"), list))
    results["get"] = get_ok

    if get_ok:
        wps = get_json["waypoints"]
        print(f"\n    Received {len(wps)} waypoints [Lat, Lon, Alt, Speed]:")
        for i, w in enumerate(wps, 1):
            print(f"      {i}. lat={w.get('lat'):.5f} lon={w.get('lon'):.5f} "
                  f"alt={w.get('altitude_m'):.1f}m v={w.get('speed_ms'):.1f}m/s "
                  f"risk={w.get('risk'):.3f}")
        results["count_ok"] = 3 <= len(wps) <= 5

    # Verdict.
    print("\n" + SEP)
    print("VERDICT")
    print(SEP)
    loop_ok = results.get("post") and results.get("get") and results.get("count_ok")
    for k in ("health", "post", "get", "count_ok"):
        if k in results:
            print(f"  {'✓' if results[k] else '✗'}  {k}")
    if loop_ok:
        print("\n  ✅ POST -> GET loop FULLY FUNCTIONAL — ready for the demo.")
    else:
        print("\n  ❌ API loop NOT reachable at this base URL "
              "(responses are not the FastAPI JSON).")
    return 0 if loop_ok else 1


if __name__ == "__main__":
    sys.exit(main())
