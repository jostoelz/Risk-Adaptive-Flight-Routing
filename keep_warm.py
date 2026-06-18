"""
keep_warm.py — keep the Render-hosted Ground Control API awake during the demo.

Render's free tier spins a service down after ~15 minutes of inactivity and then
needs ~30–60 s to cold-start. This pinger hits /health on a tight interval so the
API stays hot and instantly responsive throughout the event.

This is the most resilient option for the LIVE window: it fires on a precise
interval with no scheduler lag (unlike cron), retries transient errors, and never
crashes. Run it on the presenter's laptop a few minutes before the demo and leave
it running. (A GitHub Action under .github/workflows/keep-warm.yml provides a
laptop-independent 24/7 backup.)

Usage:
    python keep_warm.py https://drone-ground-control-api.onrender.com
    python keep_warm.py                      # uses $GROUND_CONTROL_API_URL
    python keep_warm.py <url> --interval 120 # ping every 120 s (default 240)

Stop with Ctrl+C.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import requests

# Force UTF-8 stdout so status glyphs render on any console (e.g. Windows cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_INTERVAL_S = 240          # < Render's 15-min idle window, with margin
COLD_START_TIMEOUT_S = 90         # first wake from sleep can be slow
WARM_TIMEOUT_S = 20


def _now() -> str:
    return time.strftime("%H:%M:%S")


def resolve_url(cli_url: str | None) -> str:
    url = (cli_url
           or os.environ.get("GROUND_CONTROL_API_URL")
           or "https://drone-ground-control-api.onrender.com")
    return url.rstrip("/")


def ping_once(base: str, timeout: float) -> tuple[bool, str]:
    """Ping /health once. Returns (ok, detail)."""
    try:
        t0 = time.time()
        resp = requests.get(f"{base}/health", timeout=timeout)
        dt = (time.time() - t0) * 1000.0
        if resp.status_code == 200:
            return True, f"200 OK in {dt:.0f} ms"
        return False, f"HTTP {resp.status_code} in {dt:.0f} ms"
    except requests.RequestException as exc:
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Keep the Render API warm.")
    parser.add_argument("url", nargs="?", default=None,
                        help="API base URL (else $GROUND_CONTROL_API_URL).")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S,
                        help=f"seconds between pings (default {DEFAULT_INTERVAL_S})")
    parser.add_argument("--once", action="store_true",
                        help="ping a single time and exit (e.g. for warming up).")
    args = parser.parse_args()

    base = resolve_url(args.url)
    print(f"[{_now()}] keep_warm → {base}/health  "
          f"(interval {args.interval:.0f}s){'  [single ping]' if args.once else ''}")
    print(f"[{_now()}] warming up (first wake may take up to "
          f"{COLD_START_TIMEOUT_S}s)…", flush=True)

    pings = 0
    fails = 0
    streak = 0
    try:
        while True:
            timeout = COLD_START_TIMEOUT_S if pings == 0 else WARM_TIMEOUT_S
            ok, detail = ping_once(base, timeout)
            pings += 1
            if ok:
                streak += 1
                print(f"[{_now()}] ✅ awake — {detail}  "
                      f"(ping #{pings}, {streak} in a row)", flush=True)
            else:
                fails += 1
                streak = 0
                print(f"[{_now()}] ⚠️  ping failed — {detail}  "
                      f"(failure {fails}/{pings})", flush=True)

            if args.once:
                return 0 if ok else 1

            time.sleep(max(5.0, args.interval))
    except KeyboardInterrupt:
        print(f"\n[{_now()}] stopped — {pings} pings, {fails} failures. "
              f"API left in its current state.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
