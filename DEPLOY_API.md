# Deploying the Ground Control API (cloud-native, public)

The Streamlit dashboard runs on **Streamlit Community Cloud**, which only exposes
its own web port — so the FastAPI endpoints (`/update_herd`, `/next_waypoints`)
are **not** publicly reachable there. To let a real Raspberry Pi (or any client)
reach them from anywhere, deploy `api.py` as a standalone service.

## Architecture (100% cloud-native)

```
 Raspberry Pi ──POST /update_herd──▶  FastAPI on Render  ◀──GET /state──┐
 (drone GPS,                         (api.py, public HTTPS)             │
  livestock)   ◀─GET /next_waypoints─┘                                 │
                                                              Streamlit dashboard
                                                              (reads live feed)
```

## A. Deploy the API on Render (free tier)

1. Push this repo to GitHub.
2. Render Dashboard → **New → Blueprint** → select this repo → **Apply**.
   Render reads `render.yaml`, installs `requirements-api.txt`, and runs
   `uvicorn api:app --host 0.0.0.0 --port $PORT`.
3. After build, you get a public URL, e.g.
   `https://drone-ground-control-api.onrender.com`.
4. Verify:
   ```bash
   python test_live_api.py https://drone-ground-control-api.onrender.com
   ```
   Expect: `✅ POST -> GET loop FULLY FUNCTIONAL`.

> **Free-tier cold start:** the service sleeps after ~15 min idle and takes
> ~30–60 s to wake. Hit `/health` once a few minutes before the demo to warm it.

Other platforms work identically via the included `Procfile`
(`web: uvicorn api:app --host 0.0.0.0 --port $PORT`) — e.g. Railway, Fly.io,
or a Hugging Face Space (Docker).

## B. Point the dashboard at the hosted API

**Option 1 — sidebar (per session):** In the deployed dashboard, switch to
**REAL DRONE MODE**, set **API source → Hosted (public URL)**, and paste your
Render URL into **Public API base URL**.

**Option 2 — default it via a secret (recommended for the demo):** In the
Streamlit Cloud app settings → **Secrets**, add:
```toml
ground_control_api_url = "https://drone-ground-control-api.onrender.com"
```
The dashboard then defaults to **Hosted** with this URL pre-filled. (Locally you
can instead set the env var `GROUND_CONTROL_API_URL`.)

## C. Real Raspberry Pi client (field deployment)

```python
import requests
API = "https://drone-ground-control-api.onrender.com"
packet = {
    "drone_lat": 46.7991, "drone_lon": 9.8498, "drone_alt": 25.0,
    "livestock": [[46.7995, 9.8495], [46.8000, 9.8505]],
    "threats": [],            # filled by the onboard wolf detector
    "n_waypoints": 5,
}
requests.post(f"{API}/update_herd", json=packet, timeout=10)
plan = requests.get(f"{API}/next_waypoints", timeout=10).json()
for wp in plan["waypoints"]:
    fly_to(wp["lat"], wp["lon"], wp["altitude_m"], wp["speed_ms"])
```
