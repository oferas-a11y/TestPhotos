# Frontend Photo UI — Troubleshooting Guide

Use this checklist when thumbnails show as placeholders, images don’t render, or search returns empty results. Follow the order — each step narrows the root cause quickly.

## Quick Diagnostics
- **API Base URL:** Confirm the frontend uses the right `API_BASE_URL`/`VITE_API_BASE_URL`/`NEXT_PUBLIC_API_BASE_URL`.
- **Health:** `GET <base>/health` should return `status: healthy`.
- **Status:** `GET <base>/api/status` should show Pinecone connected and photo counts.
- **List Photos:** `GET <base>/api/photos?limit=15` should return items with `thumbnail_url` and `image_url`.
- **Semantic Search:** `GET <base>/api/search/semantic?q=people&limit=5&gemini=false` should return results.

## Image Checks (Real vs Placeholder)
- **Thumbnail probe:** Open one item’s `thumbnail_url` in browser (or curl with `-I`).
  - Real image: `Content-Type: image/jpeg` (or larger PNG), size typically > 10KB.
  - Placeholder: `Content-Type: image/png` and very small (~1–4KB). This means the server couldn’t find the real file.
- **Full image:** Repeat with `image_url` — should be a real JPEG if available.

## Most Common Root Causes
- **Missing Files in Container:** Images aren’t bundled in Railway; API falls back to placeholder.
- **CDN Not Configured:** `IMAGE_CDN_BASE` not set, so the API can’t fetch from your object storage.
- **Bucket Not Public:** R2/S3 returns 403/400; the proxy can’t read files.
- **Path Mode Mismatch:** R2 keys use full paths, but `IMAGE_PATH_MODE` is `basename` (or vice‑versa).
- **Filename Mismatch:** `metadata.original_path` differs from uploaded key (spaces/encoding/case).
- **Wrong Base URL:** Frontend points to the wrong environment.

## Fix — Production (Railway + R2)
- **Set Vars (Railway → Variables):**
  - `IMAGE_CDN_BASE=https://<ACCOUNT_ID>.r2.cloudflarestorage.com/<BUCKET>`
  - `IMAGE_PATH_MODE=fullpath` (if you uploaded using the original paths)
- **Ensure Public Access:** Make the R2 bucket public (or serve via an R2 public domain).
- **Upload Photos:** From repo root:
  - `pip install boto3 python-dotenv`
  - Add R2 creds to `.env` (see `frontend/.env.example`).
  - `python scripts/setup_cloud_and_verify.py` (uploads/resumes and verifies a sample file).
- **Redeploy:** Deploy branch `feature/remote-image-fetch` on Railway after setting vars.

## Verify End‑to‑End
- **Local demo page:**
  - `python app.py` → open `http://localhost:<port>/demo`
  - Browse/Search and look for REAL badges.
- **Remote debug page:**
  - `http://localhost:<port>/debug/remote?base=<https://your-app>&q=love&limit=12`
  - Tiles will show REAL/PLACEHOLDER and link to full images.
- **CLI test:**
  - `BASE_URL=<https://your-app> python scripts/test_server_photo.py`
  - PASS = server returns a non‑placeholder thumbnail.

## Frontend Env (UI Builds)
- **Vite:** Set `VITE_API_BASE_URL` (and optionally `VITE_IMAGE_CDN_BASE`).
- **Next.js:** Set `NEXT_PUBLIC_API_BASE_URL` (and optionally `NEXT_PUBLIC_IMAGE_CDN_BASE`).
- **Sample file:** `frontend/.env.example` contains ready‑to‑copy values.

## Debugging Tips
- **Inspect Metadata:**
  - `GET <base>/api/search/semantic?...` → check each item’s `metadata.original_path`.
  - Directly open: `<IMAGE_CDN_BASE>/<original_path>` — if 404/403, fix storage or permissions.
- **Path Mode:**
  - If your bucket stores only basenames, set `IMAGE_PATH_MODE=basename`.
  - If it mirrors project paths (recommended), keep `fullpath`.
- **CORS:** Responses must include `Access-Control-Allow-Origin: *` (enabled in the API). If not, check proxies/load balancers.
- **Port Conflicts (Local):** The server auto‑picks a free port if 5000 is busy; or run `PORT=5050 python app.py`.
- **Gemini Reranking:** If results are slow or you lack a key, disable with `gemini=false`.

## When Search Returns Empty
- **Pinecone Data:** Make sure vectors were migrated. `/api/status` should show non‑zero `total_photos`.
- **Migration/Seeding:** Use `migrate_to_pinecone.py` or confirm via `run.py dashboard → Pinecone Stats`.
- **Query Text:** Try a generic query (`people`, `street`, `outdoor`) to validate pipeline.

## Golden Paths (Copy/Paste)
- **Check photo list:**
  - `curl -s "<base>/api/photos?limit=15" | jq`  
- **Search + fetch thumb headers:**
  - `PID=$(curl -s "<base>/api/search/semantic?q=people&limit=1&gemini=false" | jq -r '.results[0].id')`  
  - `curl -s -D- "<base>/api/photo/$PID/thumbnail" -o /dev/null`
- **Direct CDN probe:**
  - `curl -s -I "<IMAGE_CDN_BASE>/<original_path>"`

## Summary
- Pinecone stores vectors + metadata, not image bytes.
- The API will always return a valid image response: real JPEG/PNG if found, otherwise a safe placeholder PNG.
- Configure `IMAGE_CDN_BASE` + `IMAGE_PATH_MODE` and make your bucket public to serve real images in production.

