# Frontend (Static SPA) — Historical Photos

Simple, deployment-friendly UI that talks to your existing API (Railway).
Works locally and on Vercel as a static site. No build step required.

## Features
- Browse first N photos (`/api/photos`)
- Semantic search (`/api/search/semantic`)
- REAL/PLACEHOLDER badge on thumbnails (detects proxy placeholder)
- Open full image on click
- Runtime Base URL switcher (no rebuild needed)

## Run Locally
- Option A: Python static server
  - `cd frontend`
  - `python -m http.server 5500`
  - Open `http://localhost:5500`
- Option B: Any static server (e.g., `npx serve frontend`)

Set API base at runtime (top-right input). Defaults to the value saved in `localStorage` or to `https://testphotos-production.up.railway.app`.

## Deploy on Vercel (static)
- New Project → Import this repo → Framework: Other
- Build/Output settings:
  - Build Command: (leave empty)
  - Output Directory: `frontend`
- Environment variables are optional; Base URL is set at runtime from the UI.

## Files
- `index.html` — single-page UI
- `styles.css` — minimal styles
- `js/app.js` — logic to call the API and render results

## Tips
- PLACEHOLDER means the server didn’t find a real image. Ensure `IMAGE_CDN_BASE` and `IMAGE_PATH_MODE=fullpath` are set on your Railway app, and your R2 bucket is public.
- The API already has CORS enabled; no special proxy is needed.

