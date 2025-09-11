# Frontend Integration Guide — Historical Photos API

This document explains how to quickly fetch photos for the UI (first 1–15), perform semantic search, and display real images via proxy endpoints. It also outlines how to prepare CLIP-based search.

## Overview
- Base URL: your deployment (e.g., `https://testphotos-production.up.railway.app`)
- CORS: enabled (`Access-Control-Allow-Origin: *`)
- Images are served via API proxy endpoints:
  - `image_url`: full-size image
  - `thumbnail_url`: smaller preview
- If a real image file is missing in the container (e.g., Git LFS pointers), the API returns a placeholder PNG so the UI never breaks.

## Quickstart
- Call `GET /api/photos?limit=15` to get the first 15 photos (real images prioritized).
- Call `GET /api/search/semantic?q=<text>&limit=15` for semantic search.
- Use each item’s `thumbnail_url` and `image_url` to render real images via the API.

## Health & Status
- `GET /health` → `{"status":"healthy","service":"historical-photos-api","version":"1.0.0"}`
- `GET /api/status` → connection info (Pinecone, counts, services)

## Browse (First N Photos)
Endpoint: `GET /api/photos`

Query params:
- `limit` (optional, 1–50): number of results (default 15)
- `offset` (optional, >= 0): skip N items (default 0)

Behavior:
- Returns a list of photos, prioritizing entries with real image files available on disk.
- Remaining results (without local files) are included after, whose proxies will return placeholders.

Example:
```
GET /api/photos?limit=15
```

Response (shape):
```
{
  "results": [
    {
      "id": "photo_12345",
      "score": 0.73,
      "metadata": { "original_path": "photo_collections/.../file.jpg", ... },
      "image_url": "https://<base>/api/photo/photo_12345/image",
      "thumbnail_url": "https://<base>/api/photo/photo_12345/thumbnail"
    },
    ...
  ],
  "total": 858,
  "offset": 0,
  "limit": 15
}
```

## Search (Primary Flow)
Endpoint: `GET /api/search/semantic`

Query params:
- `q` (required): search text
- `limit` (optional, 1–50): number of results (default 15)
- `gemini` (optional, `true`/`false`): AI reranking (default `true` if key present; set `false` for faster responses)

Example:
```
GET /api/search/semantic?q=love%20is%20in%20the%20air&limit=10&gemini=false
```

Response (shape):
```
{
  "query": "...",
  "total": <number>,
  "reranked": <bool>,
  "gemini_enabled": <bool>,
  "results": [
    {
      "id": "photo_12345",
      "score": 0.78,
      "description": "optional textual description",
      "metadata": { "original_path": "photo_collections/.../file.jpg", ... },
      "image_url": "https://<base>/api/photo/photo_12345/image",
      "thumbnail_url": "https://<base>/api/photo/photo_12345/thumbnail"
    },
    ...
  ]
}
```

How to render:
1) Use `thumbnail_url` for grid/list previews.
2) On click, open `image_url` for the full image.
3) Both endpoints may return a placeholder PNG if the original is unavailable—always rely on `Content-Type`.

## Practical Examples

- cURL — first 15 photos:
```
curl -s "https://<base>/api/photos?limit=15" | jq
```

- cURL — semantic search (15 results):
```
curl -s "https://<base>/api/search/semantic?q=people%20smiling&limit=15&gemini=false" | jq
```

- JavaScript — initial load + render thumbnails:
```
const BASE = "https://<base>";

async function loadInitialPhotos() {
  const res = await fetch(`${BASE}/api/photos?limit=15`);
  const data = await res.json();
  return data.results; // array of items with id, thumbnail_url, image_url
}

async function searchSemantic(query) {
  const params = new URLSearchParams({ q: query, limit: "15", gemini: "false" });
  const res = await fetch(`${BASE}/api/search/semantic?${params}`);
  const data = await res.json();
  return data.results;
}

function renderThumb(item) {
  const img = document.createElement('img');
  img.src = item.thumbnail_url;
  img.alt = item.metadata?.original_path || item.id;
  img.loading = 'lazy';
  return img;
}
```

## Photo Details
- `GET /api/photo/{photo_id}` → returns metadata and the same `image_url`/`thumbnail_url` fields for the specific photo.

Response (example):
```
{
  "id": "photo_12345",
  "description": "...",
  "metadata": { "original_path": "photo_collections/.../file.jpg", ... },
  "image_url": "https://<base>/api/photo/photo_12345/image",
  "thumbnail_url": "https://<base>/api/photo/photo_12345/thumbnail"
}
```

## Image Proxy Endpoints
- `GET /api/photo/{photo_id}/thumbnail` → preview image
- `GET /api/photo/{photo_id}/image` → full-size image

Behavior:
- Returns `200` with a real image if present in the container.
- If the underlying file is a Git LFS pointer or missing, the server tries configured remote sources first, then falls back to a placeholder PNG.
- Sets `Content-Disposition: inline; filename="..."` to enable direct display.

Frontend tips:
- Always check `Content-Type` (may be `image/jpeg`, `image/png`, etc.).
- Thumbnails are small (up to ~300×300px). Full images can be larger.
- Placeholders are safe to display and avoid broken tiles.

Deployment note (serving real photos):
- Set `IMAGE_CDN_BASE` (and optional `THUMB_CDN_BASE`) to a public image base URL. The server will try:
  - `${BASE}/{basename}` and, if `IMAGE_PATH_MODE=fullpath`, `${BASE}/{original_path}`.
- Or set `IMAGE_URL_TEMPLATE` / `THUMB_URL_TEMPLATE` with `{basename}` and `{path}` placeholders.
- If none resolve and the file isn’t on disk, proxies return a placeholder PNG.

## Using CLIP Vectors (Overview)
- Goal: text-to-image and image-to-image search using OpenAI CLIP (512-D vectors).
- Data prep: run `python generate_clip_embeddings.py` to create embeddings under `main_app_outputs/embeddings/`.
- Storage: upsert those vectors into Pinecone (recommended in a separate index such as `historical-photos-clip`).
- Querying: encode the query with the CLIP text encoder and query the CLIP index; return IDs/paths and use the same `image_url`/`thumbnail_url` proxies to render.

Notes:
- The current `/api/search/semantic` uses a Sentence-Transformers model (384-D) for fast semantic text search.
- If you want a dedicated `/api/search/clip` endpoint, it can be added to query the CLIP index. Ask and we’ll wire it up.

Minimal CLIP migration sketch (pseudo-code):
```
# 1) Generate CLIP embeddings locally
python generate_clip_embeddings.py

# 2) Upsert to Pinecone (new index)
#   - For each photo: values = visual_embeddings[original_path] (512 floats)
#   - metadata: { original_path, text, filename }
#   - id: use the same photo_id as the semantic index (lookup by original_path) or a stable hash

# 3) Query with CLIP for text
#   - q_vec = clip.encode_text(["your prompt"]) -> 512-D
#   - pinecone.query(vector=q_vec, top_k=15)
#   - map ids -> /api/photo/{id}/thumbnail and /api/photo/{id}/image
```

## Categories (Optional)
- `GET /api/categories` → list of available category keys and descriptions.
- `GET /api/search/category?category=<key>&limit=<n>` → placeholder endpoint (returns structure, not real results yet).

## Errors & Edge Cases
- `400` on invalid query parameters (e.g., missing `q`, invalid `limit`).
- `404` on missing photo IDs (details route). Image routes fall back to a placeholder image instead of `404`.
- `500` on unexpected server errors with a short message in JSON.

## Example UI Flow (Photos then Proxies)
1) Call `GET /api/photos?limit=<n>` for an initial grid (real images first).
2) Call `GET /api/search/semantic?q=<text>&limit=<n>` for semantic results.
2) Render cards using each item’s `thumbnail_url` (proxy), title/description, and any metadata you want to show.
3) On click, open a viewer that loads `image_url` (proxy). If it’s a placeholder PNG, show a badge like “Image Unavailable”.
4) Optionally fetch `GET /api/photo/{id}` when opening the viewer to display full metadata.

## Performance Notes
- With `gemini=false`, search returns faster and uses only the local embedding model’s ranking.
- `limit` up to 50 is supported; use `20–30` for a snappy UI.
- Thumbnails are lighter to load first; lazy-load full images on demand.

## TL;DR for Frontend
- Use `thumbnail_url` for lists; `image_url` for detail view.
- Expect valid images or a placeholder PNG—never a broken image.
- Endpoints are simple `GET`s with JSON for lists/details and direct image bytes for proxies.
