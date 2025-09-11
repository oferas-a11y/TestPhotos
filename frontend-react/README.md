# React Frontend (Black & White)

A lightweight React (Vite) app that mirrors the local dashboard interactions:
- Browse first N photos
- Semantic search
- REAL/PLACEHOLDER badges for thumbnails
- Open full image in a new tab

## Run locally
```
cd frontend-react
npm install
npm run dev
# open http://localhost:5173
```

Set API base at the top input (saved to localStorage), e.g.:
- https://testphotos-production.up.railway.app

## Build/preview
```
npm run build
npm run preview  # http://localhost:5174
```

## Vercel deployment (static)
- Framework: Other
- Build Command: `npm run build`
- Output Directory: `frontend-react/dist`
- No server needed. The API already supports CORS.

## Env (optional)
- Copy `.env.example` to `.env` and edit `VITE_API_BASE_URL`.
- The runtime UI base switcher overrides this value; rebuild not required.

## Notes
- PLACEHOLDER means your backend could not find a real image. Ensure
  `IMAGE_CDN_BASE` + `IMAGE_PATH_MODE=fullpath` are set on the server and your
  R2 bucket is public, per `FRONTEND_TROUBLESHOOTING.md`.

