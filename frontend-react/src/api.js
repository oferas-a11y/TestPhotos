export async function fetchJson(url){
  const r = await fetch(url, { headers: { 'Accept':'application/json' }})
  if(!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

export function absUrl(base, url){
  return url?.startsWith('/') ? `${base.replace(/\/$/,'')}${url}` : url
}

export async function getPhotos(base, limit=15){
  const u = `${base.replace(/\/$/,'')}/api/photos?limit=${limit}`
  return fetchJson(u)
}

export async function searchSemantic(base, q, limit=15, gemini=false){
  const params = new URLSearchParams({ q, limit: String(limit), gemini: String(gemini) })
  const u = `${base.replace(/\/$/,'')}/api/search/semantic?${params}`
  return fetchJson(u)
}

export async function probeThumb(url){
  try{
    const r = await fetch(url)
    const b = await r.blob()
    const type = b.type || r.headers.get('Content-Type') || ''
    const bytes = b.size || 0
    const placeholder = type.startsWith('image/png') && bytes < 5000
    return { type, bytes, placeholder }
  }catch(e){
    return { type:'', bytes:0, placeholder:true }
  }
}

