import React, { useEffect, useMemo, useState } from 'react'
import { fetchJson, getPhotos, searchSemantic, absUrl, probeThumb } from './api.js'

const DEFAULT_BASE = import.meta.env.VITE_API_BASE_URL || 'https://testphotos-production.up.railway.app'

const Label = ({ placeholder }) => (
  <span className={`badge ${placeholder ? 'warn' : 'ok'}`}>{placeholder ? 'PLACEHOLDER' : 'REAL'}</span>
)

export default function App(){
  const [base, setBase] = useState(localStorage.getItem('apiBase') || DEFAULT_BASE)
  const [query, setQuery] = useState('love')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState('')

  useEffect(() => {
    localStorage.setItem('apiBase', base)
  }, [base])

  async function runBrowse(){
    setLoading(true); setStatus('Loading photos...')
    try{
      const data = await getPhotos(base, 15)
      setResults(data.results || [])
      setStatus('')
    }catch(e){ setStatus('Load failed: '+e.message) }
    finally{ setLoading(false) }
  }

  async function runSearch(){
    if(!query.trim()) return
    setLoading(true); setStatus('Searching...')
    try{
      const data = await searchSemantic(base, query.trim(), 15, false)
      setResults(data.results || [])
      setStatus('')
    }catch(e){ setStatus('Search failed: '+e.message) }
    finally{ setLoading(false) }
  }

  useEffect(() => { runBrowse() }, [])

  return (
    <div className="app bw-theme">
      <header className="hdr">
        <h1>Historical Photos — React</h1>
        <div className="actions">
          <input className="txt" value={base} onChange={e=>setBase(e.target.value)} placeholder="API Base (https://...)" />
          <button onClick={runBrowse}>Browse 15</button>
          <input className="txt" value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search (e.g., love)" />
          <button onClick={runSearch}>Search</button>
        </div>
      </header>
      {status && <div className="status">{status}</div>}
      <Grid base={base} items={results} />
    </div>
  )
}

function Grid({ base, items }){
  const [meta, setMeta] = useState({})

  useEffect(() => {
    let cancelled = false
    async function probeAll(){
      const out = {}
      for(const it of items){
        const thumb = absUrl(base, it.thumbnail_url)
        try{
          const info = await probeThumb(thumb)
          out[it.id] = info
        }catch{ out[it.id] = { placeholder:true, bytes:0, type:'' } }
      }
      if(!cancelled) setMeta(out)
    }
    probeAll()
    return () => { cancelled = true }
  }, [base, items])

  return (
    <main className="grid">
      {items.map(it => {
        const thumb = absUrl(base, it.thumbnail_url)
        const image = absUrl(base, it.image_url)
        const info = meta[it.id] || { placeholder: true, bytes: 0, type: '' }
        return (
          <div className="card" key={it.id}>
            <a href={image} target="_blank" rel="noreferrer"><img src={thumb} /></a>
            <div className="meta">
              <Label placeholder={!!info.placeholder} />
              <small>{info.type || 'unknown'} · {info.bytes}B</small>
            </div>
          </div>
        )
      })}
    </main>
  )
}

