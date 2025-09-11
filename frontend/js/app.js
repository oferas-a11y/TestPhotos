const $ = (sel) => document.querySelector(sel);
const grid = $('#grid');
const statusEl = $('#status');

const DEFAULT_BASE = 'https://testphotos-production.up.railway.app';

function getBase(){
  return localStorage.getItem('apiBase') || DEFAULT_BASE;
}

function setBase(v){
  localStorage.setItem('apiBase', v);
}

function badge(placeholder){
  return `<span class="badge ${placeholder ? 'warn':'ok'}">${placeholder ? 'PLACEHOLDER':'REAL'}</span>`;
}

async function fetchJSON(url){
  const r = await fetch(url, { headers: { 'Accept':'application/json' }});
  if(!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}

async function probeThumb(url){
  try{
    const r = await fetch(url);
    const b = await r.blob();
    // Browser blob.type may be empty; fallback to header
    const type = b.type || r.headers.get('Content-Type') || '';
    const bytes = b.size || 0;
    const placeholder = type.startsWith('image/png') && bytes < 5000;
    return { type, bytes, placeholder };
  }catch(e){ return { type:'', bytes:0, placeholder:true }; }
}

function absUrl(base, url){
  return url.startsWith('/') ? `${base.replace(/\/$/,'')}${url}` : url;
}

async function render(items, base){
  grid.innerHTML = '';
  for(const it of items){
    const thumb = absUrl(base, it.thumbnail_url);
    const image = absUrl(base, it.image_url);
    const info = await probeThumb(thumb);
    const el = document.createElement('div');
    el.className = 'card';
    el.innerHTML = `
      <a href="${image}" target="_blank"><img src="${thumb}"/></a>
      <div class="meta">
        ${badge(info.placeholder)}
        <small>${info.type || 'unknown'} · ${info.bytes}B</small>
      </div>
    `;
    grid.appendChild(el);
  }
}

async function browse(){
  const base = getBase();
  statusEl.textContent = 'Loading photos...';
  try{
    const data = await fetchJSON(`${base.replace(/\/$/,'')}/api/photos?limit=15`);
    await render(data.results || [], base);
    statusEl.textContent = '';
  }catch(e){ statusEl.textContent = 'Load failed: '+e.message; }
}

async function search(){
  const base = getBase();
  const q = $('#q').value.trim();
  if(!q){ return; }
  statusEl.textContent = 'Searching...';
  try{
    const params = new URLSearchParams({ q, limit: '15', gemini:'false' });
    const data = await fetchJSON(`${base.replace(/\/$/,'')}/api/search/semantic?${params}`);
    await render(data.results || [], base);
    statusEl.textContent = '';
  }catch(e){ statusEl.textContent = 'Search failed: '+e.message; }
}

function init(){
  // Wire inputs
  $('#base').value = getBase();
  $('#saveBase').onclick = () => { setBase($('#base').value.trim()); browse(); };
  $('#btnBrowse').onclick = browse;
  $('#btnSearch').onclick = search;
  // First load
  browse();
}

init();

