/* ---------------- Professional Bayesian Graph Frontend ---------------- */
const statusEl = document.getElementById('status');
const nodeSummaryEl = document.getElementById('node-summary-content');
const contribsEl = document.getElementById('contribs-content');
const pathsEl = document.getElementById('paths-content');
const searchEl = document.getElementById('search');
const fitBtn = document.getElementById('fit');
const recomputeBtn = document.getElementById('recompute');
const exportBtn = document.getElementById('export');
const zoomSlider = document.getElementById('zoom-slider');
const toggleAboutBtn = document.getElementById('toggle-about');
const themeToggleBtn = document.getElementById('theme-toggle');
const about = document.getElementById('about');
const histCanvas = document.getElementById('hist-canvas');
const histCaption = document.getElementById('hist-caption');

function setStatus(msg){ statusEl.textContent = msg || ''; }

/* ---------------- Theme-aware palette ---------------- */
const getCSSVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const palette = {
  get bg() { return getCSSVar('--bg'); },
  get panel() { return getCSSVar('--panel'); },
  get text() { return getCSSVar('--text'); },
  get muted() { return getCSSVar('--muted'); },
  get border() { return getCSSVar('--border'); },
  get high() { return getCSSVar('--node-high'); },
  get mid() { return getCSSVar('--node-mid'); },
  get low() { return getCSSVar('--node-low'); },
  get ink() { return getCSSVar('--ink'); },
  get accent() { return getCSSVar('--accent'); },
  get hover() { return getCSSVar('--hover'); },
  get selected() { return getCSSVar('--selected'); },
  get clicked() { return getCSSVar('--clicked'); },
  get edgeColor() { return getCSSVar('--edge-color'); },
  get edgeSelected() { return getCSSVar('--edge-selected'); }
};

/* ---------------- Utilities ---------------- */
const clamp = (v,min,max) => Math.max(min, Math.min(max,v));

function colorForBelief(b){
  if(b==null) return palette.muted;
  const c = Math.round(clamp(b,0,0.99)*100);
  if(c<34) return palette.low;
  if(c<67) return palette.mid;
  return palette.high;
}

function edgeColorForWeight(w){
  const ww = clamp(Number(w||0),0,1);
  if(ww<0.25) return palette.edgeColor;
  if(ww<0.6) return palette.edgeColor;
  return palette.edgeSelected;
}

function escapeHtml(s){ return (s||'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m])); }

async function fetchJSON(url, opts){
  const res = await fetch(url, opts);
  if(!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.json();
}

/* ---------------- Node sizing & font scaling ---------------- */
function computeNodeSizeForData(data){
  const label = (data.label||'').toString();
  const belief = clamp(Number(data.posterior??0),0,0.99);
  const base = 44;
  const labelFactor = Math.min(label.length,120) * 5;
  const beliefFactor = 50 * Math.sqrt(belief);
  return Math.round(base + labelFactor + beliefFactor);
}

function computeFontSizeForWidth(width){
  const minW=44, maxW=320, minF=11, maxF=18;
  const t = clamp((width-minW)/(maxW-minW),0,1);
  return Math.round(minF+(maxF-minF)*t);
}

/* ---------------- Cytoscape initialization ---------------- */
const cy = cytoscape({
  container: document.getElementById('graph'),
  wheelSensitivity: 0.22,
  layout: {name:'cose', animate:false},
  style:[
    { selector:'node', style:{
      shape:'ellipse',
      'background-color': ele=>colorForBelief(ele.data('posterior')),
      label: ele=>{ const l=ele.data('label')||''; const t=ele.data('type')||''; return t?`${l}\n(${t})`:l; },
      'text-wrap':'wrap',
      'text-max-width': ele=>Math.max(30, Math.round(computeNodeSizeForData(ele.data())*0.72)),
      'text-valign':'center',
      'text-halign':'center',
      color: palette.text,
      'text-outline-color': '#00000033',
      'text-outline-width':1.5,
      'font-family':'Inter, system-ui, "Segoe UI", Roboto, Ubuntu',
      'font-size': ele=>`${computeFontSizeForWidth(computeNodeSizeForData(ele.data()))}px`,
      width: ele=>computeNodeSizeForData(ele.data()),
      height: ele=>computeNodeSizeForData(ele.data()),
      'border-color': palette.ink,
      'border-width':1,
      'overlay-padding':6,
      'z-index':0
    }},
    { selector:'edge', style:{
      width: ele=>Math.max(1, Math.round(1+(Number(ele.data('w')||0)*3.5))),
      'line-color': ele=>edgeColorForWeight(Number(ele.data('w')||0)),
      'target-arrow-color': ele=>edgeColorForWeight(Number(ele.data('w')||0)),
      'target-arrow-shape':'triangle',
      'curve-style':'bezier',
      opacity:0.9
    }},
    { selector:':selected', style:{
      'border-color': palette.selected,
      'border-width':3,
      'shadow-blur':8,
      'shadow-color':palette.selected+'99',
      'z-index':20
    }},
    { selector:'.clicked', style:{
      'border-color': palette.clicked,
      'border-width':4,
      'shadow-blur':12,
      'shadow-color':palette.clicked+'99',
      'z-index':30
    }}
  ]
});

/* ---------------- Node appearance refresh ---------------- */
function refreshNodeAppearance(node){
  if(!node||!node.nonempty()) return;
  const data = node.data();
  const posterior = clamp(Number(data.posterior??0),0,0.99);
  const newSize = computeNodeSizeForData(data);
  const newFont = computeFontSizeForWidth(newSize);
  node.style({
    width: newSize,
    height: newSize,
    'font-size': `${newFont}px`,
    'text-max-width': Math.max(30, Math.round(newSize*0.72)),
    'background-color': colorForBelief(posterior)
  });
}

/* ---------------- Fetch network ---------------- */
async function fetchNetwork(){
  setStatus('Loading network…');
  try{
    const data = await fetchJSON('/api/v1/network');
    const nodes = (data.nodes||[]).map(n=>({
      data: {id:n.id, label:n.label, type:n.type, prior: n.prior||0, posterior: n.belief??null}
    }));
    const edges = (data.edges||[]).map(e=>({data:{source:e.source,target:e.target,w:Number(e.w||0)}}));

    cy.startBatch();
    cy.elements().remove();
    cy.add(nodes);
    cy.add(edges);
    cy.endBatch();

    cy.nodes().forEach(n=>refreshNodeAppearance(n));
    cy.layout({name:'cose',animate:false}).run();
    setStatus(`Ready — ${nodes.length} nodes, ${edges.length} edges`);
  }catch(e){ setStatus('Failed to load network: '+e.message); console.error(e);}
}

/* ---------------- SSE updates ---------------- */
function applyUpdates(evt){
  if(evt.type!=='confidence_updates') return;
  (evt.updates||[]).forEach(u=>{
    const node = cy.$id(u.id);
    if(!node.nonempty()) return;
    node.data('posterior', clamp(Number(u.new||0)/100,0,0.99));
    refreshNodeAppearance(node);
  });
}

/* ---------------- Node click panel ---------------- */
function fmtPct(x){ return `${Math.min(99, Math.round(x*100))}%`; }

async function onNodeClick(id){
  try{
    const [info, contribs, paths, hist] = await Promise.all([
      fetchJSON(`/api/v1/node?id=${encodeURIComponent(id)}`),
      fetchJSON(`/api/v1/contributions?id=${encodeURIComponent(id)}&topk=10`),
      fetchJSON(`/api/v1/paths?id=${encodeURIComponent(id)}&k=5&maxlen=4`),
      fetchJSON(`/api/v1/history?id=${encodeURIComponent(id)}`)
    ]);

    nodeSummaryEl.innerHTML =
      `<div><strong>Label:</strong> ${escapeHtml(info.name||'')}</div>
       <div><strong>ID:</strong> ${escapeHtml(info.id||'')}</div>
       <div><strong>Type:</strong> <span class="badge">${escapeHtml(info.type||'')}</span></div>
       <div><strong>Prior:</strong> ${fmtPct(info.prior||0)}</div>
       <div><strong>Posterior:</strong> ${fmtPct(info.posterior||0)}</div>`;

    contribsEl.innerHTML = contribs?.contributions?.length ? `<table class="table">
      <thead><tr><th>Parent</th><th>w</th><th>belief</th><th>score</th><th>Δ</th></tr></thead>
      <tbody>${contribs.contributions.map(c=>`<tr title="w·belief=${(c.w*c.belief).toFixed(3)}">
        <td><code>${escapeHtml(c.parent)}</code></td>
        <td>${c.w.toFixed(2)}</td>
        <td>${fmtPct(clamp(c.belief,0,0.99))}</td>
        <td>${c.score.toFixed(3)}</td>
        <td>${(c.marginal_lift*100).toFixed(1)} pp</td>
      </tr>`).join('')}</tbody></table>` : 'No contributions available.';

    pathsEl.innerHTML = paths?.paths?.length ? `<table class="table">
      <thead><tr><th>Path</th><th>Score</th></tr></thead>
      <tbody>${paths.paths.map(p=>`<tr title="Product of edge weights × start belief">
        <td>${p.path.map(x=>`<code>${escapeHtml(x)}</code>`).join(' → ')}</td>
        <td>${p.score.toFixed(3)}</td>
      </tr>`).join('')}</tbody></table>` : 'No paths found.';

    drawHistory(hist?.history||[], info);
  }catch(e){
    nodeSummaryEl.innerHTML=`<span class="muted">Failed: ${escapeHtml(e.message)}</span>`;
    contribsEl.textContent='—';
    pathsEl.textContent='—';
    drawHistory([],null);
  }
}

/* ---------------- History chart ---------------- */
function drawHistory(hist, info){
  const ctx = histCanvas.getContext('2d');
  ctx.clearRect(0,0,histCanvas.width,histCanvas.height);
  if(!hist?.length){ histCaption.textContent='No history yet.'; return; }

  const vals = hist.map(h=>(h.new>=0?h.new:(h.old>=0?h.old:0)));
  const n = vals.length, pad=6, W=histCanvas.width, H=histCanvas.height;
  const minY=0, maxY=100;

  ctx.strokeStyle=palette.high;
  ctx.lineWidth=2;
  ctx.beginPath();
  for(let i=0;i<n;i++){
    const x=pad+(W-2*pad)*(i/(Math.max(1,n-1)));
    const y=pad+(H-2*pad)*(1-( (vals[i]-minY)/(maxY-minY)));
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();

  ctx.fillStyle=palette.muted;
  ctx.font='11px system-ui, Segoe UI, Roboto';
  ctx.fillText(`${n} updates`, pad, 12);
  histCaption.textContent = info ? `${info.name} — latest ${Math.round((info.posterior||0)*100)}%` : '';
}

/* ---------------- Node selection ---------------- */
function selectAndFocus(id){
  const node = cy.$id(id);
  if(node.nonempty()){
    cy.$(':selected').unselect();
    node.select();
    cy.animate({center:{eles:node}, zoom:1.08},{duration:220});
  }
}

/* ---------------- Event hooks ---------------- */
function hookEvents(){
  cy.on('tap','node', evt=>{
    const node = evt.target;
    cy.$('.clicked').removeClass('clicked');
    node.addClass('clicked');
    onNodeClick(node.id());
  });

  searchEl.addEventListener('input',()=>{
    const q = (searchEl.value||'').trim().toLowerCase();
    cy.nodes().forEach(n=>{
      const m = (n.data('label')||'').toLowerCase().includes(q) || n.id().toLowerCase().includes(q);
      n.style('opacity', q ? (m?1:0.12) : 1);
    });
    if(q){
      const first = cy.nodes().filter(n=>(n.data('label')||'').toLowerCase().includes(q)||n.id().toLowerCase().includes(q))[0];
      if(first) selectAndFocus(first.id());
    }
  });

  fitBtn.addEventListener('click', ()=>cy.fit(50));

  exportBtn.addEventListener('click', async ()=>{
    try{
      const data = await fetchJSON('/api/v1/network');
      const blob = new Blob([JSON.stringify(data,null,2)],{type:'application/json'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a'); a.href=url; a.download='bayesian_graph.json';
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      URL.revokeObjectURL(url); setStatus('Graph exported as JSON');
    }catch(e){ setStatus('Export failed: '+e.message); }
  });

  zoomSlider.addEventListener('input', ()=>{ cy.zoom(parseFloat(zoomSlider.value)); });

  themeToggleBtn.addEventListener('click', ()=>{
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme==='light'?'dark':'light';
    document.documentElement.setAttribute('data-theme',newTheme);
    themeToggleBtn.textContent = newTheme==='light'?'☀️':'🌙';
    cy.style().update();
    cy.nodes().forEach(n=>refreshNodeAppearance(n));
  });

  toggleAboutBtn.addEventListener('click', ()=>{
    const hidden = about.classList.toggle('collapsed');
    toggleAboutBtn.textContent = hidden ? 'Show' : 'Hide';
    document.querySelector('main').style.gridTemplateColumns = hidden ? '0px 1fr':'380px 1fr';
  });

  recomputeBtn.addEventListener('click', async ()=>{
    recomputeBtn.disabled = true;
    const old = recomputeBtn.textContent;
    recomputeBtn.textContent = 'Recomputing…';
    setStatus('Running inference…');
    try{
      const res = await fetchJSON('/api/v1/recompute',{method:'POST'});
      setStatus(`Inference complete: ${res.updated} confidence updates pushed`);
      await fetchNetwork();
    }catch(e){ setStatus('Recompute failed: '+e.message); console.error(e);}
    finally{ recomputeBtn.disabled = false; recomputeBtn.textContent = old; }
  });
}

/* ---------------- Main ---------------- */
async function main(){
  await fetchNetwork();
  hookEvents();
  try{
    const es = new EventSource('/api/v1/stream');
    es.onmessage = ev => applyUpdates(JSON.parse(ev.data));
    es.onerror = ()=>setStatus('Stream disconnected…');
  }catch(e){ console.warn('SSE not supported', e);}
}

main().catch(e=>setStatus('Failed: '+e.message));
