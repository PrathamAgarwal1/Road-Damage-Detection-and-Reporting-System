/* ============================================================
   RoadSight Admin Dashboard — admin.js
   ============================================================ */

let map, markersLayer, heatLayer, hotspotLayer;
let allReports = [];
let allHotspots = [];
let currentModalId = null;
let mapFitted = false;
let trendChart = null, conditionChart = null;

// ── Helpers ────────────────────────────────────────────────
function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));
}
function safeUrl(u) {
  return typeof u === 'string' && u.startsWith('/static/') ? esc(u) : '';
}
function priorityColor(p) {
  return p === 'High' ? '#ef4444' : p === 'Medium' ? '#f59e0b' : '#22c55e';
}
function severityClass(l) {
  const m = { critical:'bg-red-100 text-red-700', poor:'bg-orange-100 text-orange-700',
               moderate:'bg-yellow-100 text-yellow-700', good:'bg-green-100 text-green-700' };
  return m[l] || 'bg-gray-100 text-gray-600';
}
function statusClass(s) {
  const m = { 'New':'bg-blue-100 text-blue-700', 'Scheduled':'bg-purple-100 text-purple-700',
              'In Progress':'bg-amber-100 text-amber-700', 'Resolved':'bg-green-100 text-green-700' };
  return m[s] || 'bg-gray-100 text-gray-600';
}
function healthColor(h) {
  return h == null ? '#6b7280' : h >= 60 ? '#16a34a' : h >= 35 ? '#d97706' : '#dc2626';
}
async function getJson(url, opts) {
  const res = await fetch(url, opts);
  if (res.status === 401) { window.location.href = '/admin/login'; throw new Error('Unauthorized'); }
  return res.json();
}

// ── Map ────────────────────────────────────────────────────
function initMap() {
  map = L.map('map').setView([20.5937, 78.9629], 5);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19, attribution: '&copy; OpenStreetMap'
  }).addTo(map);
  markersLayer = L.layerGroup().addTo(map);
  hotspotLayer = L.layerGroup().addTo(map);
}

function markerIcon(color) {
  const svg = encodeURIComponent(
    `<svg xmlns='http://www.w3.org/2000/svg' width='32' height='32'>` +
    `<circle cx='16' cy='16' r='13' fill='${color}' stroke='white' stroke-width='3'/>` +
    `<circle cx='16' cy='16' r='5' fill='white' opacity='0.85'/></svg>`
  );
  return L.icon({ iconUrl: `data:image/svg+xml,${svg}`, iconSize: [32,32], iconAnchor: [16,16] });
}

// ── Stats ──────────────────────────────────────────────────
async function loadStats() {
  try {
    const d = await getJson('/api/admin/stats');
    if (!d.success) return;
    document.getElementById('stat-total').textContent    = d.total;
    document.getElementById('stat-high').textContent     = d.by_priority.High;
    document.getElementById('stat-pending').textContent  = (d.by_status.New || 0) + (d.by_status.Scheduled || 0) + (d.by_status['In Progress'] || 0);
    document.getElementById('stat-resolved').textContent = d.by_status.Resolved || 0;
    document.getElementById('stat-hotspots').textContent = d.hotspots ?? 0;
    document.getElementById('stat-duplicates').textContent = d.duplicates_blocked ?? 0;
    document.getElementById('stat-resolution').textContent =
      d.avg_resolution_days != null ? `avg ${d.avg_resolution_days} days to fix` : '';
  } catch(e) { console.warn('Stats load failed', e); }
}

// ── Analytics ──────────────────────────────────────────────
async function loadAnalytics() {
  if (typeof Chart === 'undefined') return;
  try {
    const days = document.getElementById('analytics-days').value;
    const d = await getJson(`/api/admin/analytics?days=${days}`);
    if (!d.success) return;
    const labels = d.labels.map(x => new Date(x + 'T00:00:00').toLocaleDateString('en-IN', { day: 'numeric', month: 'short' }));
    if (trendChart) trendChart.destroy();
    trendChart = new Chart(document.getElementById('trendChart'), {
      type: 'line',
      data: { labels, datasets: [
        { label: 'Reported', data: d.reported, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,.12)', fill: true, cubicInterpolationMode: 'monotone', pointRadius: 0 },
        { label: 'Resolved', data: d.resolved, borderColor: '#22c55e', backgroundColor: 'rgba(34,197,94,.10)', fill: true, cubicInterpolationMode: 'monotone', pointRadius: 0 },
      ]},
      options: { maintainAspectRatio: false, interaction: { mode: 'index', intersect: false },
                 plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 11 } } } },
                 scales: { y: { beginAtZero: true, ticks: { precision: 0 } }, x: { ticks: { maxTicksLimit: 8, font: { size: 10 } } } } },
    });
    const order = [['critical', 'Very Poor', '#ef4444'], ['poor', 'Poor', '#f97316'], ['moderate', 'Satisfactory', '#f59e0b'], ['good', 'Good', '#22c55e']];
    if (conditionChart) conditionChart.destroy();
    conditionChart = new Chart(document.getElementById('conditionChart'), {
      type: 'doughnut',
      data: { labels: order.map(o => o[1]), datasets: [{ data: order.map(o => d.by_condition[o[0]] || 0), backgroundColor: order.map(o => o[2]), borderWidth: 2 }] },
      options: { maintainAspectRatio: false, cutout: '62%', plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 11 } } } } },
    });
    document.getElementById('avg-rhi').textContent = d.avg_open_rhi != null ? `Avg open road health ${d.avg_open_rhi}/100` : '';
  } catch (e) { console.warn('Analytics load failed', e); }
}

// ── Hotspots ───────────────────────────────────────────────
async function loadHotspots() {
  try {
    const d = await getJson('/api/admin/hotspots');
    if (!d.success) return;
    allHotspots = d.hotspots || [];
    renderHotspots();
  } catch(e) { console.warn('Hotspots load failed', e); }
}

function renderHotspots() {
  hotspotLayer.clearLayers();
  const showHot = document.getElementById('toggle-hotspots').checked;
  const list = document.getElementById('hotspot-list');
  if (!allHotspots.length) {
    list.innerHTML = '<div class="p-4 text-center text-gray-400 text-xs">No hotspots yet — clusters appear when several open reports fall within 500 m.</div>';
  } else {
    list.innerHTML = allHotspots.slice(0, 10).map((h, i) => `
      <button class="hotspot-item w-full text-left px-4 py-2 hover:bg-gray-50" data-idx="${i}">
        <div class="flex items-center justify-between gap-2">
          <span class="text-sm font-medium text-gray-700 truncate">#${i + 1} ${esc(h.address || 'Unknown area')}</span>
          <span class="text-xs font-semibold text-red-600 flex-shrink-0">${h.weight} reports</span>
        </div>
        <div class="text-xs text-gray-400">Avg severity ${h.avgSeverity} · ${h.highPriority} high priority</div>
      </button>`).join('');
    list.querySelectorAll('.hotspot-item').forEach(btn => btn.addEventListener('click', () => {
      const h = allHotspots[+btn.dataset.idx];
      map.setView([h.latitude, h.longitude], 15);
    }));
  }
  if (!showHot) return;
  allHotspots.forEach((h, i) => {
    L.circle([h.latitude, h.longitude], {
      radius: 500, color: '#dc2626', weight: 2, fillColor: '#ef4444', fillOpacity: 0.12
    }).bindPopup(`<div class="text-sm"><strong>Hotspot #${i + 1}</strong><br>${h.weight} reports · avg severity ${h.avgSeverity}<br>${h.highPriority} high priority</div>`)
      .addTo(hotspotLayer);
  });
}

// ── Reports ────────────────────────────────────────────────
function renderMap(reports) {
  markersLayer.clearLayers();
  if (heatLayer) { map.removeLayer(heatLayer); heatLayer = null; }
  const bounds = [];
  const heatPts = [];
  reports.forEach(r => {
    const lat = r.location?.latitude, lng = r.location?.longitude;
    if (lat == null || lng == null) return;
    const color = priorityColor(r.priority);
    const m = L.marker([lat, lng], { icon: markerIcon(color) });
    m.bindPopup(`
      <div style="min-width:160px">
        <div class="font-semibold text-sm">${esc(r.location?.address || 'Unknown')}</div>
        <div class="text-xs mt-1">Severity: <strong>${esc(r.severity?.level || '—')}</strong></div>
        <div class="text-xs">Status: <strong>${esc(r.status)}</strong></div>
        <div class="text-xs" style="color:${color}">Priority: <strong>${esc(r.priority)}</strong></div>
        <button onclick="openModal('${esc(r.id)}')" style="margin-top:6px;color:#3b82f6;font-size:12px;cursor:pointer;">View details →</button>
      </div>`);
    m.addTo(markersLayer);
    bounds.push([lat, lng]);
    if (r.status !== 'Resolved') {
      heatPts.push([lat, lng, Math.min(1, (r.severity?.score ?? 0.4) * (1 + (r.confirmedBy?.length || 0) * 0.5))]);
    }
  });
  if (bounds.length && !mapFitted) { map.fitBounds(bounds, { padding: [40, 40], maxZoom: 15 }); mapFitted = true; }
  if (document.getElementById('toggle-heatmap').checked && L.heatLayer && heatPts.length) {
    try {
      map.invalidateSize();
      heatLayer = L.heatLayer(heatPts, { radius: 30, blur: 20, maxZoom: 15 }).addTo(map);
    } catch (e) {
      console.warn('Heatmap unavailable', e);
      if (heatLayer) { map.removeLayer(heatLayer); heatLayer = null; }
    }
  }
}

function renderList(reports) {
  const list = document.getElementById('report-list');
  if (reports.length === 0) {
    list.innerHTML = '<div class="p-6 text-center text-gray-400 text-sm">No reports match the current filters.</div>';
    return;
  }
  list.innerHTML = '';
  reports.forEach(r => {
    const pColor = priorityColor(r.priority);
    const date = r.createdAt ? new Date(r.createdAt).toLocaleDateString('en-IN', { day:'numeric', month:'short', year:'numeric' }) : '—';
    const rhi = r.forecast?.roadHealthIndex;
    const confirmations = r.confirmedBy?.length || 0;

    const card = document.createElement('div');
    card.className = 'report-card px-4 py-3 cursor-pointer';
    card.innerHTML = `
      <div class="flex gap-3 items-start">
        <img src="${safeUrl(r.imageUrl)}" class="w-14 h-14 rounded-lg object-cover border border-gray-100 flex-shrink-0"
             onerror="this.style.display='none'">
        <div class="flex-1 min-w-0">
          <div class="flex items-start justify-between gap-2">
            <div class="text-sm font-medium text-gray-800 truncate">${esc(r.location?.address || 'Unknown location')}</div>
            <span class="text-xs px-2 py-0.5 rounded-full font-medium flex-shrink-0 ${statusClass(r.status)}">${esc(r.status)}</span>
          </div>
          <div class="flex items-center gap-2 mt-1 flex-wrap">
            <span class="text-xs px-2 py-0.5 rounded-full font-medium ${severityClass(r.severity?.level)}">${esc(r.severity?.level || '—')}</span>
            <span class="text-xs font-semibold" style="color:${pColor}">${esc(r.priority)} Priority</span>
            ${rhi != null ? `<span class="text-xs font-semibold" style="color:${healthColor(rhi)}">RHI ${rhi}</span>` : ''}
            ${confirmations ? `<span class="text-xs text-indigo-600"><i class="fa-solid fa-users"></i> +${confirmations}</span>` : ''}
            ${r.needsReview ? `<span class="text-xs px-2 py-0.5 rounded-full font-medium bg-slate-800 text-white" title="Road photo could not be fully verified"><i class="fa-solid fa-flag"></i> Review</span>` : ''}
            <span class="text-xs text-gray-400">${date}</span>
          </div>
          <div class="text-xs text-gray-400 mt-1 truncate">
            ${esc(r.reporter?.name || 'Anonymous')}${r.reporter?.email ? ' · ' + esc(r.reporter.email) : ''}
          </div>
        </div>
      </div>`;
    card.addEventListener('click', () => openModal(r.id));
    list.appendChild(card);
  });
}

function renderReports(reports) {
  allReports = reports;
  document.getElementById('reportCount').textContent = `${reports.length} report${reports.length !== 1 ? 's' : ''}`;
  const ts = document.getElementById('lastUpdated');
  if (ts) ts.textContent = `Updated ${new Date().toLocaleTimeString()}`;
  renderList(reports);
  renderMap(reports);
}

function filterQuery() {
  const qs = new URLSearchParams();
  for (const [key, id] of [['severity','filter-severity'], ['status','filter-status'], ['priority','filter-priority'], ['sort','filter-sort'], ['q','filter-q']]) {
    const v = document.getElementById(id).value.trim();
    if (v) qs.set(key, v);
  }
  return qs.toString();
}

async function loadReports() {
  try {
    const json = await getJson(`/api/reports?${filterQuery()}`);
    if (json.success) renderReports(json.reports);
  } catch(e) { console.error('loadReports failed', e); }
}

function refreshAll() {
  loadReports(); loadStats(); loadHotspots(); loadAnalytics();
}

// ── Modal ──────────────────────────────────────────────────
function detailRow(label, html, full = false) {
  return `<div class="${full ? 'col-span-2' : ''}">
    <div class="text-xs text-gray-400 uppercase tracking-wide font-medium mb-1">${label}</div>${html}</div>`;
}

async function loadTimeline(reportId) {
  const el = document.getElementById('modal-timeline');
  try {
    const d = await getJson(`/api/reports/${encodeURIComponent(reportId)}/timeline`);
    if (!d.success || currentModalId !== reportId) return;
    if (!d.milestones.length) { el.innerHTML = '<div class="text-xs text-gray-400 italic">No updates yet.</div>'; return; }
    el.innerHTML = d.milestones.map(m => `
      <div class="flex gap-2 text-xs">
        <div class="w-2 h-2 rounded-full bg-blue-500 mt-1 flex-shrink-0"></div>
        <div><div class="font-semibold text-gray-700">${esc(m.title)}</div>
          <div class="text-gray-500">${esc(m.description || '')}</div>
          <div class="text-gray-400">${new Date(m.createdAt).toLocaleString('en-IN')} · ${esc(m.createdBy || '')}</div></div>
      </div>`).join('');
  } catch(e) { el.innerHTML = '<div class="text-xs text-red-500">Could not load timeline.</div>'; }
}

function openModal(reportId) {
  currentModalId = reportId;
  const r = allReports.find(x => x.id === reportId);
  if (!r) return;
  const modal = document.getElementById('detailModal');
  const body  = document.getElementById('modalBody');
  const pColor = priorityColor(r.priority);
  const date = r.createdAt ? new Date(r.createdAt).toLocaleString('en-IN') : '—';
  const f = r.forecast || {};
  const w = r.weather || {};
  const confirmations = r.confirmedBy?.length || 0;

  body.innerHTML = `
    ${safeUrl(r.imageUrl) ? `<img src="${safeUrl(r.imageUrl)}" class="w-full h-48 object-cover rounded-xl border border-gray-100 mb-4">` : ''}
    <div class="grid grid-cols-2 gap-3 text-sm">
      ${detailRow('Location', `<div class="font-semibold text-gray-800">${esc(r.location?.address || '—')}</div>
        ${r.location?.latitude != null ? `<div class="text-xs text-gray-400">${r.location.latitude.toFixed(5)}, ${r.location.longitude.toFixed(5)}</div>` : ''}`, true)}
      ${detailRow('Condition', `<span class="px-2 py-0.5 rounded-full text-xs font-semibold ${severityClass(r.severity?.level)}">${esc(r.condition || r.severity?.level || '—')}</span>
        <div class="text-xs text-gray-500 mt-1">Confidence: ${typeof r.confidence === 'number' ? r.confidence.toFixed(1) + '%' : '—'}</div>`)}
      ${detailRow('Priority', `<span class="font-bold text-sm" style="color:${pColor}">${esc(r.priority)}</span>
        <div class="text-xs text-gray-500 mt-1">Score ${(r.priorityScore ?? 0).toFixed(2)} · Nearby ${r.reportDensity ?? 0} · Confirmations ${confirmations}</div>`)}
      ${detailRow('Road Health Index', f.roadHealthIndex != null
        ? `<span class="font-bold" style="color:${healthColor(f.roadHealthIndex)}">${f.roadHealthIndex}/100</span>
           <span class="text-xs text-gray-400">→ ${f.roadHealthIndex30d}/100 in 30 days</span>`
        : '<span class="text-gray-400">—</span>')}
      ${detailRow('Weather Risk', w.available
        ? `<span class="font-semibold">${(r.predictiveRisk ?? 0).toFixed(2)}</span>
           <div class="text-xs text-gray-500">Rain ${w.pastRainMm} mm past 7d / ${w.forecastRainMm} mm next 7d · ${w.freezeThawCycles} freeze-thaw · ${w.heavyRainDays} heavy-rain days</div>`
        : `<span class="text-gray-400">${(r.predictiveRisk ?? 0).toFixed(2)} (no weather data)</span>`)}
      ${r.needsReview ? detailRow('Needs review', `<div class="text-xs bg-slate-100 text-slate-700 rounded-lg p-2"><i class="fa-solid fa-flag mr-1"></i>The photo could not be fully verified as a road
        (${esc(r.roadCheck?.method || 'unknown')}${r.roadCheck?.score != null ? ', similarity ' + r.roadCheck.score : ''}). Check the photo, then update the status or remove it as spam.</div>`, true) : ''}
      ${f.summary ? detailRow('Deterioration Forecast', `<div class="text-gray-700 bg-amber-50 rounded-lg p-2 text-xs">${esc(f.summary)}</div>`, true) : ''}
      ${detailRow('Reporter', `<div class="text-gray-700">${esc(r.reporter?.name || 'Anonymous')}</div>
        <div class="text-xs text-gray-400">${esc(r.reporter?.email || '—')}</div>`)}
      ${detailRow('Submitted', `<div class="text-gray-600 text-xs">${date}</div>
        ${r.assignedUnit ? `<div class="text-xs text-gray-500 mt-1">Unit: ${esc(r.assignedUnit)}${r.scheduledFor ? ' · ' + new Date(r.scheduledFor).toLocaleDateString('en-IN') : ''}</div>` : ''}`)}
      ${r.description ? detailRow('Description', `<div class="text-gray-700 text-sm leading-relaxed bg-gray-50 rounded-lg p-3">${esc(r.description)}</div>`, true) : ''}
    </div>

    <div class="border-t border-gray-100 pt-4 mt-2 space-y-3">
      <div>
        <label class="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Update Status</label>
        <div class="flex gap-2">
          <select id="modal-status" class="flex-1 border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400">
            ${['New','Scheduled','In Progress','Resolved'].map(s =>
              `<option ${s === r.status ? 'selected' : ''}>${s}</option>`).join('')}
          </select>
          <button id="modal-status-btn" class="bg-blue-600 hover:bg-blue-700 text-white rounded-lg px-4 py-2 text-sm font-medium transition">Save</button>
        </div>
        <input id="modal-note" type="text" maxlength="500" placeholder="Optional note for the citizen"
          class="mt-2 w-full border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400">
      </div>
      <div>
        <label class="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Assign & Schedule Field Unit</label>
        <div class="flex gap-2 flex-wrap">
          <input id="modal-unit" type="text" maxlength="100" placeholder="e.g. Unit-7 / PWD Team Alpha"
            class="flex-1 min-w-[160px] border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400">
          <input id="modal-date" type="date" class="border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400">
          <button id="modal-assign-btn" class="bg-gray-800 hover:bg-gray-900 text-white rounded-lg px-4 py-2 text-sm font-medium transition">Assign</button>
        </div>
      </div>
      <div id="modal-feedback" class="text-sm hidden"></div>
      <div class="flex justify-end">
        <button id="modal-delete-btn" class="text-xs text-red-500 hover:text-red-700 font-medium"><i class="fa-solid fa-trash-can mr-1"></i>Remove as spam / invalid</button>
      </div>
      <div>
        <div class="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Timeline</div>
        <div id="modal-timeline" class="space-y-2"><div class="text-xs text-gray-400">Loading…</div></div>
      </div>
    </div>`;

  document.getElementById('modal-status-btn').addEventListener('click', async () => {
    const status = document.getElementById('modal-status').value;
    const note = document.getElementById('modal-note').value.trim();
    const btn = document.getElementById('modal-status-btn');
    btn.disabled = true; btn.textContent = 'Saving…';
    try {
      const j = await getJson(`/api/reports/${encodeURIComponent(reportId)}/status`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ status, note })
      });
      if (j.success) {
        showModalFeedback('Status updated to: ' + status);
        document.getElementById('modal-note').value = '';
        loadTimeline(reportId);
        refreshAll();
      } else showModalFeedback(j.error || 'Update failed', true);
    } catch(e) { showModalFeedback('Update failed', true); }
    finally { btn.disabled = false; btn.textContent = 'Save'; }
  });

  document.getElementById('modal-assign-btn').addEventListener('click', async () => {
    const unit = document.getElementById('modal-unit').value.trim();
    const scheduledFor = document.getElementById('modal-date').value;
    if (!unit) return showModalFeedback('Enter a field unit name.', true);
    const btn = document.getElementById('modal-assign-btn');
    btn.disabled = true; btn.textContent = 'Assigning…';
    try {
      const j = await getJson(`/api/reports/${encodeURIComponent(reportId)}/assign`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ unit, scheduledFor: scheduledFor || null })
      });
      if (j.success) {
        showModalFeedback(`Assigned to "${unit}" — status: ${j.status}.`);
        document.getElementById('modal-unit').value = '';
        document.getElementById('modal-status').value = j.status;
        loadTimeline(reportId);
        refreshAll();
      } else showModalFeedback(j.error || 'Assignment failed', true);
    } catch(e) { showModalFeedback('Assignment failed', true); }
    finally { btn.disabled = false; btn.textContent = 'Assign'; }
  });

  document.getElementById('modal-delete-btn').addEventListener('click', async () => {
    if (!confirm('Remove this report permanently? It will be kept in the audit log only.')) return;
    try {
      const j = await getJson(`/api/reports/${encodeURIComponent(reportId)}`, { method: 'DELETE' });
      if (j.success) { closeModal(); refreshAll(); }
      else showModalFeedback(j.error || 'Could not remove report', true);
    } catch(e) { showModalFeedback('Could not remove report', true); }
  });

  loadTimeline(reportId);
  modal.classList.remove('hidden');
  modal.classList.add('flex');
}

function closeModal() {
  const modal = document.getElementById('detailModal');
  modal.classList.add('hidden');
  modal.classList.remove('flex');
  currentModalId = null;
}

function showModalFeedback(msg, isError = false) {
  const el = document.getElementById('modal-feedback');
  el.textContent = (isError ? '✗ ' : '✓ ') + msg;
  el.className = `text-sm ${isError ? 'text-red-600' : 'text-green-600'}`;
  setTimeout(() => el.classList.add('hidden'), 3500);
}

// ── Init ───────────────────────────────────────────────────
window.openModal  = openModal;
window.closeModal = closeModal;

async function checkAdminSession() {
  try {
    const res = await fetch('/api/admin/check');
    const data = await res.json();
    if (data.success && data.admin) {
      document.getElementById('admin-email').textContent = data.admin.email;
      return true;
    }
  } catch (e) {
    console.error('Admin session check failed', e);
  }
  window.location.href = '/admin/login';
  return false;
}

window.addEventListener('DOMContentLoaded', async () => {
  const loggedIn = await checkAdminSession();
  if (!loggedIn) return;
  initMap();
  document.getElementById('refresh').addEventListener('click', refreshAll);
  ['filter-severity', 'filter-status', 'filter-priority', 'filter-sort'].forEach(id =>
    document.getElementById(id).addEventListener('change', loadReports));
  let searchTimer;
  document.getElementById('filter-q').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(loadReports, 300); });
  document.getElementById('export').addEventListener('click', () => { window.location.href = `/api/reports/export.csv?${filterQuery()}`; });
  document.getElementById('analytics-days').addEventListener('change', loadAnalytics);
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && currentModalId) closeModal(); });
  document.getElementById('toggle-heatmap').addEventListener('change', () => renderMap(allReports));
  document.getElementById('toggle-hotspots').addEventListener('change', renderHotspots);
  document.getElementById('closeModal').addEventListener('click', closeModal);
  document.getElementById('detailModal').addEventListener('click', e => {
    if (e.target === document.getElementById('detailModal')) closeModal();
  });

  refreshAll();
  setInterval(refreshAll, 20000);
});
