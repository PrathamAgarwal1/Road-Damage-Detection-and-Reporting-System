/* ============================================================
   RoadSight Admin Dashboard — admin.js
   ============================================================ */

let map, markersLayer;
let allReports = [];
let currentModalId = null;

// ── Colour helpers ─────────────────────────────────────────
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

// ── Map ────────────────────────────────────────────────────
function initMap() {
  map = L.map('map').setView([20.5937, 78.9629], 5);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19, attribution: '&copy; OpenStreetMap'
  }).addTo(map);
  markersLayer = L.layerGroup().addTo(map);
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
    const res = await fetch('/api/admin/stats');
    const d = await res.json();
    if (!d.success) return;
    document.getElementById('stat-total').textContent    = d.total;
    document.getElementById('stat-high').textContent     = d.by_priority.High;
    document.getElementById('stat-pending').textContent  = (d.by_status.New || 0) + (d.by_status.Scheduled || 0) + (d.by_status['In Progress'] || 0);
    document.getElementById('stat-resolved').textContent = d.by_status.Resolved || 0;
  } catch(e) { console.warn('Stats load failed', e); }
}

// ── Reports ────────────────────────────────────────────────
function renderMap(reports) {
  markersLayer.clearLayers();
  const bounds = [];
  reports.forEach(r => {
    const lat = r.location?.latitude, lng = r.location?.longitude;
    if (lat == null || lng == null) return;
    const color = priorityColor(r.priority);
    const m = L.marker([lat, lng], { icon: markerIcon(color) });
    m.bindPopup(`
      <div style="min-width:160px">
        <div class="font-semibold text-sm">${r.location?.address || 'Unknown'}</div>
        <div class="text-xs mt-1">Severity: <strong>${r.severity?.level || '—'}</strong></div>
        <div class="text-xs">Status: <strong>${r.status}</strong></div>
        <div class="text-xs" style="color:${color}">Priority: <strong>${r.priority}</strong></div>
        <button onclick="openModal('${r.id}')" style="margin-top:6px;color:#3b82f6;font-size:12px;cursor:pointer;">View details →</button>
      </div>`);
    m.addTo(markersLayer);
    bounds.push([lat, lng]);
  });
  if (bounds.length) map.fitBounds(bounds, { padding: [40, 40] });
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
    const sClass = severityClass(r.severity?.level);
    const stClass = statusClass(r.status);
    const date = r.createdAt ? new Date(r.createdAt).toLocaleDateString('en-IN', { day:'numeric', month:'short', year:'numeric' }) : '—';

    const card = document.createElement('div');
    card.className = 'report-card px-4 py-3 cursor-pointer';
    card.innerHTML = `
      <div class="flex gap-3 items-start">
        <img src="${r.imageUrl || ''}" class="w-14 h-14 rounded-lg object-cover border border-gray-100 flex-shrink-0"
             onerror="this.style.display='none'">
        <div class="flex-1 min-w-0">
          <div class="flex items-start justify-between gap-2">
            <div class="text-sm font-medium text-gray-800 truncate">${r.location?.address || 'Unknown location'}</div>
            <span class="text-xs px-2 py-0.5 rounded-full font-medium flex-shrink-0 ${stClass}">${r.status}</span>
          </div>
          <div class="flex items-center gap-2 mt-1 flex-wrap">
            <span class="text-xs px-2 py-0.5 rounded-full font-medium ${sClass}">${r.severity?.level || '—'}</span>
            <span class="text-xs font-semibold" style="color:${pColor}">${r.priority} Priority</span>
            <span class="text-xs text-gray-400">${date}</span>
          </div>
          <div class="text-xs text-gray-400 mt-1 truncate">
            ${r.reporter?.name || 'Anonymous'}${r.reporter?.email ? ' · ' + r.reporter.email : ''}
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
  renderMap(reports);
  renderList(reports);
}

async function loadReports() {
  const severity = document.getElementById('filter-severity').value;
  const status   = document.getElementById('filter-status').value;
  const sort     = document.getElementById('filter-sort').value;
  const qs = new URLSearchParams();
  if (severity) qs.set('severity', severity);
  if (status)   qs.set('status', status);
  if (sort)     qs.set('sort', sort);
  try {
    const res = await fetch(`/api/reports?${qs.toString()}`);
    const json = await res.json();
    if (json.success) renderReports(json.reports);
  } catch(e) { console.error('loadReports failed', e); }
}

// ── Modal ──────────────────────────────────────────────────
function openModal(reportId) {
  currentModalId = reportId;
  const r = allReports.find(x => x.id === reportId);
  if (!r) return;
  const modal = document.getElementById('detailModal');
  const body  = document.getElementById('modalBody');
  const pColor = priorityColor(r.priority);
  const sClass = severityClass(r.severity?.level);
  const stClass = statusClass(r.status);
  const date = r.createdAt ? new Date(r.createdAt).toLocaleString('en-IN') : '—';

  body.innerHTML = `
    ${r.imageUrl ? `<img src="${r.imageUrl}" class="w-full h-48 object-cover rounded-xl border border-gray-100 mb-4">` : ''}
    <div class="grid grid-cols-2 gap-3 text-sm">
      <div class="col-span-2">
        <div class="text-xs text-gray-400 uppercase tracking-wide font-medium mb-1">Location</div>
        <div class="font-semibold text-gray-800">${r.location?.address || '—'}</div>
        ${r.location?.latitude ? `<div class="text-xs text-gray-400">${r.location.latitude.toFixed(5)}, ${r.location.longitude.toFixed(5)}</div>` : ''}
      </div>
      <div>
        <div class="text-xs text-gray-400 uppercase tracking-wide font-medium mb-1">Condition</div>
        <span class="px-2 py-0.5 rounded-full text-xs font-semibold ${sClass}">${r.severity?.level || '—'}</span>
        <div class="text-xs text-gray-500 mt-1">Confidence: ${r.confidence ? r.confidence.toFixed(1) + '%' : '—'}</div>
      </div>
      <div>
        <div class="text-xs text-gray-400 uppercase tracking-wide font-medium mb-1">Priority</div>
        <span class="font-bold text-sm" style="color:${pColor}">${r.priority}</span>
        <div class="text-xs text-gray-500 mt-1">Risk: ${(r.predictiveRisk ?? 0).toFixed(2)} · Density: ${r.reportDensity ?? 0}</div>
      </div>
      <div>
        <div class="text-xs text-gray-400 uppercase tracking-wide font-medium mb-1">Reporter</div>
        <div class="text-gray-700">${r.reporter?.name || 'Anonymous'}</div>
        <div class="text-xs text-gray-400">${r.reporter?.email || '—'}</div>
      </div>
      <div>
        <div class="text-xs text-gray-400 uppercase tracking-wide font-medium mb-1">Submitted</div>
        <div class="text-gray-600 text-xs">${date}</div>
      </div>
      ${r.description ? `<div class="col-span-2">
        <div class="text-xs text-gray-400 uppercase tracking-wide font-medium mb-1">Description</div>
        <div class="text-gray-700 text-sm leading-relaxed bg-gray-50 rounded-lg p-3">${r.description}</div>
      </div>` : ''}
    </div>

    <div class="border-t border-gray-100 pt-4 mt-2 space-y-3">
      <div>
        <label class="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Update Status</label>
        <div class="flex gap-2">
          <select id="modal-status" class="flex-1 border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400">
            ${['New','Scheduled','In Progress','Resolved'].map(s =>
              `<option ${s === r.status ? 'selected' : ''}>${s}</option>`).join('')}
          </select>
          <button id="modal-status-btn" class="bg-blue-600 hover:bg-blue-700 text-white rounded-lg px-4 py-2 text-sm font-medium transition">
            Save
          </button>
        </div>
      </div>
      <div>
        <label class="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Assign Field Unit</label>
        <div class="flex gap-2">
          <input id="modal-unit" type="text" placeholder="e.g. Unit-7 / PWD Team Alpha"
            class="flex-1 border border-gray-200 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400">
          <button id="modal-assign-btn" class="bg-gray-800 hover:bg-gray-900 text-white rounded-lg px-4 py-2 text-sm font-medium transition">
            Assign
          </button>
        </div>
      </div>
      <div id="modal-feedback" class="text-sm text-green-600 hidden"></div>
    </div>`;

  document.getElementById('modal-status-btn').addEventListener('click', async () => {
    const status = document.getElementById('modal-status').value;
    const btn = document.getElementById('modal-status-btn');
    btn.disabled = true; btn.textContent = 'Saving…';
    try {
      const res = await fetch(`/api/reports/${reportId}/status`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ status })
      });
      const j = await res.json();
      if (j.success) {
        showModalFeedback('Status updated to: ' + status);
        await loadReports(); await loadStats();
      }
    } finally { btn.disabled = false; btn.textContent = 'Save'; }
  });

  document.getElementById('modal-assign-btn').addEventListener('click', async () => {
    const unit = document.getElementById('modal-unit').value.trim();
    if (!unit) return;
    const btn = document.getElementById('modal-assign-btn');
    btn.disabled = true; btn.textContent = 'Assigning…';
    try {
      const res = await fetch(`/api/reports/${reportId}/assign`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ unit })
      });
      const j = await res.json();
      if (j.success) {
        showModalFeedback(`Assigned to "${unit}" — status set to Scheduled.`);
        document.getElementById('modal-unit').value = '';
        await loadReports(); await loadStats();
      }
    } finally { btn.disabled = false; btn.textContent = 'Assign'; }
  });

  modal.classList.remove('hidden');
  modal.classList.add('flex');
}

function closeModal() {
  const modal = document.getElementById('detailModal');
  modal.classList.add('hidden');
  modal.classList.remove('flex');
  currentModalId = null;
}

function showModalFeedback(msg) {
  const el = document.getElementById('modal-feedback');
  el.textContent = '✓ ' + msg;
  el.classList.remove('hidden');
  setTimeout(() => el.classList.add('hidden'), 3500);
}

// ── Init ───────────────────────────────────────────────────
window.openModal  = openModal;
window.closeModal = closeModal;

window.addEventListener('DOMContentLoaded', () => {
  initMap();
  document.getElementById('refresh').addEventListener('click', () => { loadReports(); loadStats(); });
  document.getElementById('filter-severity').addEventListener('change', loadReports);
  document.getElementById('filter-status').addEventListener('change', loadReports);
  document.getElementById('filter-sort').addEventListener('change', loadReports);
  document.getElementById('closeModal').addEventListener('click', closeModal);
  document.getElementById('detailModal').addEventListener('click', e => {
    if (e.target === document.getElementById('detailModal')) closeModal();
  });

  loadStats();
  loadReports();
  setInterval(() => { loadReports(); loadStats(); }, 20000);
});
