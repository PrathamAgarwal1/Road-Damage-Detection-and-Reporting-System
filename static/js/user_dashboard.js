/* ============================================================
   RoadSight User Dashboard — user_dashboard.js
   ============================================================ */

let map, markersLayer;

// ── Colour helpers ─────────────────────────────────────────
function statusColor(s) {
  return { 'New':'#3b82f6', 'Scheduled':'#8b5cf6', 'In Progress':'#f59e0b', 'Resolved':'#22c55e' }[s] || '#6b7280';
}
function priorityColor(p) {
  return p === 'High' ? '#ef4444' : p === 'Medium' ? '#f59e0b' : '#22c55e';
}
function statusClass(s) {
  return { 'New':'bg-blue-100 text-blue-700', 'Scheduled':'bg-purple-100 text-purple-700',
           'In Progress':'bg-amber-100 text-amber-700', 'Resolved':'bg-green-100 text-green-700' }[s] || 'bg-gray-100 text-gray-600';
}
function severityClass(l) {
  return { 'critical':'bg-red-100 text-red-700', 'poor':'bg-orange-100 text-orange-700',
           'moderate':'bg-yellow-100 text-yellow-700', 'good':'bg-green-100 text-green-700' }[l] || 'bg-gray-100 text-gray-600';
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
    `<svg xmlns='http://www.w3.org/2000/svg' width='30' height='30'>` +
    `<circle cx='15' cy='15' r='12' fill='${color}' stroke='white' stroke-width='3'/>` +
    `<circle cx='15' cy='15' r='4' fill='white' opacity='0.9'/></svg>`
  );
  return L.icon({ iconUrl: `data:image/svg+xml,${svg}`, iconSize:[30,30], iconAnchor:[15,15] });
}

// ── Milestones ─────────────────────────────────────────────
async function loadMilestones(reportId) {
  try {
    const res = await fetch(`/api/user/milestones/${reportId}`);
    const json = await res.json();
    if (json.success) return json.milestones || [];
  } catch(e) { console.error('Milestone load failed', e); }
  return [];
}

async function toggleMilestones(reportId) {
  const container = document.getElementById(`milestones-${reportId}`);
  const btn = document.getElementById(`milestone-btn-${reportId}`);
  if (!container) return;

  if (!container.classList.contains('hidden')) {
    container.classList.add('hidden');
    if (btn) btn.innerHTML = '<i class="fa-solid fa-chevron-down mr-1 text-xs"></i> View Updates';
    return;
  }

  container.classList.remove('hidden');
  if (btn) btn.innerHTML = '<i class="fa-solid fa-chevron-up mr-1 text-xs"></i> Hide Updates';
  container.innerHTML = '<div class="text-xs text-gray-400 py-2">Loading updates…</div>';

  const milestones = await loadMilestones(reportId);
  if (milestones.length === 0) {
    container.innerHTML = '<div class="text-xs text-gray-400 py-2 italic">No updates yet — check back soon.</div>';
    return;
  }

  container.innerHTML = `
    <div class="space-y-2 pt-1">
      ${milestones.map(m => `
        <div class="flex gap-3">
          <div class="flex flex-col items-center">
            <div class="w-2.5 h-2.5 rounded-full bg-blue-500 mt-1 flex-shrink-0"></div>
            <div class="flex-1 w-px bg-gray-200 mt-1"></div>
          </div>
          <div class="pb-3 flex-1 min-w-0">
            <div class="text-xs font-semibold text-gray-700">${m.title}</div>
            <div class="text-xs text-gray-500 mt-0.5">${m.description || ''}</div>
            <div class="text-xs text-gray-400 mt-0.5">${new Date(m.createdAt).toLocaleString('en-IN')}</div>
          </div>
        </div>`).join('')}
    </div>`;
}

// ── Render Reports ─────────────────────────────────────────
function renderReports(reports) {
  // Stats
  document.getElementById('total-reports').textContent      = reports.length;
  document.getElementById('resolved-reports').textContent   = reports.filter(r => r.status === 'Resolved').length;
  document.getElementById('in-progress-reports').textContent= reports.filter(r => r.status === 'In Progress').length;
  document.getElementById('new-reports').textContent        = reports.filter(r => r.status === 'New').length;

  // Map
  markersLayer.clearLayers();
  const bounds = [];
  reports.forEach(r => {
    const lat = r.location?.latitude, lng = r.location?.longitude;
    if (lat != null && lng != null) {
      const color = statusColor(r.status);
      const m = L.marker([lat, lng], { icon: markerIcon(color) });
      m.bindPopup(`<div class="text-sm"><strong>${r.location?.address || 'Unknown'}</strong><br>Status: ${r.status}</div>`);
      m.addTo(markersLayer);
      bounds.push([lat, lng]);
    }
  });
  if (bounds.length) map.fitBounds(bounds, { padding: [30, 30] });

  // List
  const list = document.getElementById('report-list');
  list.innerHTML = '';

  if (reports.length === 0) {
    list.innerHTML = `
      <div class="p-8 text-center text-gray-400">
        <i class="fa-solid fa-road text-4xl mb-3 opacity-30"></i>
        <div class="font-medium">No reports yet</div>
        <a href="/" class="text-sm text-blue-500 hover:text-blue-700 mt-1 inline-block">Submit your first report →</a>
      </div>`;
    return;
  }

  reports.forEach(r => {
    const stColor = statusColor(r.status);
    const pColor  = priorityColor(r.priority);
    const stClass = statusClass(r.status);
    const svClass = severityClass(r.severity?.level);
    const date    = r.createdAt ? new Date(r.createdAt).toLocaleDateString('en-IN', { day:'numeric', month:'short', year:'numeric' }) : '—';

    const card = document.createElement('div');
    card.className = 'report-card px-4 py-4 border-b border-gray-50 last:border-0';
    card.innerHTML = `
      <div class="flex gap-3">
        <img src="${r.imageUrl || ''}" class="w-20 h-20 rounded-xl object-cover border border-gray-100 flex-shrink-0"
             onerror="this.style.display='none'">
        <div class="flex-1 min-w-0">
          <div class="flex items-start justify-between gap-2">
            <div class="font-semibold text-gray-800 text-sm truncate">${r.location?.address || 'Unknown location'}</div>
            <span class="text-xs px-2 py-0.5 rounded-full font-medium flex-shrink-0 ${stClass}">${r.status}</span>
          </div>
          <div class="flex flex-wrap items-center gap-2 mt-1">
            <span class="text-xs px-2 py-0.5 rounded-full font-medium ${svClass}">${r.severity?.level || '—'}</span>
            <span class="text-xs font-semibold" style="color:${pColor}">${r.priority} Priority</span>
            <span class="text-xs text-gray-400"><i class="fa-regular fa-calendar mr-1"></i>${date}</span>
          </div>
          ${r.description ? `<div class="text-xs text-gray-500 mt-2 line-clamp-2">${r.description}</div>` : ''}
          <button id="milestone-btn-${r.id}"
            class="mt-2 text-xs text-blue-600 hover:text-blue-800 font-medium transition"
            onclick="window.toggleMilestones('${r.id}')">
            <i class="fa-solid fa-chevron-down mr-1 text-xs"></i> View Updates
          </button>
          <div id="milestones-${r.id}" class="hidden mt-2 pl-1 border-l-2 border-blue-200"></div>
        </div>
      </div>`;
    list.appendChild(card);
  });
}

// ── Load ───────────────────────────────────────────────────
async function loadReports() {
  try {
    const res  = await fetch('/api/user/reports');
    const json = await res.json();
    if (json.success) renderReports(json.reports || []);
    else {
      // Session expired — redirect to login
      if (res.status === 401) window.location.href = '/user/login';
    }
  } catch(e) { console.error('loadReports failed', e); }
}

// Expose for inline onclick
window.toggleMilestones = toggleMilestones;

window.addEventListener('DOMContentLoaded', () => {
  initMap();
  loadReports();
  setInterval(loadReports, 30000);
});
