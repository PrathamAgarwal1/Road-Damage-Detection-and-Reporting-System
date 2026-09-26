/* ============================================================
   RoadSight public live map — map.js
   ============================================================ */
const SEVERITY_COLORS = { critical: '#ef4444', poor: '#f97316', moderate: '#f59e0b', good: '#10b981' };
const STATUS_LABELS = { 'New': 'Reported', 'Scheduled': 'Repair scheduled', 'In Progress': 'Repair in progress', 'Resolved': 'Fixed' };

let map, cluster, allReports = [], fitted = false;

function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[c]));
}
function safeUrl(u) {
  return typeof u === 'string' && u.startsWith('/static/') ? esc(u) : '';
}

function markerIcon(color, resolved) {
  const svg = encodeURIComponent(
    `<svg xmlns='http://www.w3.org/2000/svg' width='26' height='26'>` +
    `<circle cx='13' cy='13' r='10' fill='${resolved ? '#10b981' : color}' stroke='white' stroke-width='3'/>` +
    (resolved ? `<path d='M8.5 13.5l3 3 6-6.5' stroke='white' stroke-width='2.4' fill='none' stroke-linecap='round'/>` : '') +
    `</svg>`);
  return L.icon({ iconUrl: `data:image/svg+xml,${svg}`, iconSize: [26, 26], iconAnchor: [13, 13] });
}

function popupHtml(r) {
  const date = r.createdAt ? new Date(r.createdAt).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' }) : '';
  const img = safeUrl(r.imageUrl);
  return `<div class="rs-popup" style="width:220px">
    ${img ? `<img src="${img}" alt="Road photo" loading="lazy" onerror="this.remove()">` : ''}
    <div class="fw-semibold small">${esc(r.address || 'Unknown location')}</div>
    <div class="small mt-1"><span style="color:${SEVERITY_COLORS[r.severity] || '#6b7280'};font-weight:700">${esc(r.condition || '')}</span>
      · ${esc(STATUS_LABELS[r.status] || r.status)}</div>
    ${r.roadHealthIndex != null && r.status !== 'Resolved' ? `<div class="small">Road health: <strong>${r.roadHealthIndex}/100</strong></div>` : ''}
    ${r.forecast && r.status !== 'Resolved' ? `<div class="small text-muted">${esc(r.forecast)}</div>` : ''}
    ${r.confirmations ? `<div class="small text-primary"><i class="fa-solid fa-users me-1"></i>Confirmed by ${r.confirmations} more citizen${r.confirmations > 1 ? 's' : ''}</div>` : ''}
    <div class="small text-muted mt-1">Reported ${date}</div>
  </div>`;
}

function render() {
  const status = document.getElementById('fStatus').value;
  const severity = document.getElementById('fSeverity').value;
  const shown = allReports.filter(r =>
    (status === '' || (status === 'resolved' ? r.status === 'Resolved' : r.status !== 'Resolved')) &&
    (severity === '' || r.severity === severity));
  cluster.clearLayers();
  const markers = shown.map(r => L.marker([r.latitude, r.longitude], { icon: markerIcon(SEVERITY_COLORS[r.severity] || '#6b7280', r.status === 'Resolved') })
    .bindPopup(popupHtml(r)));
  cluster.addLayers(markers);
  if (!fitted && markers.length) {
    map.fitBounds(L.featureGroup(markers).getBounds(), { padding: [40, 40], maxZoom: 14 });
    fitted = true;
  }
}

async function load() {
  try {
    const d = await (await fetch('/api/public/reports')).json();
    if (!d.success) return;
    allReports = d.reports.filter(r => typeof r.latitude === 'number' && typeof r.longitude === 'number');
    document.getElementById('cOpen').textContent = allReports.filter(r => r.status === 'New').length;
    document.getElementById('cProgress').textContent = allReports.filter(r => ['Scheduled', 'In Progress'].includes(r.status)).length;
    document.getElementById('cResolved').textContent = allReports.filter(r => r.status === 'Resolved').length;
    render();
  } catch (e) { console.error('Failed to load reports', e); }
}

map = L.map('map', { zoomControl: true }).setView([22.5, 79], 5);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; OpenStreetMap' }).addTo(map);
cluster = L.markerClusterGroup({ showCoverageOnHover: false, maxClusterRadius: 50 });
map.addLayer(cluster);

document.getElementById('fStatus').addEventListener('change', render);
document.getElementById('fSeverity').addEventListener('change', render);
document.getElementById('locateBtn').addEventListener('click', () => {
  if (!navigator.geolocation) return;
  navigator.geolocation.getCurrentPosition(pos => {
    const ll = [pos.coords.latitude, pos.coords.longitude];
    map.setView(ll, 14);
    L.circleMarker(ll, { radius: 7, color: '#2563eb', fillColor: '#3b82f6', fillOpacity: .9 }).addTo(map).bindPopup('You are here').openPopup();
  });
});

load();
setInterval(load, 60000);
