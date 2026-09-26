/* ============================================================
   RoadSight citizen page — script.js
   ============================================================ */
const $ = id => document.getElementById(id);

const MAX_BYTES = 15 * 1024 * 1024;
const CLASS_COLORS = { 'Good': '#10b981', 'Satisfactory': '#f59e0b', 'Poor': '#f97316', 'Very Poor': '#ef4444' };
const CLASS_ORDER = ['Good', 'Satisfactory', 'Poor', 'Very Poor'];
const GPS_BTN_HTML = '<i class="fa-solid fa-location-crosshairs me-1"></i>Use GPS';

// state
let selectedFile = null;
let lastAnalysis = null;
let lastPostText = '';
let userLocation = { latitude: null, longitude: null };
let currentUser = null;
let thankYouModal = null;
let progressTimer = null;

// ── Helpers ────────────────────────────────────────────────
function setStep(n) {
  [1, 2, 3].forEach(i => {
    const el = $(`step${i}`);
    el.classList.toggle('active', i === n);
    el.classList.toggle('done', i < n);
  });
}

function setButtonLoading(button, isLoading, html) {
  button.disabled = isLoading;
  const spinner = button.querySelector('.spinner-border');
  const icon = button.querySelector('i');
  if (spinner) {  // icon buttons with a built-in spinner
    spinner.classList.toggle('visually-hidden', !isLoading);
    if (icon) icon.classList.toggle('visually-hidden', isLoading);
    return;
  }
  button.innerHTML = isLoading
    ? `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>${html}`
    : html;
}

function healthColor(h) {
  return h >= 60 ? '#10b981' : h >= 35 ? '#f59e0b' : '#ef4444';
}

async function postJson(url, body) {
  const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  let data;
  try { data = await res.json(); } catch { data = { success: false, error: `Server error (${res.status})` }; }
  return data;
}

// ── File selection (click, camera, drag & drop) ─────────────
function pickFile(file) {
  $('analyzeError').style.display = 'none';
  if (!file) return;
  if (!/^image\/(jpeg|png|webp)$/.test(file.type) && !/\.(jpe?g|png|webp)$/i.test(file.name)) {
    return showAnalyzeError('Please choose a JPG, PNG or WEBP image.');
  }
  if (file.size > MAX_BYTES) return showAnalyzeError('That image is larger than 15 MB. Please choose a smaller one.');
  selectedFile = file;
  resetResults();
  $('localPreview').src = URL.createObjectURL(file);
  $('fileLabel').textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
  $('dropPrompt').style.display = 'none';
  $('dropPreview').style.display = 'block';
  $('analyzeBtn').disabled = false;
  setStep(1);
}

function showAnalyzeError(msg) {
  const el = $('analyzeError');
  el.textContent = msg;
  el.style.display = 'block';
}

function resetResults() {
  lastAnalysis = null;
  userLocation = { latitude: null, longitude: null };
  $('analysisArea').style.display = 'none';
  $('reportSection').style.display = 'none';
  $('reportAnywayRow').style.display = 'none';
  $('locationAddress').value = '';
  $('locationText').textContent = 'No GPS position yet. Type an address or use GPS.';
  $('reportDescription').value = '';
  $('reportResult').textContent = '';
  $('shareError').style.display = 'none';
  showTab('original');
}

function resetAll() {
  selectedFile = null;
  $('uploadForm').reset();
  $('dropPrompt').style.display = 'block';
  $('dropPreview').style.display = 'none';
  $('analyzeBtn').disabled = true;
  resetResults();
  setStep(1);
}

const dz = $('dropzone');
dz.addEventListener('click', e => { if (!e.target.closest('button,a')) $('imageFile').click(); });
dz.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); $('imageFile').click(); } });
['dragenter', 'dragover'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('dragover'); }));
['dragleave', 'drop'].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('dragover'); }));
dz.addEventListener('drop', e => pickFile(e.dataTransfer.files[0]));
$('chooseBtn').addEventListener('click', () => $('imageFile').click());
$('cameraBtn').addEventListener('click', () => $('cameraFile').click());
$('imageFile').addEventListener('change', e => pickFile(e.target.files[0]));
$('cameraFile').addEventListener('change', e => pickFile(e.target.files[0]));
$('changeFile').addEventListener('click', e => { e.preventDefault(); $('imageFile').click(); });

// ── Analysis ───────────────────────────────────────────────
function startProgress() {
  const stages = ['Uploading photo…', 'Checking that this is a road…', 'Detecting the road surface…', 'Rating the road condition…'];
  let i = 0;
  $('progressText').textContent = stages[0];
  $('analyzeProgress').style.display = 'block';
  progressTimer = setInterval(() => { i = Math.min(i + 1, stages.length - 1); $('progressText').textContent = stages[i]; }, 1600);
}
function stopProgress() {
  clearInterval(progressTimer);
  $('analyzeProgress').style.display = 'none';
}

$('uploadForm').addEventListener('submit', async e => {
  e.preventDefault();
  if (!selectedFile) return;
  $('analyzeError').style.display = 'none';
  setStep(2);
  setButtonLoading($('analyzeBtn'), true, 'Analyzing…');
  startProgress();
  try {
    const fd = new FormData();
    fd.append('image', selectedFile);
    const res = await fetch('/analyze', { method: 'POST', body: fd });
    let data;
    try { data = await res.json(); } catch { data = { success: false, error: `Server error (${res.status})` }; }
    if (!data.success) throw new Error(data.error || 'Analysis failed');
    lastAnalysis = data;
    displayResults(data);
  } catch (err) {
    setStep(1);
    showAnalyzeError(err.message || String(err));
  } finally {
    stopProgress();
    setButtonLoading($('analyzeBtn'), false, '<i class="fa-solid fa-wand-magic-sparkles me-2"></i>Analyze road');
  }
});

function showTab(tab) {
  const isOrig = tab === 'original';
  $('imgOriginal').style.display = isOrig ? 'block' : 'none';
  $('imgSeg').style.display = isOrig ? 'none' : 'block';
  $('tabOriginal').className = `btn ${isOrig ? 'btn-primary' : 'btn-outline-secondary'}`;
  $('tabSeg').className = `btn ${isOrig ? 'btn-outline-secondary' : 'btn-primary'}`;
}
$('tabOriginal').addEventListener('click', () => showTab('original'));
$('tabSeg').addEventListener('click', () => showTab('seg'));

function renderGauge(value) {
  const arc = $('gaugeArc');
  arc.style.stroke = healthColor(value);
  arc.setAttribute('stroke-dasharray', `${Math.max(0, Math.min(100, value)) * 0.97},100`);
  $('healthValue').textContent = value;
  $('healthValue').style.color = healthColor(value);
}

function renderProbabilities(probs) {
  const bars = $('probBars');
  bars.innerHTML = '';
  CLASS_ORDER.filter(c => c in probs).forEach(c => {
    const row = document.createElement('div');
    row.className = 'rs-prob';
    row.innerHTML = `<span>${c}</span><div class="bar"><div style="width:0;background:${CLASS_COLORS[c]}"></div></div><span class="pct">${probs[c].toFixed(1)}%</span>`;
    bars.appendChild(row);
    requestAnimationFrame(() => { row.querySelector('.bar > div').style.width = `${probs[c]}%`; });
  });
}

function displayResults(data) {
  $('analysisArea').style.display = 'block';
  $('imagePreview').src = data.image_url;
  const color = CLASS_COLORS[data.condition] || '#6b7280';
  const cond = $('conditionText');
  cond.textContent = data.condition;
  cond.style.background = color + '22';
  cond.style.color = color;
  $('confidenceText').textContent = `${data.confidence}%`;
  $('uncertainNote').style.display = data.uncertain ? 'block' : 'none';
  renderGauge(data.road_health_index ?? 0);
  renderProbabilities(data.probabilities || {});

  if (data.segmented_url) {
    $('segPreview').src = data.segmented_url;
    $('tabSeg').style.display = '';
  } else {
    $('tabSeg').style.display = 'none';
  }
  $('roadCoverageRow').innerHTML = data.road_found
    ? `<i class="fa-solid fa-road me-1"></i>The road surface fills <strong>${data.road_coverage}%</strong> of the photo`
    : '';
  $('roadWarning').style.display = data.road_found ? 'none' : 'block';
  $('roadUncertain').style.display = data.road_check?.uncertain ? 'block' : 'none';

  // Automatic geotagging from the photo's EXIF GPS metadata
  if (data.exif_location && data.exif_location.latitude != null) {
    setLocation(data.exif_location.latitude, data.exif_location.longitude, 'from photo');
  }

  const damaged = ['critical', 'poor'].includes(data.severity.level);
  $('reportSection').style.display = damaged ? 'block' : 'none';
  $('reportAnywayRow').style.display = damaged ? 'none' : 'block';
  setStep(damaged ? 3 : 2);

  $('shareSection').style.display = 'block';
  generateShareableImage();
  $('analysisArea').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

$('reportAnywayBtn').addEventListener('click', () => {
  $('reportAnywayRow').style.display = 'none';
  $('reportSection').style.display = 'block';
  setStep(3);
});

// ── Location ───────────────────────────────────────────────
async function setLocation(latitude, longitude, source) {
  userLocation = { latitude, longitude };
  $('locationText').textContent = `${latitude.toFixed(5)}, ${longitude.toFixed(5)} (${source})`;
  if ($('locationAddress').value.trim()) return;
  try {
    const res = await fetch(`/api/geocode/reverse?lat=${latitude}&lon=${longitude}`);
    const data = await res.json();
    if (data.success && data.address && !$('locationAddress').value.trim()) $('locationAddress').value = data.address;
  } catch (e) {
    console.warn('Reverse geocoding failed', e);
  }
}

$('getLocationBtn').addEventListener('click', () => {
  const btn = $('getLocationBtn');
  if (!navigator.geolocation) { $('locationText').textContent = 'GPS is not available in this browser.'; return; }
  setButtonLoading(btn, true, 'Locating…');
  navigator.geolocation.getCurrentPosition(async pos => {
    setButtonLoading(btn, false, GPS_BTN_HTML);
    await setLocation(pos.coords.latitude, pos.coords.longitude, 'device GPS');
  }, () => {
    setButtonLoading(btn, false, GPS_BTN_HTML);
    $('locationText').textContent = 'Location permission denied or unavailable. Please type the address.';
  }, { enableHighAccuracy: true, timeout: 10000 });
});

// ── Description & share post ───────────────────────────────
$('regenDescBtn').addEventListener('click', async () => {
  if (!lastAnalysis) return;
  const btn = $('regenDescBtn');
  setButtonLoading(btn, true);
  try {
    const address = $('locationAddress').value.trim() ||
      (userLocation.latitude != null ? `${userLocation.latitude.toFixed(5)}, ${userLocation.longitude.toFixed(5)}` : '');
    const data = await postJson('/generate-description', { condition: lastAnalysis.condition, address });
    if (data.success) $('reportDescription').value = data.description;
  } catch (e) {
    console.warn('Description generation failed', e);
  } finally {
    setButtonLoading(btn, false);
  }
});

async function generateShareableImage() {
  if (!lastAnalysis) return;
  $('shareImageSpinner').style.display = 'inline-block';
  $('shareError').style.display = 'none';
  $('shareableImage').style.display = 'none';
  $('downloadShareImageBtn').style.display = 'none';
  setButtonLoading($('regenShareBtn'), true);
  try {
    const data = await postJson('/generate-shareable-image', {
      condition: lastAnalysis.condition,
      address: $('locationAddress').value.trim() || 'Nearby Area',
      original_filename: lastAnalysis.original_filename,
    });
    if (!data.success) throw new Error(data.error);
    $('shareableImage').src = `${data.shareable_image_url}?t=${Date.now()}`;
    $('downloadShareImageBtn').href = data.shareable_image_url;
    $('shareableImage').style.display = 'inline-block';
    $('downloadShareImageBtn').style.display = 'inline-block';
    lastPostText = data.post_text || '';
    $('sharePostText').textContent = lastPostText;
    const shareUrl = `${location.origin}/map`;
    $('shareXBtn').href = `https://twitter.com/intent/tweet?text=${encodeURIComponent(lastPostText)}&url=${encodeURIComponent(shareUrl)}`;
    $('shareWaBtn').href = `https://wa.me/?text=${encodeURIComponent(lastPostText + ' ' + shareUrl)}`;
  } catch (e) {
    console.error('Shareable image generation failed', e);
    $('shareError').style.display = 'block';
  } finally {
    $('shareImageSpinner').style.display = 'none';
    setButtonLoading($('regenShareBtn'), false);
  }
}
$('regenShareBtn').addEventListener('click', generateShareableImage);
$('copyPostBtn').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(lastPostText);
    $('copyPostBtn').innerHTML = '<i class="fa-solid fa-check me-1"></i>Copied';
    setTimeout(() => { $('copyPostBtn').innerHTML = '<i class="fa-regular fa-copy me-1"></i>Copy text'; }, 1800);
  } catch { /* clipboard unavailable */ }
});

// ── Submit ─────────────────────────────────────────────────
$('submitReportBtn').addEventListener('click', async () => {
  const result = $('reportResult');
  result.textContent = '';
  if (!lastAnalysis) return;
  if (!$('locationAddress').value.trim() && userLocation.latitude == null) {
    result.className = 'mt-2 small text-danger';
    result.textContent = 'Please enter the location address (or use GPS) before submitting.';
    $('locationAddress').focus();
    return;
  }
  const btn = $('submitReportBtn');
  setButtonLoading(btn, true, 'Submitting…');
  try {
    const j = await postJson('/submit-report', {
      name: currentUser ? currentUser.name : ($('reporterName').value.trim() || 'Anonymous'),
      email: currentUser ? currentUser.email : $('reporterEmail').value.trim(),
      location: { address: $('locationAddress').value.trim(), ...userLocation },
      image_url: lastAnalysis.image_url,
      captured_at: lastAnalysis.captured_at || null,
      description: $('reportDescription').value.trim(),
    });
    if (!j.success) {
      result.className = `mt-2 small ${j.duplicate ? 'text-warning-emphasis' : 'text-danger'}`;
      result.innerHTML = `<i class="fa-solid ${j.duplicate ? 'fa-clone' : 'fa-circle-exclamation'} me-1"></i>`;
      result.append(j.error || 'Failed to submit');
      return;
    }
    const f = j.forecast || {};
    const w = j.weather || {};
    $('submitPriority').textContent = j.priority || '—';
    $('submitHealth').textContent = f.roadHealthIndex != null ? `${f.roadHealthIndex}/100 (${f.roadHealthIndex30d}/100 in 30 days)` : '—';
    $('submitForecast').textContent = f.summary ? `${f.summary}.` : '';
    $('submitWeather').textContent = w.available
      ? `Weather: ${w.pastRainMm} mm of rain in the past week, ${w.forecastRainMm} mm forecast for the next week.` : '';
    $('viewDashboardBtn').style.display = currentUser ? 'inline-block' : 'none';
    thankYouModal = thankYouModal || new bootstrap.Modal($('thankYouModal'));
    thankYouModal.show();
    resetAll();
    loadPublicStats();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  } catch (err) {
    result.className = 'mt-2 small text-danger';
    result.textContent = err.message || 'Failed to submit';
  } finally {
    setButtonLoading(btn, false, '<i class="fa-solid fa-paper-plane me-2"></i>Submit report');
  }
});

// ── Session & stats ────────────────────────────────────────
async function checkUserSession() {
  try {
    const data = await (await fetch('/api/user/check')).json();
    if (data.success && data.user) {
      currentUser = data.user;
      $('loginLink').style.display = 'none';
      $('signupLink').style.display = 'none';
      $('userInfo').style.display = 'block';
      $('userDashboardLink').style.display = 'block';
      $('userName').textContent = data.user.name || data.user.email;
      $('guestFields').style.display = 'none';
    }
  } catch { currentUser = null; }
}

async function loadPublicStats() {
  try {
    const d = await (await fetch('/api/public/stats')).json();
    if (!d.success) return;
    $('statTotal').textContent = d.total.toLocaleString();
    $('statResolved').textContent = d.resolved.toLocaleString();
    $('statProgress').textContent = d.in_progress.toLocaleString();
    $('statDays').textContent = d.avg_resolution_days ?? '—';
  } catch { /* counters stay as dashes */ }
}

$('donateNowBtn').addEventListener('click', () => new bootstrap.Modal($('donateModal')).show());

checkUserSession();
loadPublicStats();
