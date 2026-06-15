// --- Cached DOM ---
const uploadForm = document.getElementById('uploadForm');
const analyzeBtn = document.getElementById('analyzeBtn');
const imageFile = document.getElementById('imageFile');
const analysisArea = document.getElementById('analysisArea');
const imagePreview = document.getElementById('imagePreview');
const conditionText = document.getElementById('conditionText');
const confidenceText = document.getElementById('confidenceText');
const reporterName = document.getElementById('reporterName');
const reporterEmail = document.getElementById('reporterEmail');
const descManual = document.getElementById('descManual');
const descGemini = document.getElementById('descGemini');
const reportDescription = document.getElementById('reportDescription');
const regenDescBtn = document.getElementById('regenDescBtn');
const regenSpinner = document.getElementById('regenSpinner');
const locationAddress = document.getElementById('locationAddress');
const getLocationBtn = document.getElementById('getLocationBtn');
const locationText = document.getElementById('locationText');
const submitReportBtn = document.getElementById('submitReportBtn');
const reportResult = document.getElementById('reportResult');
const donateNowBtn = document.getElementById('donateNowBtn');
const reportSection = document.getElementById('reportSection');
const shareSection = document.getElementById('shareSection');
const shareableImage = document.getElementById('shareableImage');
const downloadShareImageBtn = document.getElementById('downloadShareImageBtn');
const shareImageSpinner = document.getElementById('shareImageSpinner');
const regenShareBtn = document.getElementById('regenShareBtn');
const segPreview         = document.getElementById('segPreview');
const roadCoverageRow    = document.getElementById('roadCoverageRow');
const roadCoverageBadge  = document.getElementById('roadCoverageBadge');
const roadWarning        = document.getElementById('roadWarning');
const tabOriginal        = document.getElementById('tabOriginal');
const tabSeg             = document.getElementById('tabSeg');

// --- Image tab switcher ---
function showTab(tab) {
  const isOrig = (tab === 'original');
  document.getElementById('imgOriginal').style.display = isOrig ? 'block' : 'none';
  document.getElementById('imgSeg').style.display      = isOrig ? 'none'  : 'block';
  tabOriginal.className = isOrig ? 'btn btn-sm btn-primary'          : 'btn btn-sm btn-outline-secondary';
  tabSeg.className      = isOrig ? 'btn btn-sm btn-outline-secondary' : 'btn btn-sm btn-primary';
}
window.showTab = showTab;

// state
let lastAnalysis = null;
let userLocation = { latitude: null, longitude: null };
let thankYouModalInstance = null;
let currentUser = null;  // populated after session check

// --- Event Listeners ---
donateNowBtn.addEventListener('click', () => {
  new bootstrap.Modal(document.getElementById('donateModal')).show();
});

document.getElementsByName('descMode').forEach(r => r.addEventListener('change', updateDescModeUI));
uploadForm.addEventListener('submit', handleAnalysis);
imageFile.addEventListener('change', () => handleImageSelect()); // Added reset on new file selection
getLocationBtn.addEventListener('click', acquireLocation);
regenDescBtn.addEventListener('click', generateDescription);
regenShareBtn.addEventListener('click', generateShareableImage);
submitReportBtn.addEventListener('click', handleSubmit);

// --- Functions ---
function resetState() {
    lastAnalysis = null;
    userLocation = { latitude: null, longitude: null };
    
    // Clear all input fields and results
    locationAddress.value = '';
    locationText.textContent = 'Not added';
    reportDescription.value = '';
    reporterName.value = '';
    reporterEmail.value = '';
    reportResult.textContent = '';
    
    // Reset location button
    getLocationBtn.disabled = false;
    setButtonLoading(getLocationBtn, false, '<i class="fa-solid fa-location-crosshairs me-1"></i> Use my precise GPS');
    // Reset segmentation panel
    roadCoverageRow.style.display = 'none';
    roadWarning.style.display = 'none';
    if (segPreview) segPreview.src = '';
    showTab('original');
    // Hide the results area
    analysisArea.style.display = 'none';
}

function handleImageSelect() {
    if (imageFile.files && imageFile.files[0]) {
        resetState(); // Reset everything when a new image is selected
    }
}


function updateDescModeUI() {
  regenDescBtn.style.display = descGemini.checked ? 'inline-block' : 'none';
  reportDescription.readOnly = descGemini.checked;
}

async function handleAnalysis(e) {
  e.preventDefault();
  if (!imageFile.files || !imageFile.files[0]) return alert('Choose an image');
  setButtonLoading(analyzeBtn, true, 'Analyzing...');
  try {
    const fd = new FormData();
    fd.append('image', imageFile.files[0]);
    const res = await fetch('/analyze', { method: 'POST', body: fd });
    const data = await res.json();
    if (!data.success) throw new Error(data.error || 'Analyze failed');
    lastAnalysis = data;
    displayResults(data);
  } catch (err) {
    alert('Error: ' + (err.message || err));
    console.error(err);
  } finally {
    setButtonLoading(analyzeBtn, false, 'Analyze');
  }
}

function displayResults(data) {
    analysisArea.style.display = 'block';
    imagePreview.src = data.image_url;
    conditionText.textContent = data.condition;
    confidenceText.textContent = `${data.confidence}%`;

    // --- Road segmentation panel ---
    roadCoverageRow.style.display = 'block';
    if (data.segmented_url && segPreview) {
        segPreview.src = data.segmented_url;
        tabSeg.style.display = 'inline-block';
    } else {
        tabSeg.style.display = 'none';
    }
    if (data.road_found) {
        const pct = data.road_coverage || 0;
        const cls = pct >= 15 ? 'bg-success' : 'bg-warning text-dark';
        roadCoverageBadge.className = `badge ${cls} ms-1`;
        roadCoverageBadge.textContent = `${pct}% of frame`;
        roadWarning.style.display = 'none';
    } else {
        roadCoverageBadge.className = 'badge bg-danger ms-1';
        roadCoverageBadge.textContent = 'Not detected';
        roadWarning.style.display = 'block';
    }

    shareSection.style.display = 'block';
    regenShareBtn.style.display = 'block';
    generateShareableImage();

    const severity = data.severity.level;
    let colorClass = 'bg-light text-dark';
    if (severity === 'critical' || severity === 'poor') {
        colorClass = 'bg-danger text-white';
        reportSection.style.display = 'block';
    } else {
        if (severity === 'moderate') colorClass = 'bg-warning text-dark';
        else if (severity === 'good') colorClass = 'bg-success text-white';
        reportSection.style.display = 'none';
    }
    conditionText.className = `chip ${colorClass}`;
}


async function acquireLocation() {
  return new Promise((resolve) => {
    if (!navigator.geolocation) {
      locationText.textContent = "Unavailable";
      return resolve();
    }
    setButtonLoading(getLocationBtn, true, 'Getting...');
    navigator.geolocation.getCurrentPosition(pos => {
      userLocation = { latitude: pos.coords.latitude, longitude: pos.coords.longitude };
      locationText.textContent = `Lat:${userLocation.latitude.toFixed(5)}, Lon:${userLocation.longitude.toFixed(5)}`;
      setButtonLoading(getLocationBtn, false, '<i class="fa-solid fa-location-crosshairs me-1"></i> Use my precise GPS');
      resolve();
    }, err => {
      setButtonLoading(getLocationBtn, false, '<i class="fa-solid fa-location-crosshairs me-1"></i> Use my precise GPS');
      locationText.textContent = "Permission denied / unavailable";
      resolve();
    }, { enableHighAccuracy: true, timeout: 8000 });
  });
}

async function generateDescription() {
  if (!lastAnalysis) return;
  setButtonLoading(regenDescBtn, true, '', true);
  try {
    const address = locationAddress.value || `Lat/Lon: ${userLocation.latitude?.toFixed(5)}, ${userLocation.longitude?.toFixed(5)}`;
    const res = await fetch('/generate-description', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ condition: lastAnalysis.condition, address })
    });
    const data = await res.json();
    if (data.success) reportDescription.value = data.description;
  } catch (e) {
    console.warn('Gemini description failed', e);
  } finally {
    setButtonLoading(regenDescBtn, false, 'Generate', true);
  }
}

async function generateShareableImage() {
    if (!lastAnalysis) return;
    shareImageSpinner.style.display = 'block';
    shareableImage.style.display = 'none';
    downloadShareImageBtn.style.display = 'none';
    setButtonLoading(regenShareBtn, true);

    try {
        const address = locationAddress.value || "Nearby Area"; // Use entered address or fallback
        const res = await fetch('/generate-shareable-image', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                condition: lastAnalysis.condition,
                address: address,
                original_filename: lastAnalysis.original_filename
            })
        });
        const data = await res.json();
        if (data.success) {
            shareableImage.src = data.shareable_image_url + '?t=' + new Date().getTime(); // Append timestamp to break cache
            downloadShareImageBtn.href = data.shareable_image_url;
            shareableImage.style.display = 'block';
            downloadShareImageBtn.style.display = 'block';
        } else {
            throw new Error(data.error);
        }
    } catch (e) {
        console.error('Shareable image generation failed', e);
        shareSection.innerHTML += `<p class="text-danger small mt-2">Could not generate shareable image.</p>`;
    } finally {
        shareImageSpinner.style.display = 'none';
        setButtonLoading(regenShareBtn, false);
    }
}


async function handleSubmit() {
  if (!lastAnalysis) return alert('No analysis to report.');
  if (!locationAddress.value) return alert('Please provide the location address before submitting.');

  setButtonLoading(submitReportBtn, true, 'Submitting...');
  try {
    const payload = {
      // Only included for guest users; server ignores these when a session exists
      name:  currentUser ? currentUser.name  : (reporterName.value  || 'Anonymous'),
      email: currentUser ? currentUser.email : (reporterEmail.value || 'anonymous@citizen.local'),
      location: { address: locationAddress.value, ...userLocation },
      condition:       lastAnalysis.condition,
      confidence:      lastAnalysis.confidence,
      image_url:       lastAnalysis.image_url,
      description:     reportDescription.value || '',
      damage_category: 'RoadDamage',
    };
    const res = await fetch('/submit-report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const j = await res.json();
    if (j.success) {
      if (!thankYouModalInstance) {
        thankYouModalInstance = new bootstrap.Modal(document.getElementById('thankYouModal'));
      }
      const dashboardBtn = document.getElementById('viewDashboardBtn');
      dashboardBtn.style.display = currentUser ? 'inline-block' : 'none';
      thankYouModalInstance.show();
      uploadForm.reset();
      analysisArea.style.display = 'none';
    } else {
      throw new Error(j.message || 'Failed to submit');
    }
  } catch (err) {
    reportResult.textContent = `Error: ${err.message}`;
    reportResult.style.color = 'crimson';
  } finally {
    setButtonLoading(submitReportBtn, false, 'Submit Report');
  }
}

function setButtonLoading(button, isLoading, text, isIconBtn = false) {
    button.disabled = isLoading;
    const spinner = button.querySelector('.spinner-border');
    const icon = button.querySelector('i');

    if (isLoading) {
        if (spinner) spinner.classList.remove('visually-hidden');
        if (icon && isIconBtn) icon.classList.add('visually-hidden');
        if (text && !isIconBtn) button.innerHTML = `<span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>${text}`;
    } else {
        if (spinner) spinner.classList.add('visually-hidden');
        if (icon && isIconBtn) icon.classList.remove('visually-hidden');
        if (text) button.innerHTML = text;
    }
}

// Check user session and update navbar
async function checkUserSession() {
  try {
    const res = await fetch('/api/user/check');
    const data = await res.json();
    if (data.success && data.user) {
      currentUser = data.user;
      document.getElementById('loginLink').style.display = 'none';
      document.getElementById('signupLink').style.display = 'none';
      document.getElementById('userInfo').style.display = 'block';
      document.getElementById('userDashboardLink').style.display = 'block';
      document.getElementById('userName').textContent = data.user.name || data.user.email;
      // Hide redundant reporter fields for logged-in users
      const reporterRow = reporterName?.closest('.row');
      if (reporterRow) reporterRow.style.display = 'none';
    } else {
      currentUser = null;
    }
  } catch (e) {
    // User not logged in — keep login/signup visible
    currentUser = null;
  }
}

// Initial UI setup
updateDescModeUI();
checkUserSession();

