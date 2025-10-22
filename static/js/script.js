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

// state
let lastAnalysis = null;
let userLocation = { latitude: null, longitude: null };
let thankYouModalInstance = null;

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
      name: reporterName.value || 'Anonymous',
      email: reporterEmail.value || 'anonymous@citizen.local',
      location: { address: locationAddress.value, ...userLocation },
      condition: lastAnalysis.condition,
      confidence: lastAnalysis.confidence,
      description: reportDescription.value || '',
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

// Initial UI setup
updateDescModeUI();

