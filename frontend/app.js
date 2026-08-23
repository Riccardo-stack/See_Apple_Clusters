// ─── API Configuration ───────────────────────────────────────
// When running locally, leave this as an empty string "".
// For Vercel deployment, set this to your Tailscale Funnel URL
// e.g. "https://your-machine-name.tail1234.ts.net"
// (no trailing slash)
const API_BASE_URL = "https://macbook-air-di-riccardo.tail22ee74.ts.net/";
// ─────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const heroSection = document.getElementById('hero');
  const aboutSection = document.getElementById('about');
  const resultsSection = document.getElementById('results');
  const resultImg = document.getElementById('result-img');
  const statTime = document.getElementById('stat-time');
  const statCount = document.getElementById('stat-count');
  const statConfidence = document.getElementById('stat-confidence');
  const resetBtn = document.getElementById('reset-btn');
  const loadingOverlay = document.getElementById('loading');
  const errorToast = document.getElementById('error-toast');
  const errorMessage = document.getElementById('error-message');

  let currentImageUrl = null;
  let errorTimeout = null;

  // --- 1. File Upload (click + drag-and-drop) ---

  // Click to open file dialog
  dropzone.addEventListener('click', () => fileInput.click());

  // File selected via dialog
  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFile(e.target.files[0]);
    }
  });

  // Drag and drop
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  });

  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('drag-over');
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  // --- Reset Button ---
  resetBtn.addEventListener('click', () => {
    // Show hero, hide results
    heroSection.classList.remove('hidden');
    aboutSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    // Clear file input
    fileInput.value = '';

    // Revoke old object URL to prevent memory leaks
    if (currentImageUrl) {
      URL.revokeObjectURL(currentImageUrl);
      currentImageUrl = null;
    }

    // Clear image src
    resultImg.src = '';
  });

  // --- 2. handleFile(file) ---
  async function handleFile(file) {
    // Validate that file is an image
    if (!file.type.startsWith('image/')) {
      showError('Please upload a valid image file.');
      return;
    }

    // Show loading overlay
    showLoading();

    try {
      // Call the API
      const result = await callDetectAPI(file);

      // On success: hide loading, show results
      hideLoading();
      displayResults(result);
    } catch (error) {
      // On error: hide loading, show error toast
      hideLoading();
      showError(error.message || 'An unexpected error occurred.');
    }
  }

  // --- 3. API Call ---
  async function callDetectAPI(file) {
    const formData = new FormData();
    formData.append('file', file);

    let response;
    try {
      const base = API_BASE_URL.replace(/\/+$/, '');
      response = await fetch(`${base}/detect?confidence=0.75`, {
        method: 'POST',
        body: formData,
      });
    } catch (networkError) {
      throw new Error('Network error. Please check your connection and try again.');
    }

    if (!response.ok) {
      // Try to parse error JSON
      let errorMsg = `Server error: ${response.status} ${response.statusText}`;
      try {
        const errorData = await response.json();
        if (errorData && errorData.error) {
          errorMsg = errorData.error;
        }
      } catch (e) {
        // Ignore JSON parse error, use default message
      }
      throw new Error(errorMsg);
    }

    // Read custom headers
    const inferenceMs = parseFloat(response.headers.get('X-Inference-Time-Ms'));
    const detectionsCount = parseInt(response.headers.get('X-Detections-Count'));
    const confidence = parseFloat(response.headers.get('X-Confidence-Threshold'));

    // Read image blob
    const blob = await response.blob();
    const imageUrl = URL.createObjectURL(blob);

    return { imageUrl, inferenceMs, detectionsCount, confidence };
  }

  // --- 4. Show Results ---
  function displayResults({ imageUrl, inferenceMs, detectionsCount, confidence }) {
    // Store image URL so we can revoke it later
    currentImageUrl = imageUrl;

    // Set image source
    resultImg.src = imageUrl;

    // Format and set stats
    if (!isNaN(inferenceMs)) {
      statTime.textContent = `${(inferenceMs / 1000).toFixed(2)} sec`;
    } else {
      statTime.textContent = 'N/A';
    }

    if (!isNaN(detectionsCount)) {
      statCount.textContent = `${detectionsCount} cluster${detectionsCount === 1 ? '' : 's'} detected`;
    } else {
      statCount.textContent = 'N/A';
    }

    if (!isNaN(confidence)) {
      statConfidence.textContent = `${Math.round(confidence * 100)}%`;
    } else {
      statConfidence.textContent = 'N/A';
    }

    // Toggle visibility
    heroSection.classList.add('hidden');
    aboutSection.classList.add('hidden');
    resultsSection.classList.remove('hidden');

    // Smooth scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
  }

  // --- UI Helpers ---
  function showLoading() {
    loadingOverlay.classList.remove('hidden');
  }

  function hideLoading() {
    loadingOverlay.classList.add('hidden');
  }

  function showError(message) {
    errorMessage.textContent = message;
    errorToast.classList.add('visible');

    // Clear any existing timeout
    if (errorTimeout) {
      clearTimeout(errorTimeout);
    }

    // Auto-hide after 4 seconds
    errorTimeout = setTimeout(() => {
      errorToast.classList.remove('visible');
    }, 4000);
  }
});
