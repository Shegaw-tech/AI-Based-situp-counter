const debugConsole = document.getElementById('debug-console');
const debugToggle = document.getElementById('debug-toggle');
const startBtn = document.getElementById('start-btn');
const exitBtn = document.getElementById('exit-btn');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const countDisplay = document.getElementById('count');
const feedbackDiv = document.getElementById('feedback');

// Utility functions
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    document.body.appendChild(notification);
    setTimeout(() => notification.remove(), 3000);
}

function stopCamera() {
    if (state.stream) {
        state.stream.getTracks().forEach(track => track.stop());
        state.stream = null;
    }
    state.isRunning = false;
    startBtn.innerHTML = '<span class="btn-icon">▶</span> Start Session';
}

function updateRecommendations(recommendations) {
    const recommendationsList = document.getElementById('recommendations');
    recommendationsList.innerHTML = recommendations
        ?.map(rec => `<li>${rec}</li>`)
        ?.join('') || '';
}

// State management
let state = {
    isRunning: false,
    stream: null,
    sessionId: 'session_' + Date.now(),
    repCount: 0,
    debugMode: true,
    lastGestureTime: 0
};

// Initialize UI
function initUI() {
    debugToggle.checked = state.debugMode;
    debugToggle.addEventListener('change', () => {
        state.debugMode = debugToggle.checked;
        debugConsole.style.display = state.debugMode ? 'block' : 'none';
    });

    if (!debugConsole) {
        const consoleEl = document.createElement('div');
        consoleEl.id = 'debug-console';
        document.body.appendChild(consoleEl);
    }

    debugConsole.style.display = state.debugMode ? 'block' : 'none';
}

// Camera control
async function toggleCamera() {
    if (state.isRunning) {
        stopCamera();
        return;
    }

    try {
        startBtn.disabled = true;
        startBtn.innerHTML = '<span class="btn-icon">⏳</span> Starting...';

        state.stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 1280, height: 720, facingMode: 'user' },
            audio: false
        });

        video.srcObject = state.stream;
        await video.play();

        // Setup canvas
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        state.isRunning = true;
        startBtn.innerHTML = '<span class="btn-icon">⏸</span> Stop Session';
        startBtn.disabled = false;

        // Reset counter
        state.repCount = 0;
        updateDisplay();

        processFrame();
    } catch (err) {
        console.error('Camera error:', err);
        showNotification('Camera access denied. Please enable permissions.', 'error');
        startBtn.innerHTML = '<span class="btn-icon">▶</span> Start Session';
        startBtn.disabled = false;
    }
}

async function processFrame() {
    if (!state.isRunning) return;

    try {
        // Capture frame
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);

        // Process frame
        const response = await fetch('/process_frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Session-ID': state.sessionId
            },
            body: JSON.stringify({
                image: tempCanvas.toDataURL('image/jpeg', 0.85),
                debug: state.debugMode
            })
        });
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        const data = await response.json();
        // Update state
        state.repCount = data.count || 0;
        if (data.debug?.gesture_detected) {
            state.lastGestureTime = Date.now() / 1000;
            showNotification('Count reset!', 'info');
        }
        // Update UI
        updateDisplay(data);
        // Debug information
        if (state.debugMode) {
            updateDebugConsole({
                ...data,
                count: state.repCount
            });
        }
        // Draw processed frame if available
        if (data.image) {
            const img = new Image();
            img.onload = () => {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.src = `data:image/jpeg;base64,${data.image}`;
        }
    } catch (err) {
        console.error('Processing error:', err);
        if (state.debugMode) {
            debugConsole.innerHTML += `<p class="error">Error: ${err.message}</p>`;
        }
        showNotification('Processing error. Please try again.', 'error');
    }

    if (state.isRunning) {
        requestAnimationFrame(processFrame);
    }
}
// UI Updates
function updateDisplay(data) {
    // Update counter
    countDisplay.textContent = state.repCount;
    // Rep completion animation
    if (data?.completed_rep) {
        countDisplay.classList.add('rep-pulse');
        setTimeout(() => countDisplay.classList.remove('rep-pulse'), 500);
    }
    // Form feedback
    if (data?.feedback) {
        feedbackDiv.innerHTML = data.feedback.join('<br>') || '';
        feedbackDiv.style.display = data.feedback.length ? 'block' : 'none';
    }
    // Recommendations
    if (data?.recommendations) {
        updateRecommendations(data.recommendations);
    }
}
function updateDebugConsole(data) {
    const debug = data.debug || {};
    const metrics = [
        `Count: ${data.count || 0}`,
        debug.landmarks_visible ? '✅ Landmarks: Visible' : '❌ Landmarks: Not Visible',
        `Head Lift: ${debug.head_lift?.toFixed(3) ?? 'N/A'}`,
        `Shoulder Lift: ${debug.shoulder_lift?.toFixed(3) ?? 'N/A'}`,
        `Torso Angle: ${debug.torso_angle?.toFixed(1) ?? 'N/A'}°`,
        `Back Angle: ${debug.back_angle?.toFixed(1) ?? 'N/A'}°`,
        `Shoulder Symmetry: ${debug.shoulder_symmetry?.toFixed(3) ?? 'N/A'}`,
        debug.is_in_rep ? '🔄 Status: In Rep' : '💤 Status: Resting'
    ];
    // Add event markers if they exist
    if (debug.rep_start) metrics.push('🚩 REP STARTED');
    if (debug.rep_end) metrics.push('🏁 REP COMPLETED');
    if (debug.gesture_detected) metrics.push('👋 RESET GESTURE DETECTED');

    debugConsole.innerHTML = `
        <div class="debug-header">Debug Information</div>
        ${metrics.map(m => `<div class="debug-metric">${m}</div>`).join('')}
    `;
}
// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    initUI();
    startBtn.addEventListener('click', toggleCamera);
    exitBtn.addEventListener('click', () => {
        if (confirm('End workout session?')) {
            stopCamera();
            window.location.reload();
        }
    });
});