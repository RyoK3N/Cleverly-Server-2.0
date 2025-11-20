/**
 * Cleverly Dashboard - Production JavaScript
 * Version: 2.0.0
 */

// ============================================
// State Management
// ============================================
let authToken = localStorage.getItem('authToken');
let currentUser = localStorage.getItem('currentUser');
let sessionExpires = localStorage.getItem('sessionExpires');
let currentSection = 'status';

// ============================================
// Initialization
// ============================================
document.addEventListener('DOMContentLoaded', () => {
    // Check authentication
    if (authToken && sessionExpires && new Date(sessionExpires) > new Date()) {
        showDashboard();
        loadStatus();
        initializeNavigation();
    } else {
        showLogin();
    }
});

// ============================================
// Authentication
// ============================================
document.getElementById('loginForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const btn = document.getElementById('loginBtn');
    const errorEl = document.getElementById('loginError');

    btn.classList.add('loading');
    btn.disabled = true;
    errorEl.style.display = 'none';

    try {
        const response = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                username: document.getElementById('username').value,
                password: document.getElementById('password').value
            })
        });

        const data = await response.json();

        if (response.ok) {
            authToken = data.token;
            currentUser = data.user;
            sessionExpires = data.expires;

            localStorage.setItem('authToken', authToken);
            localStorage.setItem('currentUser', currentUser);
            localStorage.setItem('sessionExpires', sessionExpires);

            showDashboard();
            loadStatus();
            initializeNavigation();
            showToast(`Welcome back, ${currentUser}!`, 'success');
        } else {
            errorEl.textContent = data.error || 'Login failed';
            errorEl.style.display = 'block';
        }
    } catch (err) {
        errorEl.textContent = 'Connection error. Please try again.';
        errorEl.style.display = 'block';
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
});

function showLogin() {
    document.getElementById('loginScreen').style.display = 'flex';
    document.getElementById('dashboardScreen').style.display = 'none';
}

function showDashboard() {
    document.getElementById('loginScreen').style.display = 'none';
    document.getElementById('dashboardScreen').style.display = 'block';

    // Set user info
    document.getElementById('userName').textContent = currentUser || 'User';
    document.getElementById('userAvatar').textContent = (currentUser || 'U').charAt(0).toUpperCase();

    // Set session expiry
    if (sessionExpires) {
        const expDate = new Date(sessionExpires);
        document.getElementById('statSessionExpires').textContent = expDate.toLocaleTimeString();
    }
}

async function logout() {
    try {
        await fetch('/api/auth/logout', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${authToken}` }
        });
    } catch (err) {
        console.error('Logout error:', err);
    }

    localStorage.removeItem('authToken');
    localStorage.removeItem('currentUser');
    localStorage.removeItem('sessionExpires');
    authToken = null;
    currentUser = null;

    showLogin();
    showToast('Logged out successfully', 'info');
}

// ============================================
// Navigation
// ============================================
function initializeNavigation() {
    const navItems = document.querySelectorAll('.nav-item[data-section]');

    navItems.forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const section = item.getAttribute('data-section');
            navigateToSection(section);
        });
    });

    // Handle hash navigation
    if (window.location.hash) {
        const section = window.location.hash.substring(1);
        navigateToSection(section);
    }
}

function navigateToSection(sectionId) {
    // Update nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
        if (item.getAttribute('data-section') === sectionId) {
            item.classList.add('active');
        }
    });

    // Update sections
    document.querySelectorAll('.content-section').forEach(section => {
        section.classList.remove('active');
    });

    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.add('active');
        currentSection = sectionId;
    }

    // Update URL hash
    window.history.pushState(null, '', `#${sectionId}`);
}

// ============================================
// Status & Refresh
// ============================================
async function loadStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        // Update status indicators
        document.getElementById('statApiStatus').textContent = 'Healthy';
        document.getElementById('statLastRun').textContent = data.timestamp || 'Never';

        const cache = data.calendly_cache || {};
        if (cache.last_updated) {
            const date = new Date(cache.last_updated);
            document.getElementById('statCalendlyStatus').textContent = date.toLocaleTimeString();
        } else {
            document.getElementById('statCalendlyStatus').textContent = 'Not loaded';
        }

        // Update API status indicator
        const indicator = document.getElementById('apiStatusIndicator');
        indicator.querySelector('.status-dot').style.background = 'var(--success)';
        indicator.querySelector('.status-text').textContent = 'Connected';

    } catch (err) {
        document.getElementById('statApiStatus').textContent = 'Error';

        const indicator = document.getElementById('apiStatusIndicator');
        indicator.querySelector('.status-dot').style.background = 'var(--error)';
        indicator.querySelector('.status-text').textContent = 'Disconnected';
    }
}

function refreshStatus() {
    showToast('Refreshing status...', 'info');
    loadStatus().then(() => {
        showToast('Status refreshed', 'success');
    });
}

// Auto-refresh status every 30 seconds
setInterval(loadStatus, 30000);

// ============================================
// API Actions
// ============================================
async function executeAction(endpoint, method = 'GET', loadingMessage = 'Loading...', requireAuth = true) {
    showToast(loadingMessage, 'info');
    openLogPanel();

    const timestamp = new Date().toLocaleTimeString();
    addLogEntry(`[${timestamp}] Executing: ${method} /${endpoint}`, 'info');

    try {
        const headers = {
            'Content-Type': 'application/json'
        };

        if (requireAuth) {
            headers['Authorization'] = `Bearer ${authToken}`;
        }

        const response = await fetch(`/${endpoint}`, {
            method: method,
            headers: headers
        });

        if (response.status === 401) {
            addLogEntry('Session expired. Please login again.', 'error');
            showToast('Session expired. Please login again.', 'error');
            logout();
            return;
        }

        const data = await response.json();

        if (response.ok || response.status === 202) {
            addLogEntry(`Response received (${response.status})`, 'success');
            addLogJSON(data);
            showToast('Request completed successfully!', 'success');
            loadStatus();
        } else {
            addLogEntry(`Error: ${data.error || 'Request failed'}`, 'error');
            showToast(data.error || 'Request failed', 'error');
        }

    } catch (err) {
        addLogEntry(`Network error: ${err.message}`, 'error');
        showToast(`Network error: ${err.message}`, 'error');
    }
}

async function runPipeline() {
    executeAction('force_start_pipeline', 'GET', 'Starting data pipeline...');
}

async function downloadFile(endpoint, filename) {
    showToast('Preparing download...', 'info');

    try {
        const response = await fetch(endpoint, {
            headers: {
                'Authorization': `Bearer ${authToken}`
            }
        });

        if (response.status === 401) {
            showToast('Session expired. Please login again.', 'error');
            logout();
            return;
        }

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
            showToast('Download started!', 'success');
        } else {
            const data = await response.json();
            showToast(data.error || 'Download failed', 'error');
        }
    } catch (err) {
        showToast(`Download error: ${err.message}`, 'error');
    }
}

function downloadGroupCSV() {
    const group = document.getElementById('groupSelect').value;
    downloadFile(`/api/group_csv/${group}`, `${group}_data.csv`);
}

async function exportCalendlyData(type) {
    showToast('Preparing export...', 'info');

    try {
        const response = await fetch(`/api/calendly/export/${type}`, {
            headers: {
                'Authorization': `Bearer ${authToken}`
            }
        });

        if (response.status === 401) {
            showToast('Session expired. Please login again.', 'error');
            logout();
            return;
        }

        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `calendly_${type}_${new Date().toISOString().slice(0, 10)}.csv`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            a.remove();
            showToast('Export downloaded!', 'success');
        } else {
            const data = await response.json();
            showToast(data.error || 'Export failed', 'error');
        }
    } catch (err) {
        showToast(`Export error: ${err.message}`, 'error');
    }
}

// ============================================
// Log Panel
// ============================================
function openLogPanel() {
    document.getElementById('logPanel').classList.add('open');
}

function toggleLogPanel() {
    document.getElementById('logPanel').classList.toggle('open');
}

function clearLog() {
    document.getElementById('logContent').innerHTML = `
        <div class="log-placeholder">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="4,17 10,11 4,5"/>
                <line x1="12" y1="19" x2="20" y2="19"/>
            </svg>
            <p>Execute an action to see the response here</p>
        </div>
    `;
}

function addLogEntry(message, type = 'info') {
    const logContent = document.getElementById('logContent');

    // Remove placeholder if present
    const placeholder = logContent.querySelector('.log-placeholder');
    if (placeholder) {
        placeholder.remove();
    }

    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = message;

    logContent.appendChild(entry);
    logContent.scrollTop = logContent.scrollHeight;
}

function addLogJSON(data) {
    const logContent = document.getElementById('logContent');

    const entry = document.createElement('div');
    entry.className = 'log-entry';

    const timestamp = document.createElement('div');
    timestamp.className = 'log-timestamp';
    timestamp.textContent = new Date().toLocaleTimeString();

    const json = document.createElement('pre');
    json.className = 'log-json';
    json.textContent = JSON.stringify(data, null, 2);

    entry.appendChild(timestamp);
    entry.appendChild(json);

    logContent.appendChild(entry);
    logContent.scrollTop = logContent.scrollHeight;
}

function copyLogContent() {
    const logContent = document.getElementById('logContent');
    const text = logContent.innerText;

    navigator.clipboard.writeText(text).then(() => {
        showToast('Log copied to clipboard', 'success');
    }).catch(() => {
        showToast('Failed to copy log', 'error');
    });
}

// ============================================
// Toast Notifications
// ============================================
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    // Auto-remove after 4 seconds
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ============================================
// API Key Management
// ============================================
let newKeyValue = '';

document.getElementById('createKeyForm')?.addEventListener('submit', async (e) => {
    e.preventDefault();

    const name = document.getElementById('keyName').value;
    const permission = document.getElementById('keyPermission').value;
    const expiresIn = document.getElementById('keyExpires').value;
    const rateLimit = document.getElementById('keyRateLimit').value;
    const description = document.getElementById('keyDescription').value;

    try {
        const response = await fetch('/api/keys', {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${authToken}`,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: name,
                permission: permission,
                expires_in_days: expiresIn ? parseInt(expiresIn) : null,
                rate_limit: parseInt(rateLimit),
                description: description
            })
        });

        const data = await response.json();

        if (response.ok) {
            // Show the new key
            newKeyValue = data.key;
            document.getElementById('newKeyValue').textContent = data.key;
            document.getElementById('newKeyDisplay').style.display = 'block';

            // Clear form
            document.getElementById('createKeyForm').reset();
            document.getElementById('keyRateLimit').value = '100';

            // Reload keys list
            loadApiKeys();

            showToast('API key created successfully!', 'success');
        } else {
            showToast(data.error || 'Failed to create key', 'error');
        }
    } catch (err) {
        showToast(`Error: ${err.message}`, 'error');
    }
});

async function loadApiKeys() {
    const container = document.getElementById('apiKeysList');
    if (!container) return;

    container.innerHTML = '<p class="loading-text">Loading keys...</p>';

    try {
        const response = await fetch('/api/keys', {
            headers: {
                'Authorization': `Bearer ${authToken}`
            }
        });

        if (response.status === 401) {
            showToast('Session expired. Please login again.', 'error');
            logout();
            return;
        }

        const data = await response.json();

        if (data.keys && data.keys.length > 0) {
            container.innerHTML = data.keys.map(key => `
                <div class="key-item ${key.is_active ? '' : 'revoked'}">
                    <div class="key-info">
                        <h4>
                            ${key.name}
                            <span class="key-id">${key.key_id}</span>
                            <span class="permission-badge ${key.permission}">${key.permission}</span>
                        </h4>
                        <div class="key-meta">
                            <span>Created: ${formatDate(key.created_at)}</span>
                            <span>Used: ${key.usage_count} times</span>
                            ${key.expires_at ? `<span>Expires: ${formatDate(key.expires_at)}</span>` : ''}
                        </div>
                    </div>
                    <div class="key-actions">
                        ${key.is_active ? `
                            <button class="btn-icon" onclick="revokeKey('${key.key_id}')" title="Revoke Key">
                                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                    <circle cx="12" cy="12" r="10"/>
                                    <line x1="15" y1="9" x2="9" y2="15"/>
                                    <line x1="9" y1="9" x2="15" y2="15"/>
                                </svg>
                            </button>
                        ` : '<span style="color: var(--error); font-size: 12px;">Revoked</span>'}
                    </div>
                </div>
            `).join('');
        } else {
            container.innerHTML = `
                <div class="empty-state">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 1 1-7.778 7.778 5.5 5.5 0 0 1 7.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"/>
                    </svg>
                    <p>No API keys yet. Create one above to get started.</p>
                </div>
            `;
        }
    } catch (err) {
        container.innerHTML = `<p class="loading-text">Error loading keys: ${err.message}</p>`;
    }
}

async function revokeKey(keyId) {
    if (!confirm('Are you sure you want to revoke this API key? This action cannot be undone.')) {
        return;
    }

    try {
        const response = await fetch(`/api/keys/${keyId}`, {
            method: 'DELETE',
            headers: {
                'Authorization': `Bearer ${authToken}`
            }
        });

        if (response.ok) {
            showToast('API key revoked', 'success');
            loadApiKeys();
        } else {
            const data = await response.json();
            showToast(data.error || 'Failed to revoke key', 'error');
        }
    } catch (err) {
        showToast(`Error: ${err.message}`, 'error');
    }
}

function copyNewKey() {
    if (newKeyValue) {
        navigator.clipboard.writeText(newKeyValue).then(() => {
            showToast('API key copied to clipboard', 'success');
        }).catch(() => {
            showToast('Failed to copy key', 'error');
        });
    }
}

function showExampleTab(tab) {
    // Update tabs
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.textContent.toLowerCase() === tab) {
            btn.classList.add('active');
        }
    });

    // Update content
    document.querySelectorAll('.example-content').forEach(content => {
        content.classList.remove('active');
    });

    const targetContent = document.getElementById(`${tab}Example`);
    if (targetContent) {
        targetContent.classList.add('active');
    }
}

// Load API keys when navigating to the section
const originalNavigate = navigateToSection;
navigateToSection = function(sectionId) {
    originalNavigate(sectionId);
    if (sectionId === 'api-keys') {
        loadApiKeys();
    }
};

// ============================================
// Utility Functions
// ============================================
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString) {
    if (!dateString) return '-';
    const date = new Date(dateString);
    return date.toLocaleString();
}

// ============================================
// Keyboard Shortcuts
// ============================================
document.addEventListener('keydown', (e) => {
    // Escape to close log panel
    if (e.key === 'Escape') {
        document.getElementById('logPanel').classList.remove('open');
    }

    // Ctrl+L to toggle log panel
    if (e.ctrlKey && e.key === 'l') {
        e.preventDefault();
        toggleLogPanel();
    }
});

// ============================================
// Window Events
// ============================================
window.addEventListener('hashchange', () => {
    if (window.location.hash) {
        const section = window.location.hash.substring(1);
        navigateToSection(section);
    }
});

// Warn before leaving with unsaved actions
window.addEventListener('beforeunload', (e) => {
    // Only warn if there are pending actions
    // Currently not implemented
});
