/**
 * iSpot Frontend Application
 *
 * Single-page app for the spatial transcriptomics clustering benchmark platform.
 * Communicates with the FastAPI backend via REST.
 */
(function() {
  'use strict';

  // API base URL — same origin in production
  const API = window.location.origin + '/api';

  // State
  let state = {
    jobId: null,
    platform: null,
    methods: [],
    availableMethods: [],
    pollTimer: null,
    viewerData: null,
    viewerCanvas: null,
    viewerCtx: null,
    viewerMethod: null,
    viewerShowGT: false,
    viewerScale: 1,
    viewerOffsetX: 0,
    viewerOffsetY: 0,
  };

  // ---------------------------------------------------------------------------
  // API helpers
  // ---------------------------------------------------------------------------

  async function apiGet(path) {
    const resp = await fetch(API + path);
    if (!resp.ok) throw new Error(`API error ${resp.status}: ${await resp.text()}`);
    return resp.json();
  }

  async function apiPost(path, body) {
    const resp = await fetch(API + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(`API error ${resp.status}: ${await resp.text()}`);
    return resp.json();
  }

  async function apiUpload(file, formData) {
    const form = new FormData();
    form.append('file', file);
    for (const [k, v] of Object.entries(formData)) {
      if (v) form.append(k, v);
    }
    const resp = await fetch(API + '/upload', { method: 'POST', body: form });
    if (!resp.ok) throw new Error(`Upload error ${resp.status}: ${await resp.text()}`);
    return resp.json();
  }

  // ---------------------------------------------------------------------------
  // Initialization
  // ---------------------------------------------------------------------------

  async function init() {
    try {
      const health = await apiGet('/health');
      document.getElementById('health-methods').textContent = health.methods_available;
      document.getElementById('health-runs').textContent = health.meta_learning_runs;
      document.getElementById('health-platforms').textContent = health.platforms_supported.length;

      const methodsResp = await apiGet('/methods');
      state.availableMethods = methodsResp.methods;
      renderMethodSelection();

      const platformsResp = await apiGet('/platforms');
      renderPlatformSelect(platformsResp.platforms);
    } catch (e) {
      showAlert('Failed to connect to API server. Is the backend running?', 'error');
    }
  }

  // ---------------------------------------------------------------------------
  // Upload
  // ---------------------------------------------------------------------------

  function setupUpload() {
    const zone = document.getElementById('upload-zone');
    const input = document.getElementById('file-input');

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) handleFile(e.dataTransfer.files[0]);
    });
    input.addEventListener('change', () => {
      if (input.files.length > 0) handleFile(input.files[0]);
    });
  }

  async function handleFile(file) {
    document.getElementById('upload-filename').textContent = file.name;
    document.getElementById('upload-filesize').textContent = formatSize(file.size);
    document.getElementById('upload-info').classList.remove('hidden');

    const platform = document.getElementById('select-platform').value || null;
    const sampleId = document.getElementById('input-sample-id').value || null;
    const gtCol = document.getElementById('input-gt-col').value || null;

    try {
      showAlert('Uploading file...', 'info');
      const result = await apiUpload(file, { platform, sample_id: sampleId, ground_truth_col: gtCol });
      state.jobId = result.job_id;
      state.platform = result.platform;
      showAlert(`Upload successful! Platform: ${result.platform}. Job ID: ${result.job_id}`, 'success');
      document.getElementById('btn-start').disabled = false;
      document.getElementById('upload-platform').textContent = result.platform;
    } catch (e) {
      showAlert(`Upload failed: ${e.message}`, 'error');
    }
  }

  // ---------------------------------------------------------------------------
  // Method selection
  // ---------------------------------------------------------------------------

  function renderMethodSelection() {
    const container = document.getElementById('method-grid');
    container.innerHTML = '';
    for (const m of state.availableMethods) {
      const chip = document.createElement('label');
      chip.className = 'method-chip';
      chip.innerHTML = `
        <input type="checkbox" value="${m.name}" ${m.name === 'Leiden_PCA' ? 'checked' : ''}>
        <span>${m.display_name}</span>
        <span class="badge ${m.category}">${m.category}</span>
        ${m.is_r_based ? '<span class="badge r">R</span>' : ''}
      `;
      chip.addEventListener('change', () => chip.classList.toggle('selected', chip.querySelector('input').checked));
      if (chip.querySelector('input').checked) chip.classList.add('selected');
      container.appendChild(chip);
    }
  }

  function renderPlatformSelect(platforms) {
    const select = document.getElementById('select-platform');
    select.innerHTML = '<option value="">Auto-detect</option>';
    for (const p of platforms) {
      const opt = document.createElement('option');
      opt.value = p;
      opt.textContent = p;
      select.appendChild(opt);
    }
  }

  function getSelectedMethods() {
    return Array.from(document.querySelectorAll('#method-grid input:checked')).map(el => el.value);
  }

  // ---------------------------------------------------------------------------
  // Benchmark
  // ---------------------------------------------------------------------------

  async function startBenchmark() {
    if (!state.jobId) {
      showAlert('Please upload data first.', 'error');
      return;
    }

    const methods = getSelectedMethods();
    if (methods.length === 0) {
      showAlert('Select at least one method.', 'error');
      return;
    }

    const nClusters = document.getElementById('input-n-clusters').value;
    const seedsStr = document.getElementById('input-seeds').value || '42';
    const seeds = seedsStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
    const useML = document.getElementById('checkbox-meta-learning').checked;

    const body = {
      job_id: state.jobId,
      methods: methods,
      seeds: seeds.length > 0 ? seeds : [42],
      use_meta_learning: useML,
    };
    if (nClusters) body.n_clusters = parseInt(nClusters);

    try {
      await apiPost('/benchmark', body);
      showAlert('Benchmark started. Monitoring progress...', 'success');
      switchTab('results');
      startPolling();
    } catch (e) {
      showAlert(`Failed to start benchmark: ${e.message}`, 'error');
    }
  }

  // ---------------------------------------------------------------------------
  // Polling
  // ---------------------------------------------------------------------------

  function startPolling() {
    if (state.pollTimer) clearInterval(state.pollTimer);
    state.pollTimer = setInterval(pollJob, 2000);
    pollJob(); // immediate first poll
  }

  function stopPolling() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }

  async function pollJob() {
    if (!state.jobId) return;
    try {
      const job = await apiGet(`/jobs/${state.jobId}`);
      updateProgress(job);
      if (job.status === 'completed') {
        stopPolling();
        showAlert('Benchmark completed! Loading results...', 'success');
        await loadResults();
      } else if (job.status === 'failed') {
        stopPolling();
        showAlert(`Benchmark failed: ${job.message}`, 'error');
      }
    } catch (e) {
      console.error('Poll error:', e);
    }
  }

  function updateProgress(job) {
    const fill = document.getElementById('progress-fill');
    const text = document.getElementById('progress-text');
    fill.style.width = `${job.progress * 100}%`;
    text.textContent = `${job.message} (${Math.round(job.progress * 100)}%)`;
    document.getElementById('progress-container').classList.remove('hidden');

    // Show data profile as soon as available
    if (job.data_profile) {
      renderDataProfile(job.data_profile);
    }
    if (job.cluster_estimation) {
      renderClusterEstimation(job.cluster_estimation);
    }
    if (job.meta_learning && Object.keys(job.meta_learning).length > 0) {
      renderMetaLearning(job.meta_learning);
    }
  }

  // ---------------------------------------------------------------------------
  // Results
  // ---------------------------------------------------------------------------

  async function loadResults() {
    try {
      const results = await apiGet(`/jobs/${state.jobId}/results`);
      renderResults(results);
      await loadRankingTable();
      await loadFigures(results.figures);
      await loadViewerData();
    } catch (e) {
      showAlert(`Failed to load results: ${e.message}`, 'error');
    }
  }

  function renderResults(results) {
    document.getElementById('results-has-gt').textContent = results.has_ground_truth ? 'Yes' : 'No';
    document.getElementById('results-n-clusters').textContent = results.n_clusters;
    document.getElementById('results-job-id').textContent = results.job_id;

    // Download links
    const linksDiv = document.getElementById('download-links');
    linksDiv.innerHTML = '';
    const links = [
      { label: 'Ranking Table (CSV)', url: results.ranking_table },
      { label: 'Viewer Data (JSON)', url: results.viewer_data },
      { label: 'PDF Report', url: results.report },
    ];
    for (const l of links) {
      const a = document.createElement('a');
      a.href = API.replace('/api', '') + l.url;
      a.className = 'btn btn-secondary';
      a.textContent = l.label;
      a.style.marginRight = '8px';
      a.style.marginBottom = '8px';
      a.style.display = 'inline-block';
      linksDiv.appendChild(a);
    }

    document.getElementById('results-section').classList.remove('hidden');
  }

  async function loadRankingTable() {
    try {
      const resp = await fetch(`${API}/jobs/${state.jobId}/download/ranking_table.csv`);
      const csv = await resp.text();
      const rows = parseCSV(csv);
      renderRankingTable(rows);
    } catch (e) {
      console.error('Ranking load error:', e);
    }
  }

  function parseCSV(csv) {
    const lines = csv.trim().split('\n');
    const headers = lines[0].split(',');
    return lines.slice(1).map(line => {
      const vals = line.split(',');
      const obj = {};
      headers.forEach((h, i) => obj[h] = vals[i]);
      return obj;
    });
  }

  function renderRankingTable(rows) {
    const container = document.getElementById('ranking-table-container');
    if (rows.length === 0) {
      container.innerHTML = '<p>No results.</p>';
      return;
    }
    const headers = Object.keys(rows[0]);
    let html = '<table class="ranking-table"><thead><tr>';
    for (const h of headers) {
      html += `<th>${h}</th>`;
    }
    html += '</tr></thead><tbody>';
    for (const row of rows) {
      html += '<tr>';
      for (const h of headers) {
        if (h === 'rank') {
          html += `<td class="rank-cell">${row[h]}</td>`;
        } else if (h === 'recommendation' && row[h]) {
          html += `<td><span class="recommendation">${row[h]}</span></td>`;
        } else {
          html += `<td>${row[h] || ''}</td>`;
        }
      }
      html += '</tr>';
    }
    html += '</tbody></table>';
    container.innerHTML = html;
  }

  async function loadFigures(figureNames) {
    const container = document.getElementById('figures-grid');
    container.innerHTML = '';
    for (const fname of figureNames) {
      const card = document.createElement('div');
      card.className = 'figure-card';
      const img = document.createElement('img');
      img.src = `${API}/jobs/${state.jobId}/download/${fname}`;
      img.alt = fname;
      const caption = document.createElement('div');
      caption.className = 'caption';
      caption.textContent = fname.replace(/\.png$/, '').replace(/_/g, ' ');
      card.appendChild(img);
      card.appendChild(caption);
      container.appendChild(card);
    }
  }

  // ---------------------------------------------------------------------------
  // Data profile & meta-learning
  // ---------------------------------------------------------------------------

  function renderDataProfile(profile) {
    const container = document.getElementById('profile-grid');
    container.innerHTML = '';
    const items = [
      { label: 'Spots', value: profile.n_spots },
      { label: 'Genes', value: profile.n_genes },
      { label: 'Sparsity', value: (profile.sparsity * 100).toFixed(1) + '%' },
      { label: 'Platform', value: profile.platform },
      { label: 'Layout', value: profile.spatial_layout },
      { label: 'Spot Diameter', value: profile.spot_diameter_um ? profile.spot_diameter_um.toFixed(0) + ' um' : 'N/A' },
    ];
    for (const item of items) {
      const div = document.createElement('div');
      div.className = 'profile-item';
      div.innerHTML = `<div class="label">${item.label}</div><div class="value">${item.value}</div>`;
      container.appendChild(div);
    }
    document.getElementById('profile-section').classList.remove('hidden');
  }

  function renderClusterEstimation(est) {
    document.getElementById('est-k').textContent = est.n_clusters;
    document.getElementById('est-scs').textContent = est.spatial_coherence ? est.spatial_coherence.toFixed(3) : 'N/A';
    document.getElementById('est-section').classList.remove('hidden');
  }

  function renderMetaLearning(ml) {
    const container = document.getElementById('ml-info');
    let html = `<strong>Meta-learning recommendation:</strong> Confidence: ${ml.confidence}. `;
    html += `${ml.reason}<br>`;
    if (ml.predicted_ranking && ml.predicted_ranking.length > 0) {
      html += '<strong>Predicted top methods:</strong> ';
      html += ml.predicted_ranking.map(m => Array.isArray(m) ? `${m[0]} (${m[1]})` : m).join(', ');
    }
    container.innerHTML = html;
    container.classList.remove('hidden');
  }

  // ---------------------------------------------------------------------------
  // Interactive Viewer
  // ---------------------------------------------------------------------------

  async function loadViewerData() {
    try {
      const resp = await fetch(`${API}/jobs/${state.jobId}/download/viewer_data.json`);
      state.viewerData = await resp.json();
      setupViewer();
    } catch (e) {
      console.error('Viewer data load error:', e);
      document.getElementById('viewer-container').innerHTML = '<p>Failed to load viewer data.</p>';
    }
  }

  function setupViewer() {
    const data = state.viewerData;
    if (!data || data.n_spots === 0) return;

    const container = document.getElementById('viewer-container');
    container.innerHTML = '';

    // Canvas
    const canvas = document.createElement('canvas');
    canvas.className = 'viewer-canvas';
    container.appendChild(canvas);
    state.viewerCanvas = canvas;
    state.viewerCtx = canvas.getContext('2d');

    // Controls
    const controls = document.createElement('div');
    controls.className = 'viewer-controls';

    // Method selector
    const methodSelect = document.createElement('select');
    methodSelect.id = 'viewer-method-select';
    const methods = Object.keys(data.methods);
    if (data.has_ground_truth) {
      const gtOpt = document.createElement('option');
      gtOpt.value = '__gt__';
      gtOpt.textContent = 'Ground Truth';
      methodSelect.appendChild(gtOpt);
    }
    for (const m of methods) {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      methodSelect.appendChild(opt);
    }
    methodSelect.addEventListener('change', () => {
      state.viewerMethod = methodSelect.value;
      drawViewer();
    });
    controls.appendChild(methodSelect);

    container.appendChild(controls);

    // Tooltip
    const tooltip = document.createElement('div');
    tooltip.className = 'viewer-tooltip';
    tooltip.id = 'viewer-tooltip';
    container.appendChild(tooltip);

    // Set initial method
    state.viewerMethod = methods[0] || '__gt__';

    // Compute bounds
    const coords = data.spots.map(s => [s.x, s.y]);
    const xs = coords.map(c => c[0]);
    const ys = coords.map(c => c[1]);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    state.viewerBounds = { xMin, xMax, yMin, yMax };

    // Size canvas
    const containerWidth = container.clientWidth;
    const aspectRatio = (yMax - yMin) / (xMax - xMin);
    canvas.width = containerWidth;
    canvas.height = Math.min(containerWidth * aspectRatio, 600);
    state.viewerScale = Math.min(canvas.width / (xMax - xMin), canvas.height / (yMax - yMin)) * 0.9;
    state.viewerOffsetX = (canvas.width - (xMax - xMin) * state.viewerScale) / 2;
    state.viewerOffsetY = (canvas.height - (yMax - yMin) * state.viewerScale) / 2;

    // Mouse events for tooltip
    canvas.addEventListener('mousemove', (e) => onViewerMouseMove(e, tooltip));
    canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

    drawViewer();
    document.getElementById('viewer-section').classList.remove('hidden');
  }

  function drawViewer() {
    const ctx = state.viewerCtx;
    const canvas = state.viewerCanvas;
    const data = state.viewerData;
    if (!ctx || !data) return;

    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Get labels for current method
    let labels;
    if (state.viewerMethod === '__gt__') {
      labels = data.spots.map(s => s.ground_truth || 'N/A');
    } else {
      labels = data.methods[state.viewerMethod] || [];
    }

    // Get unique labels and assign colors
    const uniqueLabels = [...new Set(labels)].sort();
    const colors = generateColors(uniqueLabels.length);
    const labelColors = {};
    uniqueLabels.forEach((l, i) => { labelColors[l] = colors[i]; });

    // Draw spots
    const spotSize = Math.max(2, state.viewerScale * 3);
    for (let i = 0; i < data.spots.length; i++) {
      const s = data.spots[i];
      const x = (s.x - state.viewerBounds.xMin) * state.viewerScale + state.viewerOffsetX;
      const y = (s.y - state.viewerBounds.yMin) * state.viewerScale + state.viewerOffsetY;
      ctx.fillStyle = labelColors[labels[i]] || '#666';
      ctx.beginPath();
      ctx.arc(x, y, spotSize, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  function onViewerMouseMove(e, tooltip) {
    const canvas = state.viewerCanvas;
    const data = state.viewerData;
    if (!data) return;
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const my = (e.clientY - rect.top) * (canvas.height / rect.height);

    // Find nearest spot
    let nearest = -1;
    let minDist = Infinity;
    const spotSize = Math.max(2, state.viewerScale * 3);
    for (let i = 0; i < data.spots.length; i++) {
      const s = data.spots[i];
      const x = (s.x - state.viewerBounds.xMin) * state.viewerScale + state.viewerOffsetX;
      const y = (s.y - state.viewerBounds.yMin) * state.viewerScale + state.viewerOffsetY;
      const d = Math.sqrt((mx - x) ** 2 + (my - y) ** 2);
      if (d < spotSize * 2 && d < minDist) {
        minDist = d;
        nearest = i;
      }
    }

    if (nearest >= 0) {
      const s = data.spots[nearest];
      let html = `<b>Spot ${nearest}</b><br>Coords: (${s.x.toFixed(0)}, ${s.y.toFixed(0)})`;
      if (s.ground_truth) html += `<br>GT: ${s.ground_truth}`;
      if (state.viewerMethod !== '__gt__' && data.methods[state.viewerMethod]) {
        html += `<br>Cluster: ${data.methods[state.viewerMethod][nearest]}`;
      }
      // Top expressed genes
      const expr = s.expression;
      const sorted = Object.entries(expr).sort((a, b) => b[1] - a[1]).slice(0, 3);
      html += '<br><b>Top genes:</b><br>' + sorted.map(([g, v]) => `${g}: ${v.toFixed(2)}`).join('<br>');

      tooltip.innerHTML = html;
      tooltip.style.display = 'block';
      tooltip.style.left = (e.clientX - rect.left + 15) + 'px';
      tooltip.style.top = (e.clientY - rect.top + 15) + 'px';
    } else {
      tooltip.style.display = 'none';
    }
  }

  function generateColors(n) {
    const palette = [
      '#0279EE', '#FF9400', '#75A025', '#FD9BED', '#E9ED4C',
      '#17becf', '#bcbd22', '#e377c2', '#7f7f7f', '#8c564b',
      '#aec7e8', '#ffbb78', '#2ca02c', '#d62728', '#9467bd',
      '#1f77b4', '#ff7f0e', '#000000', '#f7b6d2', '#98df8a',
    ];
    if (n <= palette.length) return palette.slice(0, n);
    // Generate additional colors
    const colors = [...palette];
    for (let i = palette.length; i < n; i++) {
      const h = (i * 137.5) % 360;
      colors.push(`hsl(${h}, 65%, 55%)`);
    }
    return colors;
  }

  // ---------------------------------------------------------------------------
  // UI helpers
  // ---------------------------------------------------------------------------

  function showAlert(msg, type) {
    const container = document.getElementById('alert-container');
    container.innerHTML = `<div class="alert alert-${type}">${msg}</div>`;
    if (type === 'success') setTimeout(() => { container.innerHTML = ''; }, 5000);
  }

  function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector(`.tab[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`tab-${tabName}`).classList.add('active');
  }

  function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
  }

  // ---------------------------------------------------------------------------
  // Wire up
  // ---------------------------------------------------------------------------

  document.addEventListener('DOMContentLoaded', () => {
    init();
    setupUpload();
    document.getElementById('btn-start').addEventListener('click', startBenchmark);
    document.querySelectorAll('.tab').forEach(tab => {
      tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });
  });

})();
