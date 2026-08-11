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
    selectedFile: null,
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

  async function apiQc(file, formData) {
    const form = new FormData();
    form.append('file', file);
    for (const [k, v] of Object.entries(formData)) {
      if (v) form.append(k, v);
    }
    const resp = await fetch(API + '/qc', { method: 'POST', body: form });
    if (!resp.ok) throw new Error(`QC error ${resp.status}: ${await resp.text()}`);
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

      // Ask the backend which methods can actually run in this image (the slim
      // deploy profile excludes torch/TF/R backends) so we can disable the rest
      // and default-check only the runnable set.
      try {
        const avail = await apiGet('/methods/availability');
        state.methodAvailability = avail.availability || {};
        state.defaultMethods = avail.default_methods || [];
        if (typeof health !== 'undefined' && health.methods_runnable != null) {
          document.getElementById('health-methods').textContent =
            `${health.methods_runnable}/${health.methods_available}`;
        }
      } catch (e) {
        state.methodAvailability = {};
        state.defaultMethods = [];
      }
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
    state.selectedFile = file;
    document.getElementById('upload-filename').textContent = file.name;
    document.getElementById('upload-filesize').textContent = formatSize(file.size);
    document.getElementById('upload-info').classList.remove('hidden');
    const qcPanel = document.getElementById('qc-panel');
    if (qcPanel) qcPanel.classList.remove('hidden');

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
    const availability = state.methodAvailability || {};
    const defaults = (state.defaultMethods && state.defaultMethods.length)
      ? state.defaultMethods : ['Leiden_PCA'];
    for (const m of state.availableMethods) {
      const info = availability[m.name];
      // Treat as runnable unless the backend explicitly reports it unavailable.
      const runnable = !info || info.available !== false;
      const checked = runnable && defaults.includes(m.name);

      const chip = document.createElement('label');
      chip.className = 'method-chip' + (runnable ? '' : ' unavailable');
      if (!runnable) {
        chip.title = info && info.reason
          ? `Not runnable in this deployment: ${info.reason}`
          : 'Not runnable in this deployment';
      }
      chip.innerHTML = `
        <input type="checkbox" value="${m.name}" ${checked ? 'checked' : ''} ${runnable ? '' : 'disabled'}>
        <span>${m.display_name}</span>
        <span class="badge ${m.category}">${m.category}</span>
        ${m.is_r_based ? '<span class="badge r">R</span>' : ''}
        ${runnable ? '' : '<span class="badge unavailable">unavailable</span>'}
      `;
      chip.addEventListener('change', () => chip.classList.toggle('selected', chip.querySelector('input').checked));
      if (chip.querySelector('input').checked) chip.classList.add('selected');
      container.appendChild(chip);
    }
  }

  // ---------------------------------------------------------------------------
  // SpatialQC (fast, run-first quality check)
  // ---------------------------------------------------------------------------

  function setupQc() {
    const btn = document.getElementById('btn-qc');
    if (btn) btn.addEventListener('click', runQc);
    const demoBtn = document.getElementById('btn-demo');
    if (demoBtn) demoBtn.addEventListener('click', runDemo);
    populateDemos();
  }

  const QC_STATUS_COLOR = { pass: '#3fae49', warn: '#e0a800', fail: '#d64545' };

  async function populateDemos() {
    const sel = document.getElementById('select-demo');
    if (!sel) return;
    try {
      const resp = await apiGet('/qc/demos');
      sel.innerHTML = '';
      for (const d of (resp.demos || [])) {
        const opt = document.createElement('option');
        opt.value = d.name;
        opt.textContent = d.name;
        opt.title = d.description || '';
        sel.appendChild(opt);
      }
    } catch (e) {
      const panel = document.getElementById('demo-panel');
      if (panel) panel.classList.add('hidden');   // demos unavailable — hide the control
    }
  }

  function renderQcResult(res) {
    const result = document.getElementById('qc-result');
    const s = res.summary || {};
    const overall = (s.overall || 'warn');
    const chips = (res.report.modules || []).map(m =>
      `<span class="qc-chip" style="border-color:${QC_STATUS_COLOR[m.status] || '#888'}">
         <b style="color:${QC_STATUS_COLOR[m.status] || '#888'}">${m.status.toUpperCase()}</b> ${escapeHtml(m.name)}
       </span>`).join('');
    const label = res.demo ? `demo: ${escapeHtml(res.demo)}`
                           : `platform ${escapeHtml(res.platform)} (${escapeHtml(res.platform_confidence)})`;
    result.innerHTML = `
      <div class="qc-verdict">
        Overall: <b style="color:${QC_STATUS_COLOR[overall]}">${overall.toUpperCase()}</b>
        &nbsp;·&nbsp; ${s.pass || 0} pass / ${s.warn || 0} warn / ${s.fail || 0} fail
        &nbsp;·&nbsp; ${label}
      </div>
      <div class="qc-chips">${chips}</div>
      <a class="btn btn-secondary" href="${res.qc_report_html}" target="_blank" rel="noopener">Open full SpatialQC report ↗</a>
    `;
    result.classList.remove('hidden');
    showAlert(`SpatialQC complete: ${overall.toUpperCase()}.`, overall === 'fail' ? 'error' : 'success');
  }

  async function runQc() {
    if (!state.selectedFile) {
      showAlert('Choose a data file first.', 'error');
      return;
    }
    const btn = document.getElementById('btn-qc');
    const platform = document.getElementById('select-platform').value || null;
    const sampleId = document.getElementById('input-sample-id').value || null;

    btn.disabled = true;
    const prevLabel = btn.textContent;
    btn.textContent = 'Running SpatialQC…';
    try {
      const res = await apiQc(state.selectedFile, { platform, sample_id: sampleId });
      renderQcResult(res);
    } catch (e) {
      showAlert(`SpatialQC failed: ${e.message}`, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = prevLabel;
    }
  }

  async function runDemo() {
    const btn = document.getElementById('btn-demo');
    const name = document.getElementById('select-demo').value || '';
    btn.disabled = true;
    const prevLabel = btn.textContent;
    btn.textContent = 'Running demo…';
    try {
      const form = new FormData();
      if (name) form.append('name', name);
      const resp = await fetch(API + '/qc/demo', { method: 'POST', body: form });
      if (!resp.ok) throw new Error(`Demo error ${resp.status}: ${await resp.text()}`);
      renderQcResult(await resp.json());
    } catch (e) {
      showAlert(`Demo failed: ${e.message}`, 'error');
    } finally {
      btn.disabled = false;
      btn.textContent = prevLabel;
    }
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
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
      if (job.status === 'completed' || job.status === 'completed_partial') {
        stopPolling();
        if (job.status === 'completed_partial') {
          const nFailed = (job.method_summary && job.method_summary.n_failed) || 0;
          showAlert(`Benchmark completed with ${nFailed} failed method(s). Loading partial results…`, 'warning');
        } else {
          showAlert('Benchmark completed! Loading results...', 'success');
        }
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

  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function renderMethodSummary(summary) {
    const el = document.getElementById('results-warnings');
    if (!el) return;
    if (!summary || !summary.failed_methods || summary.failed_methods.length === 0) {
      el.innerHTML = '';
      return;
    }
    const items = summary.failed_methods
      .map(f => `<li><strong>${escapeHtml(f.method)}</strong>: <code>${escapeHtml(f.error)}</code></li>`)
      .join('');
    el.innerHTML =
      `<div class="alert alert-warning">` +
      `${summary.n_succeeded} method(s) succeeded, ${summary.n_failed} failed. ` +
      `Rankings below cover the methods that completed.` +
      `<ul class="method-fail-list">${items}</ul>` +
      `</div>`;
  }

  function renderQC(qc) {
    const container = document.getElementById('results-warnings');
    if (!container || !qc || !qc.summary) return;
    const colors = { pass: '#75A025', warn: '#E0A800', fail: '#D62728' };
    const s = qc.summary;
    const chips = (qc.modules || []).map(m =>
      `<span style="display:inline-block;margin:2px 4px;padding:2px 8px;border-radius:10px;` +
      `font-size:11px;color:#fff;background:${colors[m.status] || '#888'}" title="${escapeHtml(m.message || '')}">` +
      `${escapeHtml(m.name)}: ${String(m.status).toUpperCase()}</span>`
    ).join('');
    const cls = s.overall === 'fail' ? 'error' : (s.overall === 'warn' ? 'warning' : 'success');
    const div = document.createElement('div');
    div.className = 'alert alert-' + cls;
    div.innerHTML =
      `<strong>Data QC: ${String(s.overall).toUpperCase()}</strong> — ` +
      `${s.pass} pass &middot; ${s.warn} warn &middot; ${s.fail} fail` +
      `<div style="margin-top:6px">${chips}</div>`;
    container.appendChild(div);
  }

  function renderResults(results) {
    document.getElementById('results-has-gt').textContent = results.has_ground_truth ? 'Yes' : 'No';
    document.getElementById('results-n-clusters').textContent = results.n_clusters;
    document.getElementById('results-job-id').textContent = results.job_id;

    renderMethodSummary(results.method_summary);
    renderQC(results.qc);

    // Download links
    const linksDiv = document.getElementById('download-links');
    linksDiv.innerHTML = '';
    const links = [
      { label: 'Ranking Table (CSV)', url: results.ranking_table },
      { label: 'Viewer Data (JSON)', url: results.viewer_data },
      { label: 'PDF Report', url: results.report },
    ];
    if (results.qc && results.qc.summary) {
      links.push({ label: 'QC Report (HTML)', url: results.qc_report });
    }
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

    // Configure the coordinate transform. Two modes:
    //  - "image": a histology image is available; the content space is the
    //    image in pixels and spots map via coord * scalef (canonical Visium
    //    alignment, so spots land ON the tissue).
    //  - "coords": no image; fall back to the spot bounding box.
    const containerWidth = container.clientWidth || 800;
    if (data.histology && data.histology.data_url) {
      const H = data.histology;
      state.viewerMode = 'image';
      state.histScalef = H.scalef;
      const aspect = H.height / H.width;
      canvas.width = containerWidth;
      canvas.height = Math.min(Math.max(1, containerWidth * aspect), 700);
      state.viewerScale = Math.min(canvas.width / H.width, canvas.height / H.height);
      state.viewerOffsetX = (canvas.width - H.width * state.viewerScale) / 2;
      state.viewerOffsetY = (canvas.height - H.height * state.viewerScale) / 2;
      state.spotRadius = H.spot_diameter_fullres
        ? Math.max(1.5, 0.5 * H.spot_diameter_fullres * H.scalef * state.viewerScale)
        : Math.max(1.5, Math.min(6, 0.4 * canvas.width / Math.sqrt(data.n_spots || 1)));
      const img = new Image();
      state.viewerImage = img;
      state.viewerImageReady = false;
      img.onload = () => { state.viewerImageReady = true; drawViewer(); };
      img.src = H.data_url;
    } else {
      const xs = data.spots.map(s => s.x), ys = data.spots.map(s => s.y);
      const xMin = Math.min(...xs), xMax = Math.max(...xs);
      const yMin = Math.min(...ys), yMax = Math.max(...ys);
      const dx = (xMax - xMin) || 1, dy = (yMax - yMin) || 1;
      state.viewerMode = 'coords';
      state.viewerBounds = { xMin, xMax, yMin, yMax };
      const aspect = dy / dx;
      canvas.width = containerWidth;
      canvas.height = Math.min(Math.max(1, containerWidth * aspect), 700);
      state.viewerScale = Math.min(canvas.width / dx, canvas.height / dy) * 0.95;
      state.viewerOffsetX = (canvas.width - dx * state.viewerScale) / 2;
      state.viewerOffsetY = (canvas.height - dy * state.viewerScale) / 2;
      state.spotRadius = Math.max(1.5, Math.min(6, 0.4 * canvas.width / Math.sqrt(data.n_spots || 1)));
    }

    // Mouse events for tooltip
    canvas.addEventListener('mousemove', (e) => onViewerMouseMove(e, tooltip));
    canvas.addEventListener('mouseleave', () => { tooltip.style.display = 'none'; });

    drawViewer();
    document.getElementById('viewer-section').classList.remove('hidden');
  }

  function drawTissueMask(ctx, mask, scaleFactor) {
    const origSpaceW = mask.original_width / scaleFactor;
    const origSpaceH = mask.original_height / scaleFactor;
    const cellW = origSpaceW / mask.width;
    const cellH = origSpaceH / mask.height;
    ctx.fillStyle = 'rgba(233, 237, 76, 0.10)';
    for (let r = 0; r < mask.rows.length; r++) {
      const row = mask.rows[r];
      for (let c = 0; c < row.length; c++) {
        if (row[c] !== '1') continue;
        const origX = c * cellW;
        const origY = r * cellH;
        const x = (origX - state.viewerBounds.xMin) * state.viewerScale + state.viewerOffsetX;
        const y = (origY - state.viewerBounds.yMin) * state.viewerScale + state.viewerOffsetY;
        const w = Math.max(1, cellW * state.viewerScale);
        const h = Math.max(1, cellH * state.viewerScale);
        ctx.fillRect(x, y, w, h);
      }
    }
  }

  function spotCanvasXY(s) {
    if (state.viewerMode === 'image') {
      return [
        s.x * state.histScalef * state.viewerScale + state.viewerOffsetX,
        s.y * state.histScalef * state.viewerScale + state.viewerOffsetY,
      ];
    }
    return [
      (s.x - state.viewerBounds.xMin) * state.viewerScale + state.viewerOffsetX,
      (s.y - state.viewerBounds.yMin) * state.viewerScale + state.viewerOffsetY,
    ];
  }

  function drawViewer() {
    const ctx = state.viewerCtx;
    const canvas = state.viewerCanvas;
    const data = state.viewerData;
    if (!ctx || !data) return;

    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (state.viewerMode === 'image') {
      if (state.viewerImageReady && state.viewerImage) {
        const H = data.histology;
        ctx.drawImage(
          state.viewerImage, 0, 0, H.width, H.height,
          state.viewerOffsetX, state.viewerOffsetY,
          H.width * state.viewerScale, H.height * state.viewerScale,
        );
      }
    } else if (data.tissue_mask && data.tissue_mask_scale_factor) {
      drawTissueMask(ctx, data.tissue_mask, data.tissue_mask_scale_factor);
    }

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
    const r = state.spotRadius || 2;
    for (let i = 0; i < data.spots.length; i++) {
      const [x, y] = spotCanvasXY(data.spots[i]);
      ctx.fillStyle = labelColors[labels[i]] || '#666';
      ctx.beginPath();
      ctx.arc(x, y, r, 0, 2 * Math.PI);
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

    // Find nearest spot (same transform as drawViewer)
    let nearest = -1;
    let minDist = Infinity;
    const hitR = Math.max(4, (state.spotRadius || 2) * 2);
    for (let i = 0; i < data.spots.length; i++) {
      const [x, y] = spotCanvasXY(data.spots[i]);
      const d = Math.sqrt((mx - x) ** 2 + (my - y) ** 2);
      if (d < hitR && d < minDist) {
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
    setupQc();
    document.getElementById('btn-start').addEventListener('click', startBenchmark);
    document.querySelectorAll('.tab').forEach(tab => {
      tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });
  });

})();
