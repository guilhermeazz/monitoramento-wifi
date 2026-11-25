/**
 * ============================================================================
 * Spatial CSI v4.2 — Frontend Dark Liquid Glass
 * Modo Operacional Único Contínuo (0, 1, 2, 3+) • RTI 0.25m • MOT Kalman 2D
 * Wizard de Calibração em 2 Etapas • Editor Visual de Posicionamento RF
 * ============================================================================
 */
(() => {
  let ws = null;
  let liveState = null;
  let diagnosticsOpen = false;
  let layoutModalOpen = false;
  let initialReportLoaded = false;
  let lastSimPeopleRequested = 0;

  const SPARKLINE_MAX_POINTS = 300;
  const mvsSeries = [];

  const renderState = {
    phase: 0,
    intensity: 0,
    targetsMap: {},
    nodeIntensities: { esp_1: 0, esp_2: 0, esp_3: 0 },
  };

  // Geometria ativa da sala (sincronizada com backend)
  let roomGeometry = {
    width_m: 5.0,
    length_m: 4.5,
    tx: { id: 'tx_router', name: 'Cisco Linksys E1200 (TX)', mac: '20:aa:4b:4f:67:1d', x: 0.50, y: 0.10 },
    rx_nodes: [
      { id: 'esp_1', name: 'ESP32-S3 #1', mac: 'c0:56:27:e1:20:01', x: 0.14, y: 0.84 },
      { id: 'esp_2', name: 'ESP32-S3 #2', mac: 'c0:56:27:e1:20:02', x: 0.86, y: 0.84 },
      { id: 'esp_3', name: 'ESP32-S3 #3', mac: 'c0:56:27:e1:20:03', x: 0.50, y: 0.88 },
    ],
  };

  // Cópia editável no Editor Visual de Layout
  let draftGeometry = JSON.parse(JSON.stringify(roomGeometry));

  // Referências DOM — Top Bar
  const connDot = document.getElementById('conn-dot');
  const connModeLabel = document.getElementById('conn-mode-label');
  const statusPillDot = document.getElementById('status-pill-dot');
  const statusPillText = document.getElementById('status-pill-text');
  const statusConfidenceText = document.getElementById('status-confidence-text');
  const calQualityDot = document.getElementById('cal-quality-dot');
  const btnOpenLayoutEditor = document.getElementById('btn-open-layout-editor');
  const btnOpenCalibrate = document.getElementById('btn-open-calibrate');
  const btnOpenDiagnostics = document.getElementById('btn-open-diagnostics');

  // Referências DOM — Hero Spatial Canvas & Sparkline
  const spatialCanvas = document.getElementById('spatial-canvas');
  const spatialCtx = spatialCanvas ? spatialCanvas.getContext('2d') : null;
  const topologyBadgeText = document.getElementById('topology-badge-text');
  const activeZoneLabel = document.getElementById('active-zone-label');
  const canvasIntensityVal = document.getElementById('canvas-intensity-val');
  const sparklineCanvas = document.getElementById('sparkline-canvas');
  const sparklineCtx = sparklineCanvas ? sparklineCanvas.getContext('2d') : null;
  const metricMvsVal = document.getElementById('metric-mvs-val');
  const metricThetaVal = document.getElementById('metric-theta-val');
  const metricPeakVal = document.getElementById('metric-peak-val');

  // Referências DOM — Contagem de Pessoas
  const heroPeopleCount = document.getElementById('hero-people-count');
  const heroPeopleUnit = document.getElementById('hero-people-unit');
  const heroOccupancyTitle = document.getElementById('hero-occupancy-title');
  const heroOccupancySubtitle = document.getElementById('hero-occupancy-subtitle');
  const heroPeakCount = document.getElementById('hero-peak-count');
  const heroZoneShort = document.getElementById('hero-zone-short');
  const heroConfVal = document.getElementById('hero-conf-val');
  const quickPeopleSelector = document.getElementById('quick-people-selector');
  const simPeopleBtns = document.querySelectorAll('.sim-p-btn');

  // Referências DOM — Análise Inteligente (LLM)
  const llmStatusDot = document.getElementById('llm-status-dot');
  const llmProviderBadge = document.getElementById('llm-provider-badge');
  const selectLlmModel = document.getElementById('select-llm-model');
  const btnFullReport = document.getElementById('btn-full-report');
  const llmUpdatedTime = document.getElementById('llm-updated-time');
  const llmInsightText = document.getElementById('llm-insight-text');
  const btnRefreshSummary = document.getElementById('btn-refresh-summary');
  const llmChips = document.querySelectorAll('.llm-chip');
  const llmMarkdownOutput = document.getElementById('llm-markdown-output');
  const formAskLlm = document.getElementById('form-ask-llm');
  const inputAskLlm = document.getElementById('input-ask-llm');

  // Referências DOM — Wizard de Calibração em 2 Etapas
  const calOverlay = document.getElementById('calibration-overlay');
  const calStep1Card = document.getElementById('cal-step1-card');
  const calStep2Card = document.getElementById('cal-step2-card');
  const calStep1Status = document.getElementById('cal-step1-status');
  const calStep2Status = document.getElementById('cal-step2-status');
  const calRingProgress = document.getElementById('cal-ring-progress');
  const calRingLabel = document.getElementById('cal-ring-label');
  const calModalTitle = document.getElementById('cal-modal-title');
  const calModalDesc = document.getElementById('cal-modal-desc');
  const calQualityBadge = document.getElementById('cal-quality-badge');
  const btnStartCalStage1 = document.getElementById('btn-start-cal-stage1') || document.getElementById('btn-start-cal-confirm');
  const btnStartCalStage2 = document.getElementById('btn-start-cal-stage2');
  const btnCloseCalModal = document.getElementById('btn-close-cal-modal');

  // Referências DOM — Editor Visual de Layout da Sala
  const layoutEditorModal = document.getElementById('layout-editor-modal');
  const btnCloseLayoutEditor = document.getElementById('btn-close-layout-editor');
  const layoutEditorCanvas = document.getElementById('layout-editor-canvas');
  const layoutEditorCtx = layoutEditorCanvas ? layoutEditorCanvas.getContext('2d') : null;
  const layoutGridInfo = document.getElementById('layout-grid-info');
  const chkSnap025 = document.getElementById('chk-snap-025');
  const layoutWarningsBox = document.getElementById('layout-warnings-box');
  const inpRoomW = document.getElementById('inp-room-w');
  const inpRoomL = document.getElementById('inp-room-l');
  const layoutNodesInputs = document.getElementById('layout-nodes-inputs');
  const btnSaveGeometry = document.getElementById('btn-save-geometry');
  const btnResetGeometry = document.getElementById('btn-reset-geometry');

  // Referências DOM — Gaveta Configurações & Diagnostics
  const drawerBackdrop = document.getElementById('drawer-backdrop');
  const diagnosticsDrawer = document.getElementById('diagnostics-drawer');
  const btnCloseDiagnostics = document.getElementById('btn-close-diagnostics');
  const btnSourceSim = document.getElementById('btn-source-sim');
  const btnSourceHw = document.getElementById('btn-source-hw');
  const drawerSimControls = document.getElementById('drawer-sim-controls');
  const nodeCntBtns = document.querySelectorAll('.node-cnt-btn');
  const btnInjectAgc = document.getElementById('btn-inject-agc');
  const drawerRoomSummary = document.getElementById('drawer-room-summary');
  const btnDrawerOpenLayout = document.getElementById('btn-drawer-open-layout');
  const diagCalQuality = document.getElementById('diag-cal-quality');
  const diagSubspaceSummary = document.getElementById('diag-subspace-summary');
  const diagNodesList = document.getElementById('diag-nodes-list');
  const diagNodeSelect = document.getElementById('diag-node-select');
  const subcarrierCanvas = document.getElementById('subcarrier-canvas');
  const subcarrierCtx = subcarrierCanvas ? subcarrierCanvas.getContext('2d') : null;
  const sliderSensMult = document.getElementById('slider-sens-mult');
  const lblSensMult = document.getElementById('lbl-sens-mult');
  const sliderHoldSec = document.getElementById('slider-hold-sec');
  const lblHoldSec = document.getElementById('lbl-hold-sec');

  const STATE_THEME = {
    EMPTY: {
      label: 'Ambiente Livre',
      hex: '#38BDF8',
      rgb: '56, 189, 248',
      dotClass: 'w-2.5 h-2.5 rounded-full bg-sky-400',
    },
    PRESENCE_STATIC: {
      label: 'Presença Retida',
      hex: '#10B981',
      rgb: '16, 185, 129',
      dotClass: 'w-2.5 h-2.5 rounded-full bg-emerald-400',
    },
    MOTION: {
      label: 'Em Movimento',
      hex: '#F59E0B',
      rgb: '245, 158, 11',
      dotClass: 'w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse',
    },
  };

  const TARGET_COLORS = [
    { hex: '#F59E0B', rgb: '245, 158, 11' },
    { hex: '#38BDF8', rgb: '56, 189, 248' },
    { hex: '#A855F7', rgb: '168, 85, 247' },
    { hex: '#10B981', rgb: '16, 185, 129' },
  ];

  // ==========================================================================
  // 1. Renderizador Markdown para Relatório LLM
  // ==========================================================================
  function renderMarkdownToHtml(md) {
    if (!md) return '';
    const lines = md.split('\n');
    let html = '';
    let inList = false;
    let inTable = false;

    const fmt = (str) =>
      str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\*\*(.+?)\*\*/g, '<strong class="text-white font-semibold">$1</strong>')
        .replace(/\*(.+?)\*/g, '<em class="text-sky-200">$1</em>')
        .replace(/`([^`]+)`/g, '<code>$1</code>');

    for (let i = 0; i < lines.length; i++) {
      const raw = lines[i].trim();
      if (!raw) {
        if (inList) { html += '</ul>'; inList = false; }
        if (inTable) { html += '</tbody></table>'; inTable = false; }
        continue;
      }

      if (raw.startsWith('|') && raw.endsWith('|')) {
        if (raw.includes('---')) continue;
        const cells = raw.slice(1, -1).split('|').map((c) => fmt(c.trim()));
        if (!inTable) {
          html += '<table><thead><tr>' + cells.map((c) => `<th>${c}</th>`).join('') + '</tr></thead><tbody>';
          inTable = true;
        } else {
          html += '<tr>' + cells.map((c) => `<td>${c}</td>`).join('') + '</tr>';
        }
        continue;
      } else if (inTable) {
        html += '</tbody></table>';
        inTable = false;
      }

      if (raw.startsWith('# ')) html += `<h1>${fmt(raw.slice(2))}</h1>`;
      else if (raw.startsWith('## ')) html += `<h2>${fmt(raw.slice(3))}</h2>`;
      else if (raw.startsWith('### ')) html += `<h3>${fmt(raw.slice(4))}</h3>`;
      else if (raw.startsWith('> ')) html += `<blockquote>${fmt(raw.slice(2))}</blockquote>`;
      else if (raw.startsWith('- ') || /^\d+\.\s/.test(raw)) {
        if (!inList) { html += '<ul>'; inList = true; }
        html += `<li>${fmt(raw.replace(/^(-|\d+\.)\s+/, ''))}</li>`;
      } else {
        html += `<p>${fmt(raw)}</p>`;
      }
    }
    if (inList) html += '</ul>';
    if (inTable) html += '</tbody></table>';
    return html;
  }

  // ==========================================================================
  // 2. Conexão WebSocket Leve (10 Hz)
  // ==========================================================================
  function connectWebSocket() {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${window.location.host}/ws/live?diagnostics=${diagnosticsOpen}`;
    ws = new WebSocket(url);

    ws.onopen = () => {
      connDot.className = 'w-2 h-2 rounded-full bg-emerald-400';
      if (diagnosticsOpen) ws.send('diagnostics:on');
      if (!initialReportLoaded) {
        initialReportLoaded = true;
        setTimeout(() => requestLlmAnalysis('full_report'), 500);
      }
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        liveState = data;
        onStateReceived(data);
      } catch (err) {
        console.error('Erro ao decodificar frame WS:', err);
      }
    };

    ws.onclose = () => {
      connDot.className = 'w-2 h-2 rounded-full bg-rose-400';
      connModeLabel.textContent = 'Reconectando...';
      setTimeout(connectWebSocket, 1500);
    };

    ws.onerror = () => ws.close();
  }

  // ==========================================================================
  // 3. Atualização do Modo Único de Ocupação e Contagem (0, 1, 2, 3+)
  // ==========================================================================
  function formatScientificMvs(val) {
    if (!val || val <= 0) return '0.00';
    return (val * 1e5).toFixed(2);
  }

  function syncSimButtonsHighlight(simCount) {
    simPeopleBtns.forEach((btn) => {
      const pVal = parseInt(btn.getAttribute('data-sim-people'), 10);
      btn.classList.toggle('active', pVal === simCount);
    });
  }

  function updatePeopleCounterBlock(state) {
    const count = state.target_count || 0;
    const disp = state.count_display || (count >= 4 ? '3+' : `${count}`);
    const peak = state.peak_occupancy || count;
    const occ = state.occupancy_state || 'EMPTY';
    const confPct = Math.round((state.confidence || 0) * 100);
    const cleanZone = (state.active_zone || 'NONE') === 'NONE'
      ? 'Livre'
      : state.active_zone.replace(/_/g, ' ');

    heroPeopleCount.textContent = disp;
    heroPeopleUnit.textContent = count === 1 ? 'Pessoa' : 'Pessoas';
    heroPeakCount.textContent = `${peak} ${peak === 1 ? 'pessoa' : 'pessoas'}`;
    heroZoneShort.textContent = cleanZone;
    heroConfVal.textContent = `${confPct}%`;

    if (occ === 'EMPTY' || count === 0) {
      heroOccupancyTitle.textContent = 'AMBIENTE LIVRE';
      heroOccupancySubtitle.textContent = '0 pessoas na sala • Monitoramento contínuo ativo';
    } else if (occ === 'PRESENCE_STATIC') {
      heroOccupancyTitle.textContent = count === 1
        ? '1 PESSOA NA SALA (PARADA)'
        : `${disp} PESSOAS NA SALA (ESTÁTICAS)`;
      heroOccupancySubtitle.textContent = `Presença contabilizada via Hold-Timer • ${cleanZone}`;
    } else {
      heroOccupancyTitle.textContent = count === 1
        ? '1 PESSOA NA SALA'
        : `${disp} PESSOAS NA SALA`;
      heroOccupancySubtitle.textContent = `Rastreamento ativo em ${cleanZone} (${(state.intensity || 0).toFixed(1)}x limiar)`;
    }

    if (state.mode === 'simulation') {
      quickPeopleSelector.classList.remove('hidden');
      if (state.diagnostics && typeof state.diagnostics.simulated_people === 'number') {
        lastSimPeopleRequested = state.diagnostics.simulated_people;
      }
      syncSimButtonsHighlight(lastSimPeopleRequested);
    } else {
      quickPeopleSelector.classList.add('hidden');
    }
  }

  function onStateReceived(state) {
    const occ = state.occupancy_state || 'EMPTY';
    const theme = STATE_THEME[occ] || STATE_THEME.EMPTY;

    if (document.body.getAttribute('data-state') !== occ) {
      document.body.setAttribute('data-state', occ);
    }

    connModeLabel.textContent = state.mode === 'hardware' ? 'Hardware Real (ESP32)' : 'Simulador';
    statusPillDot.className = theme.dotClass;
    const count = state.target_count || 0;
    const disp = state.count_display || `${count}`;
    statusPillText.textContent = count === 0
      ? 'Ocupação: 0 Pessoas'
      : `Ocupação: ${disp} ${count === 1 ? 'Pessoa' : 'Pessoas'} (${theme.label})`;
    statusConfidenceText.textContent = `${Math.round((state.confidence || 0) * 100)}%`;

    updatePeopleCounterBlock(state);

    const topoLabels = {
      NO_SENSORS: 'Aguardando ESP32-S3',
      SINGLE_NODE_FRESNEL: '1 Link • Zona/Halo de Incerteza Fresnel',
      DUAL_NODE_SECTOR: '2 Links • RTI + Kalman 2D',
      MULTI_NODE_KALMAN_2D: `${state.active_nodes_count || 3} Links • RTI 0.25m + Multi-Kalman`,
    };
    topologyBadgeText.textContent = topoLabels[state.topology_mode] || state.topology_mode;
    const cleanZone = (state.active_zone || 'NONE') === 'NONE'
      ? 'Livre'
      : state.active_zone.replace(/_/g, ' ');
    activeZoneLabel.textContent = `Zona: ${cleanZone}`;
    canvasIntensityVal.textContent = `${(state.intensity || 0).toFixed(1)}x`;

    mvsSeries.push({
      mvs: state.mvs_current || 0,
      theta: state.threshold_on || 2.5e-6,
    });
    if (mvsSeries.length > SPARKLINE_MAX_POINTS) mvsSeries.shift();

    metricMvsVal.textContent = formatScientificMvs(state.mvs_current);
    metricThetaVal.textContent = formatScientificMvs(state.threshold_on);
    const peak30s = mvsSeries.reduce((acc, item) => Math.max(acc, item.mvs), 0);
    metricPeakVal.textContent = formatScientificMvs(peak30s);
    drawSparkline();

    if (state.llm_insight) {
      llmInsightText.textContent = state.llm_insight.summary || '';
      llmUpdatedTime.textContent = state.llm_insight.updated_at || '--:--:--';
    }

    if (Array.isArray(state.nodes) && !layoutModalOpen) {
      state.nodes.forEach((n) => {
        const localNode = roomGeometry.rx_nodes.find((r) => r.id === n.id);
        if (localNode) {
          localNode.x = n.x;
          localNode.y = n.y;
          localNode.name = n.name;
        }
      });
    }

    if (state.calibration) {
      updateCalibrationUI(state.calibration);
    }

    if (diagnosticsOpen) {
      renderDiagnosticsDrawer(state);
    }
  }

  // ==========================================================================
  // 4. Camada de Análise Inteligente (LLM)
  // ==========================================================================
  async function checkLlmStatus(preferredModel = null) {
    try {
      const q = preferredModel ? `?model=${encodeURIComponent(preferredModel)}` : '';
      const res = await fetch(`/api/llm/status${q}`);
      if (!res.ok) return;
      const status = await res.json();

      if (status.online) {
        llmStatusDot.className = 'w-2 h-2 rounded-full bg-emerald-400 animate-pulse';
        llmProviderBadge.textContent = status.badge_text || `Ollama (${status.active_model})`;
      } else {
        llmStatusDot.className = 'w-2 h-2 rounded-full bg-sky-400';
        llmProviderBadge.textContent = 'Motor Analítico Local';
      }

      if (selectLlmModel && Array.isArray(status.available_models) && status.available_models.length) {
        const installedSet = new Set(status.installed_models || []);
        selectLlmModel.innerHTML = status.available_models
          .map((m) => {
            const mark = installedSet.has(m) ? ' ✓' : '';
            const sel = m === status.active_model ? 'selected' : '';
            return `<option value="${m}" class="bg-slate-900" ${sel}>${m}${mark}</option>`;
          })
          .join('');
      }
    } catch (e) {
      console.debug('Falha ao verificar status da LLM:', e);
    }
  }

  async function requestLlmAnalysis(mode = 'full_report', questionText = null) {
    const model = selectLlmModel ? selectLlmModel.value : 'qwen2.5:7b';
    if (mode !== 'quick') {
      llmMarkdownOutput.innerHTML = `<p class="text-sky-300 animate-pulse">Sintetizando análise de ocupação, contagem e sensores (${model})...</p>`;
    }

    try {
      const res = await fetch('/api/analysis/summarize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode, question: questionText, model }),
      });
      if (!res.ok) return;
      const data = await res.json();
      if (data.insight) {
        llmInsightText.textContent = data.insight.summary;
        llmUpdatedTime.textContent = data.insight.updated_at;
      }
      if (data.report && data.report.markdown_report) {
        llmMarkdownOutput.innerHTML = renderMarkdownToHtml(data.report.markdown_report);
      }
    } catch (err) {
      llmMarkdownOutput.innerHTML = `<p class="text-rose-300">Erro ao consultar análise: ${err.message}</p>`;
    }
  }

  selectLlmModel?.addEventListener('change', () => checkLlmStatus(selectLlmModel.value));
  btnFullReport?.addEventListener('click', () => requestLlmAnalysis('full_report'));
  btnRefreshSummary?.addEventListener('click', () => requestLlmAnalysis('full_report'));
  llmChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const q = chip.getAttribute('data-ask');
      if (inputAskLlm) inputAskLlm.value = q;
      requestLlmAnalysis('ask', q);
    });
  });
  formAskLlm?.addEventListener('submit', (e) => {
    e.preventDefault();
    const q = (inputAskLlm?.value || '').trim();
    if (q) requestLlmAnalysis('ask', q);
  });

  // ==========================================================================
  // 5. Wizard de Calibração Guiada em 2 Etapas (Sala Vazia + Ref. 1 Pessoa)
  // ==========================================================================
  function updateCalibrationUI(cal) {
    const qColors = {
      OPTIMAL: 'bg-emerald-400',
      ACCEPTABLE: 'bg-amber-400',
      NOISY: 'bg-rose-400',
      UNCALIBRATED: 'bg-slate-500',
    };
    if (calQualityDot) {
      calQualityDot.className = `w-1.5 h-1.5 rounded-full ${qColors[cal.quality] || 'bg-slate-500'}`;
    }

    if (calStep1Status) calStep1Status.textContent = cal.calibrated ? `✓ ${cal.quality}` : '5s + 15s';
    if (calStep2Status) calStep2Status.textContent = cal.stage2_calibrated ? '✓ Equalizado' : '20s andando';
    calStep2Card?.classList.toggle('opacity-75', !cal.calibrated && cal.stage !== 2);
    calStep1Card?.classList.toggle('border-sky-400/50', cal.stage === 1);
    calStep2Card?.classList.toggle('border-emerald-400/50', cal.stage === 2 || cal.stage2_calibrated);

    const circumference = 263.89;
    const busy = ['PREPARING', 'CAPTURING', 'PREPARING_STAGE2', 'CAPTURING_STAGE2'].includes(cal.state);
    if (btnStartCalStage1) btnStartCalStage1.disabled = busy;
    if (btnStartCalStage2) btnStartCalStage2.disabled = busy;

    if (cal.state === 'PREPARING') {
      calOverlay?.classList.add('open');
      if (calRingProgress) {
        calRingProgress.style.stroke = '#F59E0B';
        calRingProgress.style.strokeDashoffset = `${circumference * (1 - (cal.progress || 0))}`;
      }
      if (calRingLabel) calRingLabel.textContent = `${(cal.remaining_sec || 0).toFixed(1)}s`;
      if (calModalTitle) calModalTitle.textContent = 'Etapa 1: Esvazie a Sala';
      if (calModalDesc) calModalDesc.textContent = 'Saia da área monitorada. A captura do baseline vazio (15s) iniciará automaticamente.';
      if (calQualityBadge) calQualityBadge.textContent = 'Aguardando saída da sala...';
    } else if (cal.state === 'CAPTURING') {
      calOverlay?.classList.add('open');
      if (calRingProgress) {
        calRingProgress.style.stroke = '#38BDF8';
        calRingProgress.style.strokeDashoffset = `${circumference * (1 - (cal.progress || 0))}`;
      }
      if (calRingLabel) calRingLabel.textContent = `${Math.round((cal.progress || 0) * 100)}%`;
      if (calModalTitle) calModalTitle.textContent = 'Etapa 1: Coletando Baseline de Sala Vazia (15s)';
      if (calModalDesc) calModalDesc.textContent = 'Calculando NBVI (12 subportadoras), μ₀, σ₀, P95 por link e piso de autovalores (λ₁,₀, λ₂,₀).';
      if (calQualityBadge) calQualityBadge.textContent = `Restante: ${(cal.remaining_sec || 0).toFixed(1)}s`;
    } else if (cal.state === 'STAGE1_DONE') {
      calOverlay?.classList.add('open');
      if (calRingProgress) {
        calRingProgress.style.stroke = '#10B981';
        calRingProgress.style.strokeDashoffset = '0';
      }
      if (calRingLabel) calRingLabel.textContent = 'Etapa 1 ✓';
      if (calModalTitle) calModalTitle.textContent = 'Etapa 1 Concluída! Avance para a Etapa 2';
      if (calModalDesc) calModalDesc.textContent = cal.noise_diagnosis || 'Agora clique em "2. Referência 1 Pessoa (20s)" e caminhe sozinho pela sala.';
      if (calQualityBadge) calQualityBadge.textContent = `Qualidade do Ruído: ${cal.quality}`;
    } else if (cal.state === 'PREPARING_STAGE2') {
      calOverlay?.classList.add('open');
      if (calRingProgress) {
        calRingProgress.style.stroke = '#F59E0B';
        calRingProgress.style.strokeDashoffset = `${circumference * (1 - (cal.progress || 0))}`;
      }
      if (calRingLabel) calRingLabel.textContent = `${(cal.remaining_sec || 0).toFixed(1)}s`;
      if (calModalTitle) calModalTitle.textContent = 'Etapa 2: Posicione 1 Pessoa na Sala';
      if (calModalDesc) calModalDesc.textContent = 'Prepare-se para caminhar normalmente cobrindo os setores da sala durante 20 segundos.';
      if (calQualityBadge) calQualityBadge.textContent = 'Iniciando captura de referência...';
    } else if (cal.state === 'CAPTURING_STAGE2') {
      calOverlay?.classList.add('open');
      if (calRingProgress) {
        calRingProgress.style.stroke = '#10B981';
        calRingProgress.style.strokeDashoffset = `${circumference * (1 - (cal.progress || 0))}`;
      }
      if (calRingLabel) calRingLabel.textContent = `${Math.round((cal.progress || 0) * 100)}%`;
      if (calModalTitle) calModalTitle.textContent = 'Etapa 2: Caminhe pela Sala (1 Pessoa • 20s)';
      if (calModalDesc) calModalDesc.textContent = 'Medindo ganho espacial G_l por link e envelope espectral (λ₁,ref, λ₂,ref, E₁,ref) para destravar 2+ pessoas.';
      if (calQualityBadge) calQualityBadge.textContent = `Caminhada restante: ${(cal.remaining_sec || 0).toFixed(1)}s`;
    } else if (cal.state === 'COMPLETED') {
      if (calRingProgress) {
        calRingProgress.style.stroke = '#10B981';
        calRingProgress.style.strokeDashoffset = '0';
      }
      if (calRingLabel) calRingLabel.textContent = '100%';
      if (calModalTitle) calModalTitle.textContent = 'Calibração de 2 Etapas Finalizada!';
      if (calModalDesc) calModalDesc.textContent = cal.noise_diagnosis || 'Ganhos G_l e régua espectral salvos em calibration_profile.json.';
      if (calQualityBadge) calQualityBadge.textContent = `Qualidade: ${cal.quality} • Pronto para 0, 1, 2 e 3+ pessoas`;
    } else if (cal.calibrated && calQualityBadge) {
      calQualityBadge.textContent = cal.noise_diagnosis || `Calibrado (${cal.quality})`;
    }
  }

  btnOpenCalibrate?.addEventListener('click', () => calOverlay?.classList.add('open'));
  btnCloseCalModal?.addEventListener('click', async () => {
    calOverlay?.classList.remove('open');
    if (liveState && liveState.calibration && liveState.calibration.state === 'STAGE1_DONE') {
      try {
        await fetch('/api/calibration/cancel', { method: 'POST' });
      } catch (_) {}
    }
  });

  btnStartCalStage1?.addEventListener('click', async () => {
    try {
      await fetch('/api/calibration/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stage: 1, prep_seconds: 5, capture_seconds: 15 }),
      });
    } catch (e) {
      console.error('Erro ao iniciar Etapa 1:', e);
    }
  });

  btnStartCalStage2?.addEventListener('click', async () => {
    try {
      await fetch('/api/calibration/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ stage: 2, prep_seconds: 2, capture_seconds: 20 }),
      });
    } catch (e) {
      console.error('Erro ao iniciar Etapa 2:', e);
    }
  });

  // ==========================================================================
  // 6. Sparkline em Tempo Real (30s M(t) + Limiar theta_on)
  // ==========================================================================
  function drawSparkline() {
    const dpr = window.devicePixelRatio || 1;
    const rect = sparklineCanvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    if (sparklineCanvas.width !== Math.round(rect.width * dpr) ||
        sparklineCanvas.height !== Math.round(rect.height * dpr)) {
      sparklineCanvas.width = Math.round(rect.width * dpr);
      sparklineCanvas.height = Math.round(rect.height * dpr);
    }

    const ctx = sparklineCtx;
    ctx.save();
    ctx.scale(dpr, dpr);
    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    if (mvsSeries.length < 2) {
      ctx.restore();
      return;
    }

    const latestTheta = mvsSeries[mvsSeries.length - 1].theta || 2.5e-6;
    const maxData = mvsSeries.reduce((m, p) => Math.max(m, p.mvs), latestTheta * 2.2);
    const yMax = Math.max(maxData * 1.15, latestTheta * 1.8, 1e-7);

    const thetaY = h - Math.min(h - 6, Math.max(6, (latestTheta / yMax) * (h - 12)));
    ctx.save();
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.55)';
    ctx.lineWidth = 1.1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, thetaY);
    ctx.lineTo(w, thetaY);
    ctx.stroke();
    ctx.restore();

    const occ = liveState ? liveState.occupancy_state : 'EMPTY';
    const theme = STATE_THEME[occ] || STATE_THEME.EMPTY;

    ctx.beginPath();
    for (let i = 0; i < mvsSeries.length; i++) {
      const x = (i / (SPARKLINE_MAX_POINTS - 1)) * w;
      const val = mvsSeries[i].mvs;
      const y = h - Math.min(h - 4, Math.max(4, (val / yMax) * (h - 10)));
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = theme.hex;
    ctx.lineWidth = 2;
    ctx.stroke();

    const lastX = ((mvsSeries.length - 1) / (SPARKLINE_MAX_POINTS - 1)) * w;
    ctx.lineTo(lastX, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, `rgba(${theme.rgb}, 0.28)`);
    grad.addColorStop(1, `rgba(${theme.rgb}, 0.0)`);
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.restore();
  }

  // ==========================================================================
  // 7. Hero Spatial Canvas a 60 FPS (RTI + Incerteza 1-Link + Multi-Kalman)
  // ==========================================================================
  let viewportBox = { x: 40, y: 24, w: 600, h: 400 };

  function normToCanvas(nx, ny, box = viewportBox) {
    return {
      x: box.x + nx * box.w,
      y: box.y + ny * box.h,
    };
  }

  function drawFresnelEllipse(ctx, txPt, rxPt, intensity, stateColorRgb, isOnline) {
    const cx = (txPt.x + rxPt.x) * 0.5;
    const cy = (txPt.y + rxPt.y) * 0.5;
    const dx = rxPt.x - txPt.x;
    const dy = rxPt.y - txPt.y;
    const dist = Math.hypot(dx, dy);
    const angle = Math.atan2(dy, dx);

    const rxMajor = dist * 0.53;
    const ryMinor = Math.max(24, dist * 0.19);

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(angle);

    const normInt = Math.min(1.0, intensity / 4.5);
    const fillAlpha = isOnline ? (0.03 + normInt * 0.13) : 0.01;
    const strokeAlpha = isOnline ? (0.16 + normInt * 0.44) : 0.06;

    const grad = ctx.createRadialGradient(0, 0, 4, 0, 0, rxMajor);
    grad.addColorStop(0, `rgba(${stateColorRgb}, ${fillAlpha * 1.45})`);
    grad.addColorStop(1, `rgba(${stateColorRgb}, 0.0)`);

    ctx.beginPath();
    ctx.ellipse(0, 0, rxMajor, ryMinor, 0, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.lineWidth = 1.1 + normInt;
    ctx.strokeStyle = `rgba(${stateColorRgb}, ${strokeAlpha})`;
    ctx.stroke();

    ctx.beginPath();
    ctx.setLineDash([3, 6]);
    ctx.moveTo(-dist * 0.5, 0);
    ctx.lineTo(dist * 0.5, 0);
    ctx.strokeStyle = `rgba(255, 255, 255, ${isOnline ? 0.14 : 0.05})`;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();
  }

  function renderSpatialFrame() {
    renderState.phase += 0.035;

    const dpr = window.devicePixelRatio || 1;
    const rect = spatialCanvas.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) {
      if (spatialCanvas.width !== Math.round(rect.width * dpr) ||
          spatialCanvas.height !== Math.round(rect.height * dpr)) {
        spatialCanvas.width = Math.round(rect.width * dpr);
        spatialCanvas.height = Math.round(rect.height * dpr);
      }

      const ctx = spatialCtx;
      ctx.save();
      ctx.scale(dpr, dpr);
      const W = rect.width;
      const H = rect.height;
      ctx.clearRect(0, 0, W, H);

      const roomW = roomGeometry.width_m || 5.0;
      const roomL = roomGeometry.length_m || 4.5;
      const padX = 48;
      const padY = 38;
      const availW = W - padX * 2;
      const availH = H - padY * 2;
      const aspect = roomW / roomL;
      let boxW = availW;
      let boxH = boxW / aspect;
      if (boxH > availH) {
        boxH = availH;
        boxW = boxH * aspect;
      }
      viewportBox = { x: (W - boxW) * 0.5, y: (H - boxH) * 0.5, w: boxW, h: boxH };

      // 1. Moldura + Grade Métrica de 1.0m
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.14)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(viewportBox.x, viewportBox.y, viewportBox.w, viewportBox.h, 16);
      ctx.stroke();

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.045)';
      for (let xm = 1.0; xm < roomW - 0.05; xm += 1.0) {
        const gx = viewportBox.x + (xm / roomW) * viewportBox.w;
        ctx.beginPath();
        ctx.moveTo(gx, viewportBox.y);
        ctx.lineTo(gx, viewportBox.y + viewportBox.h);
        ctx.stroke();
      }
      for (let ym = 1.0; ym < roomL - 0.05; ym += 1.0) {
        const gy = viewportBox.y + (ym / roomL) * viewportBox.h;
        ctx.beginPath();
        ctx.moveTo(viewportBox.x, gy);
        ctx.lineTo(viewportBox.x + viewportBox.w, gy);
        ctx.stroke();
      }

      ctx.fillStyle = 'rgba(148, 163, 184, 0.55)';
      ctx.font = '500 9.5px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText('(0,0)', viewportBox.x + 6, viewportBox.y - 6);
      ctx.textAlign = 'right';
      ctx.fillText(`${roomW.toFixed(2)} m`, viewportBox.x + viewportBox.w - 4, viewportBox.y - 6);
      ctx.textAlign = 'left';
      ctx.fillText(`${roomL.toFixed(2)} m`, viewportBox.x + 6, viewportBox.y + viewportBox.h + 13);
      ctx.restore();

      const occ = liveState ? liveState.occupancy_state : 'EMPTY';
      const theme = STATE_THEME[occ] || STATE_THEME.EMPTY;
      const uncertaintyMode = liveState ? liveState.uncertainty_mode : 'NONE';

      const targetInt = liveState ? (liveState.intensity || 0) : 0;
      renderState.intensity += (targetInt - renderState.intensity) * 0.14;

      const txPt = normToCanvas(roomGeometry.tx.x, roomGeometry.tx.y);
      const nodesList = (liveState && Array.isArray(liveState.nodes)) ? liveState.nodes : [];

      // 2. Elipses de Zona de Fresnel por Link
      roomGeometry.rx_nodes.forEach((rx) => {
        const liveNode = nodesList.find((n) => n.id === rx.id);
        const isOnline = liveNode ? liveNode.online : true;
        const rawNodeInt = (liveNode && isOnline) ? (liveNode.intensity || 0) : 0;
        const prevInt = renderState.nodeIntensities[rx.id] || 0;
        const smoothNodeInt = prevInt + (rawNodeInt - prevInt) * 0.15;
        renderState.nodeIntensities[rx.id] = smoothNodeInt;

        if (isOnline) {
          const rxPt = normToCanvas(rx.x, rx.y);
          drawFresnelEllipse(ctx, txPt, rxPt, smoothNodeInt, theme.rgb, isOnline);
        }
      });

      // 3. Incerteza Espacial de 1 Link (Halo Elíptico) vs Alvos Multi-Objeto Kalman 2D
      if (occ !== 'EMPTY') {
        if (uncertaintyMode === 'FRESNEL_HALO') {
          const activeRx = roomGeometry.rx_nodes.find((r) => {
            const ln = nodesList.find((n) => n.id === r.id);
            return ln && ln.online;
          }) || roomGeometry.rx_nodes[0];
          const rxPt = normToCanvas(activeRx.x, activeRx.y);
          const cx = (txPt.x + rxPt.x) * 0.5;
          const cy = (txPt.y + rxPt.y) * 0.5;
          const dx = rxPt.x - txPt.x;
          const dy = rxPt.y - txPt.y;
          const dist = Math.hypot(dx, dy);
          const angle = Math.atan2(dy, dx);
          const pulse = Math.sin(renderState.phase * 1.4) * 5;

          ctx.save();
          ctx.translate(cx, cy);
          ctx.rotate(angle);
          const haloGrad = ctx.createRadialGradient(0, 0, 6, 0, 0, dist * 0.45);
          haloGrad.addColorStop(0, `rgba(${theme.rgb}, 0.34)`);
          haloGrad.addColorStop(1, `rgba(${theme.rgb}, 0.0)`);
          ctx.fillStyle = haloGrad;
          ctx.beginPath();
          ctx.ellipse(0, 0, dist * 0.44 + pulse, Math.max(34, dist * 0.20) + pulse * 0.5, 0, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = `rgba(${theme.rgb}, 0.45)`;
          ctx.setLineDash([5, 5]);
          ctx.lineWidth = 1.4;
          ctx.stroke();
          ctx.restore();
        } else if (liveState && Array.isArray(liveState.targets) && liveState.targets.length > 0) {
          const activeIds = new Set();
          liveState.targets.forEach((t, idx) => {
            activeIds.add(t.id);
            const palette = TARGET_COLORS[idx % TARGET_COLORS.length];
            if (!renderState.targetsMap[t.id]) {
              renderState.targetsMap[t.id] = { x: t.x, y: t.y, trail: [] };
            }
            const rt = renderState.targetsMap[t.id];
            rt.x += (t.x - rt.x) * 0.20;
            rt.y += (t.y - rt.y) * 0.20;

            const pt = normToCanvas(rt.x, rt.y);
            const lastTrail = rt.trail[rt.trail.length - 1];
            if (!lastTrail || Math.hypot(pt.x - lastTrail.x, pt.y - lastTrail.y) > 3) {
              rt.trail.push({ x: pt.x, y: pt.y });
              if (rt.trail.length > 22) rt.trail.shift();
            }

            if (rt.trail.length > 1) {
              ctx.beginPath();
              rt.trail.forEach((p, pIdx) => {
                if (pIdx === 0) ctx.moveTo(p.x, p.y);
                else ctx.lineTo(p.x, p.y);
              });
              ctx.strokeStyle = `rgba(${palette.rgb}, 0.34)`;
              ctx.lineWidth = 2;
              ctx.stroke();
            }

            const pulseR = 25 + Math.sin(renderState.phase * 1.8 + idx) * 3.5;
            const tGrad = ctx.createRadialGradient(pt.x, pt.y, 2, pt.x, pt.y, pulseR);
            tGrad.addColorStop(0, `rgba(${palette.rgb}, 0.56)`);
            tGrad.addColorStop(1, `rgba(${palette.rgb}, 0.0)`);
            ctx.fillStyle = tGrad;
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, pulseR, 0, Math.PI * 2);
            ctx.fill();

            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 6.5, 0, Math.PI * 2);
            ctx.fillStyle = t.state === 'PRESENCE_STATIC' ? '#10B981' : palette.hex;
            ctx.fill();
            ctx.lineWidth = 2;
            ctx.strokeStyle = '#FFFFFF';
            ctx.stroke();

            const xm = (rt.x * roomW).toFixed(1);
            const ym = (rt.y * roomL).toFixed(1);
            ctx.fillStyle = '#FFFFFF';
            ctx.font = '600 10px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(`${t.id} (${xm}m, ${ym}m)`, pt.x, pt.y - 12);
          });

          Object.keys(renderState.targetsMap).forEach((k) => {
            if (!activeIds.has(k)) delete renderState.targetsMap[k];
          });
        }
      } else {
        renderState.targetsMap = {};
      }

      // 4. Roteador Cisco Linksys E1200 (TX)
      const txXm = (roomGeometry.tx.x * roomW).toFixed(2);
      const txYm = (roomGeometry.tx.y * roomL).toFixed(2);
      ctx.save();
      ctx.beginPath();
      ctx.arc(txPt.x, txPt.y, 11, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
      ctx.fill();
      ctx.lineWidth = 1.6;
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.88)';
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(txPt.x, txPt.y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = '#38BDF8';
      ctx.fill();
      ctx.fillStyle = 'rgba(241, 245, 249, 0.90)';
      ctx.font = '500 10px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Linksys E1200 (TX)', txPt.x, txPt.y - 23);
      ctx.fillStyle = 'rgba(125, 211, 252, 0.78)';
      ctx.font = '500 9px Inter, sans-serif';
      ctx.fillText(`(${txXm}m, ${txYm}m)`, txPt.x, txPt.y - 13);
      ctx.restore();

      // 5. Nós ESP32-S3 (RX)
      roomGeometry.rx_nodes.forEach((rx) => {
        const liveNode = nodesList.find((n) => n.id === rx.id);
        const isOnline = liveNode ? liveNode.online : true;
        const rxPt = normToCanvas(rx.x, rx.y);
        const rxXm = rx.x * roomW;
        const rxYm = rx.y * roomL;
        const distM = Math.hypot(rxXm - roomGeometry.tx.x * roomW, rxYm - roomGeometry.tx.y * roomL);

        ctx.save();
        ctx.beginPath();
        ctx.roundRect(rxPt.x - 9, rxPt.y - 9, 18, 18, 5);
        ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
        ctx.fill();
        ctx.lineWidth = 1.4;
        ctx.strokeStyle = isOnline ? `rgba(${theme.rgb}, 0.85)` : 'rgba(148, 163, 184, 0.25)';
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(rxPt.x, rxPt.y, 3, 0, Math.PI * 2);
        ctx.fillStyle = isOnline ? theme.hex : '#475569';
        ctx.fill();

        ctx.fillStyle = isOnline ? 'rgba(241, 245, 249, 0.88)' : 'rgba(148, 163, 184, 0.4)';
        ctx.font = '500 10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(rx.name, rxPt.x, rxPt.y + 21);
        ctx.fillStyle = isOnline ? 'rgba(148, 163, 184, 0.75)' : 'rgba(148, 163, 184, 0.35)';
        ctx.font = '500 9px Inter, sans-serif';
        ctx.fillText(`(${rxXm.toFixed(2)}m, ${rxYm.toFixed(2)}m) • ${distM.toFixed(1)}m`, rxPt.x, rxPt.y + 32);
        ctx.restore();
      });

      ctx.restore();
    }

    if (layoutModalOpen) {
      renderLayoutEditorCanvas();
    }
    requestAnimationFrame(renderSpatialFrame);
  }

  // ==========================================================================
  // 8. Editor Visual de Layout da Sala (Drag & Drop + Snap 0.25m + Avisos RF)
  // ==========================================================================
  let editorBox = { x: 36, y: 28, w: 480, h: 320 };
  let editorDraggingId = null;

  function computeLocalGeometryWarnings(cfg) {
    const warnings = [];
    const w = cfg.width_m || 5.0;
    const l = cfg.length_m || 4.5;
    const txX = cfg.tx.x * w;
    const txY = cfg.tx.y * l;
    const unitVecs = [];

    cfg.rx_nodes.forEach((rx) => {
      const rxX = rx.x * w;
      const rxY = rx.y * l;
      const dx = rxX - txX;
      const dy = rxY - txY;
      const dist = Math.hypot(dx, dy);
      const safe = Math.max(0.2, dist);
      unitVecs.push({ rx, ux: dx / safe, uy: dy / safe, dist });

      if (dist < 1.50) {
        warnings.push(`${rx.name}: distância curta (${dist.toFixed(2)} m < 1.5 m) — risco de saturação direta.`);
      } else if (dist > 7.00) {
        warnings.push(`${rx.name}: distância longa (${dist.toFixed(2)} m > 7.0 m) — menor sensibilidade.`);
      }
    });

    for (let i = 0; i < unitVecs.length; i++) {
      for (let j = i + 1; j < unitVecs.length; j++) {
        const a = unitVecs[i];
        const b = unitVecs[j];
        const cross = Math.abs(a.ux * b.uy - a.uy * b.ux);
        const dot = a.ux * b.ux + a.uy * b.uy;
        if (cross < 0.18 && dot > 0) {
          warnings.push(`${a.rx.name} e ${b.rx.name} estão quase colineares em relação ao roteador.`);
        }
      }
    }
    return warnings;
  }

  function updateLayoutWarningsUI(customWarnings = null) {
    const msgs = customWarnings
      ? customWarnings.map((w) => (typeof w === 'string' ? w : w.message))
      : computeLocalGeometryWarnings(draftGeometry);

    const nx = Math.max(8, Math.round((draftGeometry.width_m || 5.0) / 0.25));
    const ny = Math.max(8, Math.round((draftGeometry.length_m || 4.5) / 0.25));
    layoutGridInfo.textContent = `Grade RTI: ${nx} × ${ny} células (0.25 m/célula)`;
    if (drawerRoomSummary) {
      drawerRoomSummary.textContent = `Sala ${(roomGeometry.width_m || 5).toFixed(2)}m × ${(roomGeometry.length_m || 4.5).toFixed(2)}m • Grid 0.25m`;
    }

    if (!msgs.length) {
      layoutWarningsBox.className = 'liquid-glass-inner px-3 py-2 text-[11px] text-emerald-300 flex flex-col gap-1';
      layoutWarningsBox.innerHTML = '<span>✓ Geometria RF ideal: distâncias entre 1.5 m e 7.0 m e boa separação angular para RTI 2D.</span>';
    } else {
      layoutWarningsBox.className = 'liquid-glass-inner px-3 py-2 text-[11px] text-amber-300 flex flex-col gap-1 border border-amber-400/30';
      layoutWarningsBox.innerHTML = msgs.map((m) => `<span>⚠ ${m}</span>`).join('');
    }
  }

  function buildLayoutSideInputs() {
    const w = draftGeometry.width_m || 5.0;
    const l = draftGeometry.length_m || 4.5;
    inpRoomW.value = w.toFixed(2);
    inpRoomL.value = l.toFixed(2);

    const allNodes = [
      { ref: draftGeometry.tx, isTx: true },
      ...draftGeometry.rx_nodes.map((r) => ({ ref: r, isTx: false })),
    ];

    layoutNodesInputs.innerHTML = allNodes
      .map(({ ref, isTx }) => {
        const xm = (ref.x * w).toFixed(2);
        const ym = (ref.y * l).toFixed(2);
        const distTxt = isTx
          ? 'TX Origem'
          : `${Math.hypot(ref.x * w - draftGeometry.tx.x * w, ref.y * l - draftGeometry.tx.y * l).toFixed(2)} m`;
        const badgeColor = isTx ? 'text-sky-300' : 'text-emerald-300';

        return `
          <div class="p-2 rounded-xl bg-black/30 border border-white/10 flex flex-col gap-1.5" data-node-card="${ref.id}">
            <div class="flex items-center justify-between text-[11px]">
              <strong class="text-white truncate">${ref.name} <span class="text-slate-400 font-normal">(${ref.id})</span></strong>
              <span class="tabular-nums font-semibold ${badgeColor}" data-dist-for="${ref.id}">${distTxt}</span>
            </div>
            <div class="grid grid-cols-12 gap-1.5 items-center text-[11px]">
              <label class="col-span-3 flex items-center gap-1 text-slate-300">
                <span>X:</span>
                <input type="number" step="0.05" min="0" max="30" value="${xm}" data-coord-x="${ref.id}"
                  class="w-full bg-black/45 border border-white/15 rounded px-1.5 py-1 text-white tabular-nums text-center focus:outline-none focus:border-sky-400/70" />
              </label>
              <label class="col-span-3 flex items-center gap-1 text-slate-300">
                <span>Y:</span>
                <input type="number" step="0.05" min="0" max="30" value="${ym}" data-coord-y="${ref.id}"
                  class="w-full bg-black/45 border border-white/15 rounded px-1.5 py-1 text-white tabular-nums text-center focus:outline-none focus:border-sky-400/70" />
              </label>
              <label class="col-span-6 flex items-center gap-1 text-slate-300">
                <span>MAC:</span>
                <input type="text" value="${ref.mac || ''}" placeholder="aa:bb:cc:dd:ee:ff" data-mac="${ref.id}"
                  class="w-full bg-black/45 border border-white/15 rounded px-1.5 py-1 text-[10px] text-slate-200 font-mono focus:outline-none focus:border-sky-400/70" />
              </label>
            </div>
          </div>
        `;
      })
      .join('');

    layoutNodesInputs.querySelectorAll('input').forEach((inp) => {
      inp.addEventListener('input', onSideInputEdited);
    });
    updateLayoutWarningsUI();
  }

  function syncSideInputsNumbersOnly() {
    const w = draftGeometry.width_m || 5.0;
    const l = draftGeometry.length_m || 4.5;
    const updateOne = (ref, isTx) => {
      const ix = layoutNodesInputs.querySelector(`[data-coord-x="${ref.id}"]`);
      const iy = layoutNodesInputs.querySelector(`[data-coord-y="${ref.id}"]`);
      const idist = layoutNodesInputs.querySelector(`[data-dist-for="${ref.id}"]`);
      if (ix && document.activeElement !== ix) ix.value = (ref.x * w).toFixed(2);
      if (iy && document.activeElement !== iy) iy.value = (ref.y * l).toFixed(2);
      if (idist && !isTx) {
        const d = Math.hypot(ref.x * w - draftGeometry.tx.x * w, ref.y * l - draftGeometry.tx.y * l);
        idist.textContent = `${d.toFixed(2)} m`;
        idist.className = (d >= 1.5 && d <= 7.0)
          ? 'tabular-nums font-semibold text-emerald-300'
          : 'tabular-nums font-semibold text-amber-300';
      }
    };
    updateOne(draftGeometry.tx, true);
    draftGeometry.rx_nodes.forEach((r) => updateOne(r, false));
    updateLayoutWarningsUI();
  }

  function clampNorm(valM, dimM) {
    return Number(Math.max(0.02, Math.min(0.98, valM / Math.max(1.5, dimM))).toFixed(4));
  }

  function onSideInputEdited() {
    const w = Math.max(1.5, Math.min(25.0, parseFloat(inpRoomW.value) || 5.0));
    const l = Math.max(1.5, Math.min(25.0, parseFloat(inpRoomL.value) || 4.5));
    draftGeometry.width_m = Number(w.toFixed(2));
    draftGeometry.length_m = Number(l.toFixed(2));

    const applyRef = (ref) => {
      const ix = layoutNodesInputs.querySelector(`[data-coord-x="${ref.id}"]`);
      const iy = layoutNodesInputs.querySelector(`[data-coord-y="${ref.id}"]`);
      const imac = layoutNodesInputs.querySelector(`[data-mac="${ref.id}"]`);
      if (ix) ref.x = clampNorm(parseFloat(ix.value) || 0, w);
      if (iy) ref.y = clampNorm(parseFloat(iy.value) || 0, l);
      if (imac) ref.mac = imac.value.trim();
    };
    applyRef(draftGeometry.tx);
    draftGeometry.rx_nodes.forEach(applyRef);
    syncSideInputsNumbersOnly();
  }

  inpRoomW?.addEventListener('input', onSideInputEdited);
  inpRoomL?.addEventListener('input', onSideInputEdited);

  function renderLayoutEditorCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = layoutEditorCanvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    if (layoutEditorCanvas.width !== Math.round(rect.width * dpr) ||
        layoutEditorCanvas.height !== Math.round(rect.height * dpr)) {
      layoutEditorCanvas.width = Math.round(rect.width * dpr);
      layoutEditorCanvas.height = Math.round(rect.height * dpr);
    }

    const ctx = layoutEditorCtx;
    ctx.save();
    ctx.scale(dpr, dpr);
    const W = rect.width;
    const H = rect.height;
    ctx.clearRect(0, 0, W, H);

    const wM = draftGeometry.width_m || 5.0;
    const lM = draftGeometry.length_m || 4.5;
    const padX = 38;
    const padY = 28;
    const aspect = wM / lM;
    let bw = W - padX * 2;
    let bh = bw / aspect;
    if (bh > H - padY * 2) {
      bh = H - padY * 2;
      bw = bh * aspect;
    }
    editorBox = { x: (W - bw) * 0.5, y: (H - bh) * 0.5, w: bw, h: bh };

    // Grade fina de 0.25m + Grade principal de 1.0m com régua
    ctx.save();
    for (let xm = 0.25; xm < wM - 0.01; xm += 0.25) {
      const gx = editorBox.x + (xm / wM) * editorBox.w;
      const isMajor = Math.abs(xm - Math.round(xm)) < 0.02;
      ctx.strokeStyle = isMajor ? 'rgba(255, 255, 255, 0.12)' : 'rgba(255, 255, 255, 0.035)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(gx, editorBox.y);
      ctx.lineTo(gx, editorBox.y + editorBox.h);
      ctx.stroke();

      if (isMajor) {
        ctx.fillStyle = 'rgba(148, 163, 184, 0.70)';
        ctx.font = '500 9px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(`${Math.round(xm)}m`, gx, editorBox.y - 6);
      }
    }

    for (let ym = 0.25; ym < lM - 0.01; ym += 0.25) {
      const gy = editorBox.y + (ym / lM) * editorBox.h;
      const isMajor = Math.abs(ym - Math.round(ym)) < 0.02;
      ctx.strokeStyle = isMajor ? 'rgba(255, 255, 255, 0.12)' : 'rgba(255, 255, 255, 0.035)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(editorBox.x, gy);
      ctx.lineTo(editorBox.x + editorBox.w, gy);
      ctx.stroke();

      if (isMajor) {
        ctx.fillStyle = 'rgba(148, 163, 184, 0.70)';
        ctx.font = '500 9px Inter, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(`${Math.round(ym)}m`, editorBox.x - 6, gy + 3);
      }
    }

    ctx.strokeStyle = 'rgba(125, 211, 252, 0.45)';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.roundRect(editorBox.x, editorBox.y, editorBox.w, editorBox.h, 10);
    ctx.stroke();
    ctx.restore();

    const txPt = normToCanvas(draftGeometry.tx.x, draftGeometry.tx.y, editorBox);

    // Elipses de Fresnel em tempo real durante edição/drag
    draftGeometry.rx_nodes.forEach((rx) => {
      const rxPt = normToCanvas(rx.x, rx.y, editorBox);
      const distM = Math.hypot(rx.x * wM - draftGeometry.tx.x * wM, rx.y * lM - draftGeometry.tx.y * lM);
      const validDist = distM >= 1.5 && distM <= 7.0;
      const rgb = validDist ? '16, 185, 129' : '245, 158, 11';

      const cx = (txPt.x + rxPt.x) * 0.5;
      const cy = (txPt.y + rxPt.y) * 0.5;
      const pixDist = Math.hypot(rxPt.x - txPt.x, rxPt.y - txPt.y);
      const angle = Math.atan2(rxPt.y - txPt.y, rxPt.x - txPt.x);
      // Raio físico da 1ª Zona de Fresnel ~ sqrt(lambda * d / 4) convertido em pixels
      const fresnelR_m = Math.sqrt(0.122 * Math.max(0.5, distM) / 4.0);
      const pxPerM = editorBox.w / wM;
      const ryPix = Math.max(14, fresnelR_m * pxPerM);

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(angle);
      ctx.fillStyle = `rgba(${rgb}, 0.10)`;
      ctx.strokeStyle = `rgba(${rgb}, 0.52)`;
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.ellipse(0, 0, pixDist * 0.5, ryPix, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(-pixDist * 0.5, 0);
      ctx.lineTo(pixDist * 0.5, 0);
      ctx.strokeStyle = `rgba(${rgb}, 0.75)`;
      ctx.stroke();
      ctx.restore();

      ctx.save();
      ctx.fillStyle = validDist ? '#6EE7B7' : '#FCD34D';
      ctx.font = '600 9.5px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`${distM.toFixed(2)} m`, cx, cy - 4);
      ctx.restore();
    });

    // Desenha TX
    ctx.save();
    ctx.beginPath();
    ctx.arc(txPt.x, txPt.y, 12, 0, Math.PI * 2);
    ctx.fillStyle = '#0F172A';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#38BDF8';
    ctx.stroke();
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '700 9px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('TX', txPt.x, txPt.y + 3);
    ctx.fillStyle = '#BAE6FD';
    ctx.font = '600 10px Inter, sans-serif';
    ctx.fillText(`Linksys (${(draftGeometry.tx.x * wM).toFixed(2)}m, ${(draftGeometry.tx.y * lM).toFixed(2)}m)`, txPt.x, txPt.y - 16);
    ctx.restore();

    // Desenha RXs
    draftGeometry.rx_nodes.forEach((rx, idx) => {
      const rxPt = normToCanvas(rx.x, rx.y, editorBox);
      ctx.save();
      ctx.beginPath();
      ctx.roundRect(rxPt.x - 11, rxPt.y - 11, 22, 22, 6);
      ctx.fillStyle = '#0F172A';
      ctx.fill();
      ctx.lineWidth = 1.8;
      ctx.strokeStyle = '#10B981';
      ctx.stroke();
      ctx.fillStyle = '#FFFFFF';
      ctx.font = '700 9px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`#${idx + 1}`, rxPt.x, rxPt.y + 3);
      ctx.fillStyle = '#E2E8F0';
      ctx.font = '600 9.5px Inter, sans-serif';
      ctx.fillText(`${rx.name} (${(rx.x * wM).toFixed(2)}m, ${(rx.y * lM).toFixed(2)}m)`, rxPt.x, rxPt.y + 23);
      ctx.restore();
    });

    ctx.restore();
  }

  function findEditorNodeAt(cx, cy) {
    const txPt = normToCanvas(draftGeometry.tx.x, draftGeometry.tx.y, editorBox);
    if (Math.hypot(cx - txPt.x, cy - txPt.y) <= 18) return 'tx_router';
    for (const rx of draftGeometry.rx_nodes) {
      const rxPt = normToCanvas(rx.x, rx.y, editorBox);
      if (Math.hypot(cx - rxPt.x, cy - rxPt.y) <= 18) return rx.id;
    }
    return null;
  }

  layoutEditorCanvas?.addEventListener('mousedown', (e) => {
    const rect = layoutEditorCanvas.getBoundingClientRect();
    editorDraggingId = findEditorNodeAt(e.clientX - rect.left, e.clientY - rect.top);
  });

  window.addEventListener('mousemove', (e) => {
    if (!editorDraggingId || !layoutModalOpen || !layoutEditorCanvas) return;
    const rect = layoutEditorCanvas.getBoundingClientRect();
    const wM = draftGeometry.width_m || 5.0;
    const lM = draftGeometry.length_m || 4.5;
    let xm = ((e.clientX - rect.left - editorBox.x) / Math.max(1, editorBox.w)) * wM;
    let ym = ((e.clientY - rect.top - editorBox.y) / Math.max(1, editorBox.h)) * lM;

    if (chkSnap025 && chkSnap025.checked) {
      xm = Math.round(xm / 0.25) * 0.25;
      ym = Math.round(ym / 0.25) * 0.25;
    }

    const nx = clampNorm(xm, wM);
    const ny = clampNorm(ym, lM);
    if (editorDraggingId === 'tx_router') {
      draftGeometry.tx.x = nx;
      draftGeometry.tx.y = ny;
    } else {
      const rx = draftGeometry.rx_nodes.find((r) => r.id === editorDraggingId);
      if (rx) {
        rx.x = nx;
        rx.y = ny;
      }
    }
    syncSideInputsNumbersOnly();
  });

  window.addEventListener('mouseup', () => {
    editorDraggingId = null;
  });

  function openLayoutModal() {
    draftGeometry = JSON.parse(JSON.stringify(roomGeometry));
    layoutModalOpen = true;
    layoutEditorModal?.classList.add('open');
    buildLayoutSideInputs();
  }

  function closeLayoutModal() {
    layoutModalOpen = false;
    editorDraggingId = null;
    layoutEditorModal?.classList.remove('open');
  }

  btnOpenLayoutEditor?.addEventListener('click', openLayoutModal);
  btnDrawerOpenLayout?.addEventListener('click', () => {
    setDiagnosticsDrawer(false);
    openLayoutModal();
  });
  btnCloseLayoutEditor?.addEventListener('click', closeLayoutModal);

  async function loadRoomGeometry() {
    try {
      const res = await fetch('/api/config/layout');
      if (res.ok) {
        const data = await res.json();
        roomGeometry = data.room_config;
        draftGeometry = JSON.parse(JSON.stringify(roomGeometry));
        updateLayoutWarningsUI(data.warnings || []);
      }
    } catch (e) {
      console.debug('Falha ao carregar layout da sala:', e);
    }
  }

  btnSaveGeometry?.addEventListener('click', async () => {
    onSideInputEdited();
    try {
      const res = await fetch('/api/config/layout', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(draftGeometry),
      });
      if (res.ok) {
        const data = await res.json();
        roomGeometry = data.room_config;
        draftGeometry = JSON.parse(JSON.stringify(roomGeometry));
        buildLayoutSideInputs();
        updateLayoutWarningsUI(data.warnings || []);
        const origText = btnSaveGeometry.textContent;
        btnSaveGeometry.textContent = 'Matriz RTI Atualizada (Hot-Reload) ✓';
        setTimeout(() => { btnSaveGeometry.textContent = origText; }, 1800);
      }
    } catch (e) {
      console.error('Erro ao salvar layout da sala:', e);
    }
  });

  btnResetGeometry?.addEventListener('click', async () => {
    try {
      const res = await fetch('/api/config/room/reset', { method: 'POST' });
      if (res.ok) {
        roomGeometry = await res.json();
        draftGeometry = JSON.parse(JSON.stringify(roomGeometry));
        buildLayoutSideInputs();
      }
    } catch (e) {
      console.debug('Erro ao restaurar layout padrão:', e);
    }
  });

  // ==========================================================================
  // 9. Gaveta de Configurações, Fonte de Dados & Telemetria
  // ==========================================================================
  function setDiagnosticsDrawer(open) {
    diagnosticsOpen = open;
    if (open) {
      drawerBackdrop?.classList.add('open');
      diagnosticsDrawer?.classList.add('open');
      if (ws && ws.readyState === WebSocket.OPEN) ws.send('diagnostics:on');
    } else {
      drawerBackdrop?.classList.remove('open');
      diagnosticsDrawer?.classList.remove('open');
      if (ws && ws.readyState === WebSocket.OPEN) ws.send('diagnostics:off');
    }
  }

  btnOpenDiagnostics?.addEventListener('click', () => setDiagnosticsDrawer(true));
  btnCloseDiagnostics?.addEventListener('click', () => setDiagnosticsDrawer(false));
  drawerBackdrop?.addEventListener('click', () => setDiagnosticsDrawer(false));

  function renderDiagnosticsDrawer(state) {
    if (state.mode === 'simulation') {
      btnSourceSim.className = 'px-2.5 py-1 rounded-md text-[11px] font-medium bg-white/20 text-white transition';
      btnSourceHw.className = 'px-2.5 py-1 rounded-md text-[11px] font-medium text-slate-300 hover:text-white transition';
      drawerSimControls.classList.remove('hidden');
    } else {
      btnSourceHw.className = 'px-2.5 py-1 rounded-md text-[11px] font-medium bg-emerald-500/25 text-emerald-200 transition';
      btnSourceSim.className = 'px-2.5 py-1 rounded-md text-[11px] font-medium text-slate-300 hover:text-white transition';
      drawerSimControls.classList.add('hidden');
    }

    nodeCntBtns.forEach((btn) => {
      const n = parseInt(btn.getAttribute('data-nodes'), 10);
      btn.classList.toggle('active', n === state.active_nodes_count);
    });

    if (state.calibration) {
      diagCalQuality.textContent = state.calibration.quality || 'OPTIMAL';
    }

    const diag = state.diagnostics;
    if (!diag) return;

    if (diagSubspaceSummary) {
      diagSubspaceSummary.textContent =
        `Picos RTI: ${diag.spatial_peaks_count || 0} • Modos Espectrais (λ): ${diag.spectral_modes_count || 0} • Descorrelação: ${(diag.link_decorrelation || 0).toFixed(2)}`;
    }

    const details = Object.values(diag.nodes_detail || {});
    diagNodesList.innerHTML = details
      .map((nd) => {
        const dot = nd.online ? 'bg-emerald-400' : 'bg-slate-600';
        const statusBadge = nd.online
          ? `${nd.pps.toFixed(0)} Hz • G=${(nd.gain_g || 1).toFixed(2)}x • E=${(nd.calibrated_energy || 0).toFixed(2)}`
          : 'OFFLINE';
        return `
          <div class="flex items-center justify-between py-1.5 px-2.5 rounded-xl bg-black/25 border border-white/10">
            <div class="flex items-center gap-2">
              <span class="w-2 h-2 rounded-full ${dot}"></span>
              <span class="font-medium text-slate-100">${nd.name}</span>
              <span class="text-[10px] text-slate-300 uppercase">${nd.state}</span>
            </div>
            <span class="text-[10.5px] text-slate-200">${statusBadge}</span>
          </div>
        `;
      })
      .join('');

    const selectedId = diagNodeSelect.value || 'esp_1';
    const nodeDetail = (diag.nodes_detail && diag.nodes_detail[selectedId]) || details[0];
    if (nodeDetail) {
      drawSubcarrierChart(nodeDetail.amplitudes_norm || [], nodeDetail.nbvi_indices || []);
    }
  }

  function drawSubcarrierChart(amplitudes, nbviIndices) {
    const dpr = window.devicePixelRatio || 1;
    const rect = subcarrierCanvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;

    if (subcarrierCanvas.width !== Math.round(rect.width * dpr) ||
        subcarrierCanvas.height !== Math.round(rect.height * dpr)) {
      subcarrierCanvas.width = Math.round(rect.width * dpr);
      subcarrierCanvas.height = Math.round(rect.height * dpr);
    }

    const ctx = subcarrierCtx;
    ctx.save();
    ctx.scale(dpr, dpr);
    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    if (!amplitudes.length) {
      ctx.restore();
      return;
    }

    const nbviSet = new Set(nbviIndices);
    const n = amplitudes.length;
    const maxVal = Math.max(...amplitudes, 0.22);
    const barW = Math.max(2, (w - (n - 1) * 1.5) / n);

    for (let i = 0; i < n; i++) {
      const val = amplitudes[i];
      const barH = Math.max(3, (val / maxVal) * (h - 10));
      const x = i * (barW + 1.5);
      const y = h - barH;
      ctx.fillStyle = nbviSet.has(i) ? '#38BDF8' : 'rgba(255, 255, 255, 0.20)';
      ctx.beginPath();
      ctx.roundRect(x, y, barW, barH, 2);
      ctx.fill();
    }
    ctx.restore();
  }

  async function sendControlCommand(payload) {
    try {
      await fetch('/api/simulator/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
    } catch (e) {
      console.error('Erro ao enviar controle:', e);
    }
  }

  simPeopleBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const simulated_people = parseInt(btn.getAttribute('data-sim-people'), 10);
      lastSimPeopleRequested = simulated_people;
      syncSimButtonsHighlight(simulated_people);
      sendControlCommand({ mode: 'simulation', simulated_people });
    });
  });

  btnSourceSim?.addEventListener('click', () => sendControlCommand({ mode: 'simulation' }));
  btnSourceHw?.addEventListener('click', () => sendControlCommand({ mode: 'hardware' }));

  nodeCntBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const active_nodes = parseInt(btn.getAttribute('data-nodes'), 10);
      sendControlCommand({ mode: 'simulation', active_nodes });
    });
  });

  btnInjectAgc?.addEventListener('click', () => sendControlCommand({ inject_agc_spike: 4.5 }));

  sliderSensMult?.addEventListener('input', () => {
    if (lblSensMult) lblSensMult.textContent = `${parseFloat(sliderSensMult.value).toFixed(2)}x`;
  });
  sliderSensMult?.addEventListener('change', () => {
    sendControlCommand({ sensitivity_multiplier: parseFloat(sliderSensMult.value) });
  });

  sliderHoldSec?.addEventListener('input', () => {
    if (lblHoldSec) lblHoldSec.textContent = `${parseFloat(sliderHoldSec.value).toFixed(1)} s`;
  });
  sliderHoldSec?.addEventListener('change', () => {
    sendControlCommand({ hold_time_sec: parseFloat(sliderHoldSec.value) });
  });

  // Inicialização
  loadRoomGeometry();
  connectWebSocket();
  checkLlmStatus();
  requestAnimationFrame(renderSpatialFrame);
})();
