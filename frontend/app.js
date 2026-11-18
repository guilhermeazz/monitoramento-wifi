/**
 * Wi-Fi CSI Spatial Radar v3.0 — Dashboard Principal (/)
 * - Hero Counter Gigante (text-8xl) com cores dinâmicas (0=neutro, 1=verde, 2+=ciano/amarelo)
 * - Calibração Inteligente em 2 Etapas (Etapa 1: Ambiente/Plateia + Etapa 2: Referência 1 Pessoa) + Web Audio API
 * - Telemetria de Autovalores PCA (SVD s1, s2, s3), CSI Ratio, Top-18 Subportadoras e Dupla Banda Humana
 * - Auto-Mapeamento da Área de Abrangência e LLM Local via Ollama
 */

(() => {
  let liveState = null;
  let ws = null;
  let animPhase = 0;

  const personTrails = {};

  let audioCtx = null;
  let prevCalState = 'idle';
  let lastBeepSecond = -1;

  const radarCanvas = document.getElementById('radar-canvas');
  const radarCtx = radarCanvas.getContext('2d');
  const heatmapOffscreen = document.createElement('canvas');
  heatmapOffscreen.width = 20;
  heatmapOffscreen.height = 20;
  const heatmapOffCtx = heatmapOffscreen.getContext('2d');

  const csiCanvas = document.getElementById('csi-spectrum-canvas');
  const csiCtx = csiCanvas.getContext('2d');

  // Elementos do Header e Controles
  const wsDot = document.getElementById('ws-indicator-dot');
  const wsStatusText = document.getElementById('ws-status-text');
  const topTopologyPill = document.getElementById('top-topology-pill');
  const kpiPps = document.getElementById('kpi-pps');

  const btnModeSim = document.getElementById('btn-mode-sim');
  const btnModeHw = document.getElementById('btn-mode-hw');
  const simNodesControls = document.getElementById('sim-nodes-controls');
  const simNodesBtns = document.querySelectorAll('.sim-nodes-btn');
  const simPeopleControls = document.getElementById('sim-people-controls');
  const simCountBtns = document.querySelectorAll('.sim-count-btn');
  const sensBtns = document.querySelectorAll('.sens-btn');

  // Elementos do Modal/Banner Gigante de Calibração
  const calGiantModal = document.getElementById('calibration-giant-modal');
  const calGiantBox = document.getElementById('calibration-giant-box');
  const calGiantIcon = document.getElementById('calibration-giant-icon');
  const calGiantTitle = document.getElementById('calibration-giant-title');
  const calGiantSubtitle = document.getElementById('calibration-giant-subtitle');
  const calGiantMetric = document.getElementById('calibration-giant-metric');
  const calGiantBar = document.getElementById('calibration-giant-bar');

  // Elementos do Hero Counter (Topo da Página)
  const heroNumberBox = document.getElementById('hero-number-box');
  const heroPeopleCount = document.getElementById('hero-people-count');
  const heroStatusTitle = document.getElementById('hero-status-title');
  const badgeStep1 = document.getElementById('badge-step1');
  const badgeStep2 = document.getElementById('badge-step2');
  const heroConfidencePct = document.getElementById('hero-confidence-pct');
  const kpiActiveZone = document.getElementById('kpi-active-zone');
  const heroAgitationBadge = document.getElementById('hero-agitation-badge');
  const heroAgitationBar = document.getElementById('hero-agitation-bar');
  const heroCoverageMetric = document.getElementById('hero-coverage-metric');
  const lastCalibratedTime = document.getElementById('last-calibrated-time');
  const heroSvdRatioText = document.getElementById('hero-svd-ratio-text');
  const heroGateBadge = document.getElementById('hero-gate-badge');

  // Botões do Assistente Científico de Calibração (Wizard 3 Estágios + Instant Zero)
  const btnInstantZero = document.getElementById('btn-instant-zero');
  const btnWizardStage1 = document.getElementById('btn-wizard-stage1');
  const btnWizardStage2 = document.getElementById('btn-wizard-stage2');
  const btnWizardStage3 = document.getElementById('btn-wizard-stage3');
  const btnResetNodes = document.getElementById('btn-reset-nodes');

  // Elementos do Inspetor de Telemetria de Sinal CSI & Decisão Matemática
  const inspectorConsensusPill = document.getElementById('inspector-consensus-pill');
  const inspectorDecisionReason = document.getElementById('inspector-decision-reason');
  const inspectorSvdVals = document.getElementById('inspector-svd-vals');
  const inspectorSvdRatio = document.getElementById('inspector-svd-ratio');
  const inspectorEnergyRatio = document.getElementById('inspector-energy-ratio');
  const inspectorNodesTriggered = document.getElementById('inspector-nodes-triggered');
  const inspectorNodesGrid = document.getElementById('inspector-nodes-grid');

  // Elementos do Radar da Área de Abrangência
  const radarSubtitleBadge = document.getElementById('radar-subtitle-badge');
  const coverageBannerText = document.getElementById('coverage-banner-text');
  const noSensorsOverlay = document.getElementById('no-sensors-overlay');
  const quickSimBtns = document.querySelectorAll('.quick-sim-btn');
  const trackedPeopleCards = document.getElementById('tracked-people-cards');
  const nodesTelemetryList = document.getElementById('nodes-telemetry-list');
  const selectCsiNode = document.getElementById('select-csi-node');

  // Card de Diagnóstico de Topologia & Capacidades
  const topoStatusDot = document.getElementById('topo-status-dot');
  const topoModeBadge = document.getElementById('topo-mode-badge');
  const topoPrecisionText = document.getElementById('topo-precision-text');
  const topoCapabilitiesList = document.getElementById('topo-capabilities-list');
  const topoLimitationsList = document.getElementById('topo-limitations-list');
  const topoRecommendationText = document.getElementById('topo-recommendation-text');

  // Elementos de Eventos e IA Local (Ollama)
  const eventsFeedList = document.getElementById('events-feed-list');
  const eventCounterBadge = document.getElementById('event-counter-badge');
  const btnGenerateReport = document.getElementById('btn-generate-report');
  const btnReportText = document.getElementById('btn-report-text');
  const formAskLlm = document.getElementById('form-ask-llm');
  const inputAskLlm = document.getElementById('input-ask-llm');
  const llmOutputContainer = document.getElementById('llm-output-container');
  const llmProviderBadge = document.getElementById('llm-provider-badge');
  const llmStatusDot = document.getElementById('llm-status-dot');
  const selectLlmModel = document.getElementById('select-llm-model');
  const btnRefreshLlm = document.getElementById('btn-refresh-llm');
  const quickAskChips = document.querySelectorAll('.quick-ask-chip');

  let roomViewport = { x: 50, y: 36, w: 740, h: 448 };

  // ==========================================================================
  // 1. Sintetizador Sonoro via Web Audio API (Feedback de Calibração)
  // ==========================================================================

  function ensureAudioContext() {
    try {
      if (!audioCtx) {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (AudioContextClass) {
          audioCtx = new AudioContextClass();
        }
      }
      if (audioCtx && audioCtx.state === 'suspended') {
        audioCtx.resume();
      }
    } catch (e) {
      console.debug('Web Audio API indisponível:', e);
    }
  }

  function playTone(freqHz, durationSec, type = 'sine', volume = 0.24, delaySec = 0) {
    ensureAudioContext();
    if (!audioCtx) return;
    try {
      const start = audioCtx.currentTime + delaySec;
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freqHz, start);

      gain.gain.setValueAtTime(0.001, start);
      gain.gain.exponentialRampToValueAtTime(volume, start + 0.015);
      gain.gain.exponentialRampToValueAtTime(0.001, start + durationSec);

      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(start);
      osc.stop(start + durationSec + 0.02);
    } catch (e) {
      console.debug('Falha ao emitir som:', e);
    }
  }

  function playCountdownBeep(secRemaining) {
    const freq = secRemaining <= 3 ? 880 : 660;
    playTone(freq, 0.12, 'sine', 0.26, 0);
  }

  function playCalibratingStartBeep() {
    playTone(1046.5, 0.28, 'triangle', 0.32, 0);
  }

  function playCalibrationCompleteDoubleChime() {
    // Duplo bipe alto e claro ao concluir a calibração
    playTone(987.77, 0.24, 'triangle', 0.42, 0.0);
    playTone(1318.51, 0.28, 'triangle', 0.42, 0.26);
    playTone(1567.98, 0.45, 'sine', 0.44, 0.56);
    playTone(1975.53, 0.52, 'triangle', 0.42, 0.84);
  }

  // ==========================================================================
  // 2. Conexão WebSocket (/ws/live) a 10 Hz
  // ==========================================================================

  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      wsDot.className = 'absolute -top-1 -right-1 w-3 h-3 rounded-full bg-emerald-400 ring-4 ring-slate-950 animate-pulse';
      wsStatusText.textContent = 'ONLINE • Stream WebSocket 10 Hz Ativo (20Hz Resampled PCA)';
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        liveState = data;
        updateDashboardUI(data);
      } catch (err) {
        console.error('Erro ao processar frame WebSocket:', err);
      }
    };

    ws.onclose = () => {
      wsDot.className = 'absolute -top-1 -right-1 w-3 h-3 rounded-full bg-rose-500 ring-4 ring-slate-950';
      wsStatusText.textContent = 'Reconectando ao servidor CSI...';
      setTimeout(connectWebSocket, 1500);
    };

    ws.onerror = () => {
      ws.close();
    };
  }

  // ==========================================================================
  // 3. Atualização do Hero Counter, Calibração 2 Etapas e Indicadores DOM
  // ==========================================================================

  function updateDashboardUI(state) {
    kpiPps.textContent = (state.total_packets_per_sec || 0).toFixed(1);

    updateHeroCounterUI(state);
    updateCalibrationFlowUI(state);

    if (state.mode === 'simulation') {
      btnModeSim.className = 'px-3 py-1.5 rounded-md text-xs font-semibold transition-all bg-cyan-500 text-slate-950 shadow';
      btnModeHw.className = 'px-3 py-1.5 rounded-md text-xs font-semibold transition-all text-slate-400 hover:text-white';
      simPeopleControls.classList.remove('opacity-40', 'pointer-events-none');
      simNodesControls.classList.remove('opacity-40', 'pointer-events-none');
    } else {
      btnModeHw.className = 'px-3 py-1.5 rounded-md text-xs font-semibold transition-all bg-emerald-500 text-slate-950 shadow';
      btnModeSim.className = 'px-3 py-1.5 rounded-md text-xs font-semibold transition-all text-slate-400 hover:text-white';
      simPeopleControls.classList.add('opacity-40', 'pointer-events-none');
      simNodesControls.classList.add('opacity-40', 'pointer-events-none');
    }

    const simNodes = state.simulated_nodes ?? 1;
    simNodesBtns.forEach((btn) => {
      const nVal = parseInt(btn.getAttribute('data-sim-nodes'), 10);
      if (nVal === simNodes) {
        const activeStyle =
          nVal === 1
            ? 'border-cyan-400 bg-cyan-500/25 text-cyan-200'
            : nVal === 2
            ? 'border-amber-400 bg-amber-500/25 text-amber-200'
            : 'border-emerald-400 bg-emerald-500/25 text-emerald-200';
        btn.className = `sim-nodes-btn px-2 h-7 rounded-lg text-xs font-mono font-bold border transition shadow-sm ${activeStyle}`;
      } else {
        btn.className = 'sim-nodes-btn px-2 h-7 rounded-lg text-xs font-mono font-bold border border-slate-700/80 bg-slate-950 text-slate-400 hover:border-cyan-500/50 hover:text-white transition';
      }
    });

    simCountBtns.forEach((btn) => {
      const count = parseInt(btn.getAttribute('data-sim-count'), 10);
      if (count === state.simulated_people) {
        btn.className = 'sim-count-btn w-7 h-7 rounded-lg text-xs font-mono font-bold border border-cyan-400 bg-cyan-500/25 text-cyan-200 transition shadow-sm';
      } else {
        btn.className = 'sim-count-btn w-7 h-7 rounded-lg text-xs font-mono font-bold border border-slate-700/80 bg-slate-950 text-slate-400 hover:border-cyan-500/50 hover:text-white transition';
      }
    });

    const currentSens = state.sensitivity || 'normal';
    sensBtns.forEach((btn) => {
      const sVal = btn.getAttribute('data-sensitivity');
      if (sVal === currentSens) {
        btn.className = 'sens-btn px-2.5 h-7 rounded-lg text-[11px] font-mono font-bold border border-emerald-400 bg-emerald-500/20 text-emerald-200 transition shadow-sm';
      } else {
        btn.className = 'sens-btn px-2 h-7 rounded-lg text-[11px] font-mono font-semibold border border-slate-700/80 bg-slate-950 text-slate-400 hover:text-white transition';
      }
    });

    if (state.coverage_info) {
      const cov = state.coverage_info;
      coverageBannerText.textContent = cov.description || `Área de Abrangência Ativa: ~${cov.active_area_m2} m² (Distância TX-RX: ${cov.avg_tx_rx_distance_m} m)`;
      heroCoverageMetric.textContent = `~${cov.active_area_m2.toFixed(1)} m² (TX-RX: ${cov.avg_tx_rx_distance_m.toFixed(1)} m)`;
    }

    if (state.topology_info) {
      renderTopologyDiagnostics(state.topology_info);
    }

    const activeIds = new Set();
    (state.people_coordinates || []).forEach((p) => {
      activeIds.add(p.id);
      if (!personTrails[p.id]) {
        personTrails[p.id] = [];
      }
      const trail = personTrails[p.id];
      const lastPt = trail[trail.length - 1];
      if (!lastPt || Math.hypot(lastPt.x - p.x, lastPt.y - p.y) > 0.006) {
        trail.push({ x: p.x, y: p.y });
        if (trail.length > 22) trail.shift();
      }
    });
    Object.keys(personTrails).forEach((id) => {
      if (!activeIds.has(id)) delete personTrails[id];
    });

    renderTrackedPeopleCards(state.people_coordinates || []);
    renderNodesTelemetry(state.nodes_status || {});
    renderTelemetryInspector(state);
    renderEventsFeed(state.recent_events || []);
  }

  function updateHeroCounterUI(state) {
    const count = state.people_count || 0;
    const people = state.people_coordinates || [];
    const movingCount = people.filter((p) => p.activity === 'moving').length;
    heroPeopleCount.textContent = count;
    kpiActiveZone.textContent = state.active_zone || 'Zona de Abrangência Livre';

    const confPct = Math.round((state.overall_confidence || 0.92) * 100);
    heroConfidencePct.textContent = `${confPct}%`;

    if (state.last_calibrated_at) {
      lastCalibratedTime.textContent = `Calibrado às ${state.last_calibrated_at}`;
    }

    // Badges das Etapas 1 e 2 de Calibração
    if (state.step1_calibrated) {
      const mLabel = state.calibration_mode === 'empty_room' ? 'Sala Vazia ✓' : 'Modo Plateia ✓';
      badgeStep1.className = 'text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/40';
      badgeStep1.textContent = `Etapa 1: ${mLabel}`;
    } else {
      badgeStep1.className = 'text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700';
      badgeStep1.textContent = 'Etapa 1: Nominal';
    }

    if (state.step2_calibrated) {
      badgeStep2.className = 'text-[10px] font-mono font-bold px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/40';
      badgeStep2.textContent = 'Etapa 2: Ref. 1 Pessoa PCA ✓';
    } else {
      badgeStep2.className = 'text-[10px] font-mono font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-slate-400 border border-slate-700';
      badgeStep2.textContent = 'Etapa 2: Nominal';
    }

    // Cores dinâmicas do Hero Counter (0 = neutro, 1 = verde esmeralda, 2+ = ciano/amarelo)
    if (state.topology_info && state.topology_info.mode_code === 'NO_SENSORS') {
      heroNumberBox.className = 'min-w-[130px] md:min-w-[150px] h-[120px] md:h-[136px] rounded-3xl bg-slate-900 border-2 border-slate-700 flex items-center justify-center px-4';
      heroPeopleCount.className = 'text-8xl font-mono font-black tracking-tighter text-slate-500 leading-none select-none';
      heroStatusTitle.className = 'text-2xl md:text-3xl font-black tracking-tight text-slate-400 uppercase leading-tight';
      heroStatusTitle.textContent = 'AGUARDANDO SENSORES';
    } else if (count === 0) {
      // 0 = Neutro
      heroNumberBox.className = 'min-w-[130px] md:min-w-[150px] h-[120px] md:h-[136px] rounded-3xl bg-slate-900/90 border-2 border-slate-700 flex items-center justify-center shadow-inner px-4';
      heroPeopleCount.className = 'text-8xl font-mono font-black tracking-tighter text-slate-300 leading-none select-none';
      heroStatusTitle.className = 'text-2xl md:text-3xl font-black tracking-tight text-slate-200 uppercase leading-tight';
      heroStatusTitle.textContent = 'AMBIENTE LIVRE';
    } else if (count === 1) {
      // 1 = Verde Esmeralda
      heroNumberBox.className = 'min-w-[130px] md:min-w-[150px] h-[120px] md:h-[136px] rounded-3xl bg-emerald-500/15 border-2 border-emerald-400 flex items-center justify-center shadow-lg shadow-emerald-500/20 px-4';
      heroPeopleCount.className = 'text-8xl font-mono font-black tracking-tighter text-emerald-400 leading-none select-none';
      heroStatusTitle.className = 'text-2xl md:text-3xl font-black tracking-tight text-emerald-400 uppercase leading-tight';
      heroStatusTitle.textContent = movingCount >= 1 ? '1 PESSOA EM MOVIMENTO' : '1 PESSOA (PARADA)';
    } else {
      // 2+ = Ciano / Amarelo
      const isTwo = count === 2;
      heroNumberBox.className = isTwo
        ? 'min-w-[130px] md:min-w-[150px] h-[120px] md:h-[136px] rounded-3xl bg-cyan-500/15 border-2 border-cyan-400 flex items-center justify-center shadow-lg shadow-cyan-500/25 px-4'
        : 'min-w-[130px] md:min-w-[150px] h-[120px] md:h-[136px] rounded-3xl bg-amber-500/15 border-2 border-amber-400 flex items-center justify-center shadow-lg shadow-amber-500/25 px-4';
      heroPeopleCount.className = isTwo
        ? 'text-8xl font-mono font-black tracking-tighter text-cyan-300 leading-none select-none'
        : 'text-8xl font-mono font-black tracking-tighter text-amber-300 leading-none select-none';
      heroStatusTitle.className = isTwo
        ? 'text-2xl md:text-3xl font-black tracking-tight text-cyan-300 uppercase leading-tight'
        : 'text-2xl md:text-3xl font-black tracking-tight text-amber-300 uppercase leading-tight';

      if (movingCount > 0) {
        heroStatusTitle.textContent = `${count} PESSOAS EM MOVIMENTO`;
      } else {
        heroStatusTitle.textContent = `${count} PESSOAS (PARADAS)`;
      }
    }

    const agScorePct = Math.round((state.agitation_score || 0) * 100);
    const agLevel = state.agitation_level || 'none';
    heroAgitationBar.style.width = `${agScorePct}%`;

    if (agLevel === 'high') {
      heroAgitationBadge.className = 'font-mono font-bold text-[10px] px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/40';
      heroAgitationBadge.textContent = `ALTA (${agScorePct}%)`;
      heroAgitationBar.className = 'bg-rose-500 h-full transition-all duration-150';
    } else if (agLevel === 'moderate') {
      heroAgitationBadge.className = 'font-mono font-bold text-[10px] px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40';
      heroAgitationBadge.textContent = `MODERADA (${agScorePct}%)`;
      heroAgitationBar.className = 'bg-amber-400 h-full transition-all duration-150';
    } else if (agLevel === 'low') {
      heroAgitationBadge.className = 'font-mono font-bold text-[10px] px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/40';
      heroAgitationBadge.textContent = `BAIXA (${agScorePct}%)`;
      heroAgitationBar.className = 'bg-cyan-400 h-full transition-all duration-150';
    } else {
      heroAgitationBadge.className = 'font-mono font-bold text-[10px] px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
      heroAgitationBadge.textContent = `REPOUSO (${agScorePct}%)`;
      heroAgitationBar.className = 'bg-emerald-400 h-full transition-all duration-150';
    }
  }

  function updateCalibrationFlowUI(state) {
    const calState = state.calibration_state || 'idle';
    const calMode = state.calibration_mode || 'crowd_compensated';
    const remSec = state.countdown_remaining || 0;
    const progressPct = Math.round((state.calibration_progress || 0) * 100);

    if (calState === 'countdown') {
      const ceilSec = Math.ceil(remSec);
      if (ceilSec > 0 && ceilSec !== lastBeepSecond) {
        lastBeepSecond = ceilSec;
        playCountdownBeep(ceilSec);
      }
    } else if (calState === 'calibrating' && prevCalState !== 'calibrating') {
      lastBeepSecond = -1;
      playCalibratingStartBeep();
    } else if (calState === 'completed' && prevCalState !== 'completed') {
      lastBeepSecond = -1;
      playCalibrationCompleteDoubleChime();
    }
    prevCalState = calState;

    if (calState === 'idle') {
      calGiantModal.classList.add('hidden');
      return;
    }

    calGiantModal.classList.remove('hidden');

    if (calState === 'countdown') {
      calGiantBox.className = 'rounded-2xl border-2 border-amber-400 bg-amber-950/90 text-amber-100 p-5 shadow-2xl flex flex-col md:flex-row items-center justify-between gap-4';
      calGiantIcon.className = 'w-14 h-14 rounded-2xl bg-amber-500/25 border border-amber-400/50 flex items-center justify-center text-2xl shrink-0 animate-bounce';
      calGiantIcon.textContent = '⏱️';
      calGiantTitle.textContent = `SAIA DA ÁREA DE ALCANCE! Iniciando calibração em ${remSec.toFixed(1)} segundos...`;
      calGiantSubtitle.textContent = 'Deixe a área livre. Um sinal sonoro duplo alto avisará quando você puder voltar.';
      calGiantMetric.textContent = `${remSec.toFixed(1)}s`;
      calGiantBar.className = 'bg-amber-400 h-full rounded-full transition-all duration-100';
      calGiantBar.style.width = '100%';
    } else if (calState === 'calibrating') {
      calGiantBox.className = 'rounded-2xl border-2 border-cyan-400 bg-cyan-950/90 text-cyan-100 p-5 shadow-2xl flex flex-col md:flex-row items-center justify-between gap-4';
      calGiantIcon.className = 'w-14 h-14 rounded-2xl bg-cyan-500/25 border border-cyan-400/50 flex items-center justify-center text-2xl shrink-0 animate-pulse';
      calGiantIcon.textContent = (calMode === 'stage2_1person' || calMode === 'person_ref') ? '🚶' : (calMode === 'stage3_2persons' ? '👥' : '📡');
      if (calMode === 'stage2_1person' || calMode === 'person_ref') {
        calGiantTitle.textContent = `ESTÁGIO 2: CAMINHE NORMALMENTE ENTRE O ROTEADOR E O ESP32 (${progressPct}%)...`;
        calGiantSubtitle.textContent = 'Gravando unit_person_energy e autovalores SVD (λ1/λ2) de 1 pessoa nesta sala.';
      } else if (calMode === 'stage3_2persons') {
        calGiantTitle.textContent = `ESTÁGIO 3: 2 PESSOAS CAMINHANDO SIMULTANEAMENTE (${progressPct}%)...`;
        calGiantSubtitle.textContent = 'Gravando autovalores ortogonais multi-alvo e ratio de referência SVD.';
      } else if (calMode === 'instant_zero') {
        calGiantTitle.textContent = '🧹 FORÇANDO ZERO INSTANTÂNEO...';
        calGiantSubtitle.textContent = 'Piso Z-Score reajustado com 40% de margem no buffer.';
      } else {
        const modeDesc = calMode === 'crowd_compensated' ? 'SALA COM PLATEIA' : 'SALA VAZIA (SILÊNCIO)';
        calGiantTitle.textContent = `CALIBRANDO ASSINATURA WI-FI DO AMBIENTE (${progressPct}%)... Mantenha a sala livre.`;
        calGiantSubtitle.textContent = `ESTÁGIO 1: ${modeDesc} — Anulando móveis, medindo piso Z-Score e gerando blacklist de subportadoras.`;
      }
      calGiantMetric.textContent = `${progressPct}%`;
      calGiantBar.className = 'bg-cyan-400 h-full rounded-full transition-all duration-150';
      calGiantBar.style.width = `${progressPct}%`;
    } else if (calState === 'completed') {
      calGiantBox.className = 'rounded-2xl border-2 border-emerald-400 bg-emerald-950/90 text-emerald-100 p-5 shadow-2xl flex flex-col md:flex-row items-center justify-between gap-4';
      calGiantIcon.className = 'w-14 h-14 rounded-2xl bg-emerald-500/25 border border-emerald-400/50 flex items-center justify-center text-2xl shrink-0';
      calGiantIcon.textContent = '✅';
      calGiantTitle.textContent = '✅ CALIBRAÇÃO CONCLUÍDA COM SUCESSO!';
      calGiantSubtitle.textContent = 'Perfil salvo no disco (calibration_profile.json) e preservado para reinicializações!';
      calGiantMetric.textContent = '100%';
      calGiantBar.className = 'bg-emerald-400 h-full rounded-full transition-all duration-150';
      calGiantBar.style.width = '100%';
    }
  }

  function renderTopologyDiagnostics(topo) {
    const code = topo.mode_code;
    const nActive = topo.active_nodes || 0;

    if (code === 'NO_SENSORS') {
      noSensorsOverlay.classList.remove('hidden');
    } else {
      noSensorsOverlay.classList.add('hidden');
    }

    let badgeLabel = topo.mode_title || '1 ESP: Modo Corredor Fresnel';
    let badgeClass = 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
    let dotClass = 'bg-emerald-400';

    if (code === 'NO_SENSORS') {
      badgeClass = 'bg-rose-500/20 text-rose-300 border-rose-500/40';
      dotClass = 'bg-rose-500 animate-ping';
    } else if (code === 'SINGLE_LINK_CORRIDOR') {
      badgeClass = 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40';
      dotClass = 'bg-cyan-400';
    } else if (code === 'DUAL_LINK_SECTOR') {
      badgeClass = 'bg-amber-500/20 text-amber-300 border-amber-500/40';
      dotClass = 'bg-amber-400';
    }

    topoStatusDot.className = `w-2.5 h-2.5 rounded-full ${dotClass}`;
    topoModeBadge.className = `text-[10px] font-mono font-bold px-2 py-0.5 rounded border ${badgeClass}`;
    topoModeBadge.textContent = badgeLabel;

    topTopologyPill.className = `text-[10px] font-mono uppercase tracking-widest px-2 py-0.5 rounded border ${badgeClass}`;
    topTopologyPill.textContent = badgeLabel;
    radarSubtitleBadge.textContent = `${badgeLabel} • ${nActive} ESP Online`;

    topoPrecisionText.textContent = topo.precision_level || '';
    topoRecommendationText.textContent = topo.setup_recommendation || '';

    topoCapabilitiesList.innerHTML = (topo.capabilities_enabled || [])
      .map((item) => `<li class="flex items-start gap-1.5"><span class="text-emerald-400 font-bold">•</span><span>${item}</span></li>`)
      .join('');

    topoLimitationsList.innerHTML = (topo.limitations || [])
      .map((item) => `<li class="flex items-start gap-1.5"><span class="text-amber-400 font-bold">•</span><span>${item}</span></li>`)
      .join('');
  }

  function renderTrackedPeopleCards(people) {
    if (!people.length) {
      const msg =
        liveState && liveState.topology_info && liveState.topology_info.mode_code === 'NO_SENSORS'
          ? 'Sistema em espera — conecte 1+ ESP32 ou ative o Simulador.'
          : 'Zona de Abrangência Livre — 0 pessoas detectadas.';
      trackedPeopleCards.innerHTML = `
        <div class="col-span-full bg-slate-950/70 border border-slate-800/80 rounded-xl px-4 py-2.5 text-xs text-slate-400 font-mono flex items-center justify-between">
          <span>${msg}</span>
          <span class="text-emerald-400">Anti-Drop Tracker Ativo</span>
        </div>
      `;
      return;
    }

    trackedPeopleCards.innerHTML = people
      .map((p) => {
        const isMoving = p.activity === 'moving';
        const badgeClass = isMoving
          ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
          : 'bg-amber-500/20 text-amber-300 border-amber-500/40';
        const stateLabel = isMoving ? 'EM MOVIMENTO' : 'PARADO (RETIDO)';
        const confPct = Math.round(p.confidence * 100);

        return `
          <div class="bg-slate-950/90 border border-slate-800 rounded-xl p-2.5 flex flex-col gap-1">
            <div class="flex items-center justify-between">
              <span class="font-mono font-bold text-xs text-white flex items-center gap-1.5">
                <span class="w-2 h-2 rounded-full ${isMoving ? 'bg-rose-500 animate-ping' : 'bg-amber-400'}"></span>
                Pessoa ${p.id}
              </span>
              <span class="text-[10px] font-mono px-1.5 py-0.5 rounded border ${badgeClass}">${stateLabel}</span>
            </div>
            <div class="flex items-center justify-between text-[11px] font-mono text-slate-400 mt-0.5">
              <span>Coord: <strong class="text-slate-200">(${p.x_m.toFixed(1)}m, ${p.y_m.toFixed(1)}m)</strong></span>
              <span class="text-cyan-300 truncate max-w-[155px]" title="${p.zone}">${p.zone}</span>
            </div>
            <div class="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden mt-1">
              <div class="bg-cyan-400 h-full" style="width: ${confPct}%"></div>
            </div>
          </div>
        `;
      })
      .join('');
  }

  function renderNodesTelemetry(nodesStatus) {
    const entries = Object.values(nodesStatus);
    if (!entries.length) return;

    nodesTelemetryList.innerHTML = entries
      .map((node) => {
        if (!node.online) {
          return `
            <div class="bg-slate-950/50 border border-dashed border-slate-800 rounded-xl p-2.5 flex items-center justify-between text-xs opacity-60">
              <div class="flex items-center gap-2">
                <span class="w-2 h-2 rounded-full bg-slate-600"></span>
                <span class="font-semibold text-slate-400">${node.name}</span>
              </div>
              <span class="font-mono text-[10px] uppercase px-2 py-0.5 rounded bg-slate-900 text-slate-500 border border-slate-800">
                OFFLINE (&gt;3.0s)
              </span>
            </div>
          `;
        }

        const curE = node.current_energy || 0;
        const floorE = Math.max(0.05, node.effective_noise_floor || 0.30);
        const netE = node.net_motion_energy || 0;
        const zScore = (node.z_score || 0).toFixed(1);
        const ttlSec = (node.ttl_remaining_sec || 0).toFixed(0);
        const s1 = (node.pca_s1 || 0).toFixed(2);
        const s2 = (node.pca_s2 || 0).toFixed(2);
        const distM = (node.estimated_distance_m || 3.0).toFixed(1);

        // Escala visual da barra: Limiar de Disparo (effective_noise_floor) fica fixo na marca de 45% da barra
        const scaleMax = floorE / 0.45;
        const fillPct = Math.min(100, Math.round((curE / scaleMax) * 100));
        const isTriggered = netE > 0 && (node.z_score || 0) > 4.5;
        const barColor = isTriggered
          ? 'bg-rose-500'
          : curE > floorE * 0.75
          ? 'bg-amber-400'
          : 'bg-emerald-400';

        return `
          <div class="bg-slate-950/90 border ${isTriggered ? 'border-rose-500/50' : 'border-slate-800/90'} rounded-xl p-2.5 flex flex-col gap-1.5">
            <div class="flex items-center justify-between text-xs">
              <div class="flex items-center gap-2">
                <span class="w-2 h-2 rounded-full ${isTriggered ? 'bg-rose-500 animate-ping' : 'bg-emerald-400'}"></span>
                <span class="font-bold text-slate-200">${node.name}</span>
                <span class="text-[10px] font-mono px-1.5 py-0.2 rounded bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
                  ~${distM}m
                </span>
              </div>
              <div class="font-mono text-[11px] text-slate-300">
                <span class="text-cyan-300">${node.rssi} dBm</span> • <span>${node.packets_per_sec.toFixed(0)} pkt/s</span>
              </div>
            </div>

            <!-- Barra de Ruído Atual vs. Limiar de Disparo (Z-Score Noise Gate) -->
            <div class="flex flex-col gap-1 bg-slate-900/90 rounded-lg p-1.5">
              <div class="flex items-center justify-between text-[10px] font-mono">
                <span class="text-slate-400">Sinal: <strong class="text-white">${curE.toFixed(2)}</strong> / Gate: <strong class="text-amber-300">${floorE.toFixed(2)}</strong></span>
                <span class="${isTriggered ? 'text-rose-300 font-bold' : 'text-emerald-300'}">
                  Z-Score: ${zScore} ${isTriggered ? '🔥 DISPARO' : '✓ ABAIXO DO GATE'}
                </span>
              </div>
              <div class="relative w-full bg-slate-950 rounded-full h-2.5 overflow-hidden border border-slate-800">
                <div class="${barColor} h-full transition-all duration-150" style="width: ${fillPct}%"></div>
                <!-- Linha vertical marcando o Limiar de Disparo (45%) -->
                <div class="absolute top-0 bottom-0 w-0.5 bg-amber-300 shadow" style="left: 45%" title="Limiar de Disparo (Noise Gate)"></div>
              </div>
            </div>

            <div class="flex items-center justify-between text-[10px] font-mono text-slate-400">
              <span>Líquido: <strong class="text-cyan-300">${netE.toFixed(2)}</strong></span>
              <span>SVD s₁/s₂: <strong class="text-slate-200">${s1}/${s2}</strong></span>
              <span>TTL Estático: <strong class="${ttlSec > 0 ? 'text-amber-300' : 'text-slate-500'}">${ttlSec}s</strong></span>
            </div>
          </div>
        `;
      })
      .join('');
  }

  function renderTelemetryInspector(state) {
    if (!inspectorConsensusPill) return;

    const gd = state.global_decision || {};
    const count = state.people_count || 0;
    const trigCount = gd.nodes_triggered_count || 0;
    const s1 = (gd.global_svd_s1 || 0).toFixed(2);
    const s2 = (gd.global_svd_s2 || 0).toFixed(2);
    const ratio = (gd.global_svd_ratio || 0).toFixed(2);
    const eRatio = (gd.global_energy_ratio || 0).toFixed(2);

    inspectorConsensusPill.textContent = `Consenso SVD: ${count} Pessoa${count === 1 ? '' : 's'}`;
    inspectorDecisionReason.textContent = gd.decision_reason || 'Zona de Abrangência Livre';
    inspectorSvdVals.textContent = `λ1=${s1} | λ2=${s2}`;

    if (parseFloat(ratio) < 0.38) {
      inspectorSvdRatio.className = 'text-emerald-300 block font-bold';
      inspectorSvdRatio.textContent = `${ratio} (< 0.38 Dominante)`;
    } else {
      inspectorSvdRatio.className = 'text-amber-300 block font-bold';
      inspectorSvdRatio.textContent = `${ratio} (>= 0.38 Multi-Alvo)`;
    }

    inspectorEnergyRatio.textContent = `${eRatio}x`;
    const totalNodes = Object.keys(state.nodes_status || {}).length;
    inspectorNodesTriggered.textContent = `${trigCount} / ${totalNodes}`;

    if (heroSvdRatioText) {
      heroSvdRatioText.textContent = `SVD λ1/λ2: ${s1}/${s2} • Ratio: ${ratio}`;
    }

    const nodes = Object.values(state.nodes_status || {});
    if (!inspectorNodesGrid || !nodes.length) return;

    inspectorNodesGrid.innerHTML = nodes.map((node) => {
      if (!node.online) {
        return `
          <div class="bg-slate-950/70 border border-slate-800 rounded-xl p-3 text-xs opacity-60">
            <div class="flex items-center justify-between">
              <strong class="text-slate-400">${node.name}</strong>
              <span class="text-[10px] font-mono text-slate-500">OFFLINE</span>
            </div>
          </div>
        `;
      }

      const isTrig = Boolean(node.is_triggered);
      const pr = (node.pearson_r !== undefined ? node.pearson_r : 1.0).toFixed(3);
      const mi = (node.motion_index !== undefined ? node.motion_index : 0.0).toFixed(3);
      const th = (node.motion_threshold !== undefined ? node.motion_threshold : 0.045).toFixed(3);
      const zs = (node.z_score || 0).toFixed(1);
      const zPass = (node.z_score || 0) >= 3.8;

      const fillPct = Math.min(100, Math.round((node.motion_index / (Math.max(0.01, node.motion_threshold) * 2.2)) * 100));
      const badgeStyle = isTrig
        ? 'bg-rose-500/20 text-rose-300 border-rose-500/40 animate-pulse'
        : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30';
      const badgeText = isTrig ? '🔥 DISPARADO (Z>=3.8)' : '✓ ABAIXO DO GATE';

      return `
        <div class="bg-slate-950/90 border ${isTrig ? 'border-rose-500/40' : 'border-slate-800'} rounded-xl p-3 flex flex-col gap-2 text-xs font-mono">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-1.5">
              <span class="w-2.5 h-2.5 rounded-full ${isTrig ? 'bg-rose-500' : 'bg-emerald-400'}"></span>
              <strong class="text-white text-xs">${node.name}</strong>
            </div>
            <span class="text-[10px] px-2 py-0.5 rounded border ${badgeStyle}">${badgeText}</span>
          </div>

          <div class="grid grid-cols-2 gap-2 bg-slate-900/80 p-2 rounded-lg text-[11px]">
            <div>
              <span class="text-slate-400 block text-[10px]">Pearson (r):</span>
              <strong class="${parseFloat(pr) < 0.90 ? 'text-amber-300' : 'text-slate-200'}">${pr}</strong>
            </div>
            <div>
              <span class="text-slate-400 block text-[10px]">Motion (1 - r²):</span>
              <strong class="${parseFloat(mi) >= parseFloat(th) ? 'text-rose-300' : 'text-emerald-300'}">${mi}</strong>
            </div>
            <div>
              <span class="text-slate-400 block text-[10px]">Limiar Corte:</span>
              <span class="text-amber-400">${th}</span>
            </div>
            <div>
              <span class="text-slate-400 block text-[10px]">Z-Score:</span>
              <strong class="${zPass ? 'text-rose-400' : 'text-slate-300'}">${zs} (meta &ge; 3.8)</strong>
            </div>
          </div>

          <div class="w-full bg-slate-900 rounded-full h-2 overflow-hidden border border-slate-800">
            <div class="${isTrig ? 'bg-rose-500' : 'bg-emerald-400'} h-full transition-all duration-150" style="width: ${fillPct}%"></div>
          </div>
        </div>
      `;
    }).join('');
  }

  function renderEventsFeed(events) {
    eventCounterBadge.textContent = `${events.length} eventos`;
    if (!events.length) {
      eventsFeedList.innerHTML = `<div class="text-xs text-slate-500 font-mono">Aguardando agregação temporal...</div>`;
      return;
    }

    eventsFeedList.innerHTML = events
      .map((ev) => {
        let badgeStyle = 'bg-slate-800 text-slate-300 border-slate-700';
        if (ev.activity_level === 'high') {
          badgeStyle = 'bg-rose-500/20 text-rose-300 border-rose-500/40';
        } else if (ev.activity_level === 'moderate') {
          badgeStyle = 'bg-amber-500/20 text-amber-300 border-amber-500/40';
        } else if (ev.activity_level === 'low') {
          badgeStyle = 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40';
        }

        const zonesHtml = (ev.active_zones || [])
          .map((z) => `<span class="px-1.5 py-0.2 rounded bg-cyan-500/15 text-cyan-300 font-mono text-[10px]">${z}</span>`)
          .join(' ');

        return `
          <div class="bg-slate-950/90 border border-slate-800/90 rounded-xl p-2.5 text-xs flex flex-col gap-1">
            <div class="flex items-center justify-between">
              <span class="font-mono text-[11px] text-slate-400">${ev.timestamp}</span>
              <div class="flex items-center gap-1.5">
                ${zonesHtml}
                <span class="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded border ${badgeStyle}">${ev.activity_level}</span>
              </div>
            </div>
            <p class="text-slate-200 font-medium leading-snug">${ev.summary}</p>
          </div>
        `;
      })
      .join('');
  }

  // ==========================================================================
  // 4. Renderização do Radar 2D e Espectro OFDM (com Top-18 Subportadoras)
  // ==========================================================================

  function computeRoomViewport(widthM, lengthM) {
    const padX = 54;
    const padY = 38;
    const maxW = radarCanvas.width - padX * 2;
    const maxH = radarCanvas.height - padY * 2;
    const aspect = widthM / Math.max(0.1, lengthM);

    let drawW = maxW;
    let drawH = drawW / aspect;
    if (drawH > maxH) {
      drawH = maxH;
      drawW = drawH * aspect;
    }

    const offsetX = Math.round((radarCanvas.width - drawW) / 2);
    const offsetY = Math.round((radarCanvas.height - drawH) / 2);
    roomViewport = { x: offsetX, y: offsetY, w: drawW, h: drawH };
    return roomViewport;
  }

  function normToCanvas(nx, ny) {
    return {
      cx: roomViewport.x + nx * roomViewport.w,
      cy: roomViewport.y + ny * roomViewport.h,
    };
  }

  function canvasToNorm(cx, cy) {
    return {
      nx: Math.max(0.04, Math.min(0.96, (cx - roomViewport.x) / roomViewport.w)),
      ny: Math.max(0.05, Math.min(0.95, (cy - roomViewport.y) / roomViewport.h)),
    };
  }

  function getCanvasMousePos(evt) {
    const rect = radarCanvas.getBoundingClientRect();
    const scaleX = radarCanvas.width / rect.width;
    const scaleY = radarCanvas.height / rect.height;
    return {
      cx: (evt.clientX - rect.left) * scaleX,
      cy: (evt.clientY - rect.top) * scaleY,
    };
  }

  let draggingNodeId = null;
  let isDraggingNode = false;

  function findNodeUnderPointer(cx, cy) {
    if (!liveState || !liveState.room_config) return null;
    const cfg = liveState.room_config;
    const txPos = normToCanvas(cfg.tx.x, cfg.tx.y);
    if (Math.hypot(cx - txPos.cx, cy - txPos.cy) <= 24) {
      return { id: 'tx_router', isTx: true };
    }
    for (const rx of cfg.rx_nodes) {
      const rxPos = normToCanvas(rx.x, rx.y);
      if (Math.hypot(cx - rxPos.cx, cy - rxPos.cy) <= 22) {
        return { id: rx.id, isTx: false, rx };
      }
    }
    return null;
  }

  async function saveNodePositionsToServer() {
    if (!liveState || !liveState.room_config) return;
    try {
      await fetch('/api/config/room', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(liveState.room_config),
      });
    } catch (e) {
      console.debug('Falha ao salvar posições dos nós:', e);
    }
  }

  radarCanvas.addEventListener('mousedown', (evt) => {
    const pos = getCanvasMousePos(evt);
    const hit = findNodeUnderPointer(pos.cx, pos.cy);
    if (hit) {
      isDraggingNode = true;
      draggingNodeId = hit.id;
      radarCanvas.style.cursor = 'grabbing';
    }
  });

  radarCanvas.addEventListener('mousemove', (evt) => {
    const pos = getCanvasMousePos(evt);
    if (isDraggingNode && draggingNodeId && liveState && liveState.room_config) {
      const norm = canvasToNorm(pos.cx, pos.cy);
      if (draggingNodeId === 'tx_router') {
        liveState.room_config.tx.x = Math.round(norm.nx * 100) / 100;
        liveState.room_config.tx.y = Math.round(norm.ny * 100) / 100;
      } else {
        const rx = liveState.room_config.rx_nodes.find((n) => n.id === draggingNodeId);
        if (rx) {
          rx.x = Math.round(norm.nx * 100) / 100;
          rx.y = Math.round(norm.ny * 100) / 100;
        }
      }
      return;
    }
    const hover = findNodeUnderPointer(pos.cx, pos.cy);
    radarCanvas.style.cursor = hover ? 'grab' : 'default';
  });

  const onPointerUp = async () => {
    if (isDraggingNode) {
      isDraggingNode = false;
      draggingNodeId = null;
      radarCanvas.style.cursor = 'default';
      await saveNodePositionsToServer();
    }
  };

  radarCanvas.addEventListener('mouseup', onPointerUp);
  radarCanvas.addEventListener('mouseleave', onPointerUp);

  radarCanvas.addEventListener('touchstart', (evt) => {
    if (evt.touches.length === 1) {
      const pos = getCanvasMousePos(evt.touches[0]);
      const hit = findNodeUnderPointer(pos.cx, pos.cy);
      if (hit) {
        isDraggingNode = true;
        draggingNodeId = hit.id;
        evt.preventDefault();
      }
    }
  }, { passive: false });

  radarCanvas.addEventListener('touchmove', (evt) => {
    if (isDraggingNode && evt.touches.length === 1) {
      const pos = getCanvasMousePos(evt.touches[0]);
      const norm = canvasToNorm(pos.cx, pos.cy);
      if (draggingNodeId === 'tx_router') {
        liveState.room_config.tx.x = Math.round(norm.nx * 100) / 100;
        liveState.room_config.tx.y = Math.round(norm.ny * 100) / 100;
      } else {
        const rx = liveState.room_config.rx_nodes.find((n) => n.id === draggingNodeId);
        if (rx) {
          rx.x = Math.round(norm.nx * 100) / 100;
          rx.y = Math.round(norm.ny * 100) / 100;
        }
      }
      evt.preventDefault();
    }
  }, { passive: false });

  radarCanvas.addEventListener('touchend', onPointerUp);
  radarCanvas.addEventListener('touchcancel', onPointerUp);

  function heatmapColorRGBA(val) {
    const v = Math.max(0, Math.min(1, val));
    if (v < 0.05) return [0, 0, 0, 0];
    if (v < 0.30) {
      const t = (v - 0.05) / 0.25;
      return [6, Math.round(110 + 72 * t), Math.round(180 + 32 * t), Math.round(135 * t)];
    }
    if (v < 0.60) {
      const t = (v - 0.30) / 0.30;
      return [
        Math.round(6 + 239 * t),
        Math.round(182 - 24 * t),
        Math.round(212 - 201 * t),
        Math.round(135 + 65 * t),
      ];
    }
    const t = (v - 0.60) / 0.40;
    return [
      Math.round(245 - 1 * t),
      Math.round(158 - 95 * t),
      Math.round(11 + 83 * t),
      Math.round(200 + 45 * t),
    ];
  }

  function drawRadarFrame() {
    animPhase += 0.06;
    radarCtx.clearRect(0, 0, radarCanvas.width, radarCanvas.height);

    radarCtx.fillStyle = '#04070f';
    radarCtx.fillRect(0, 0, radarCanvas.width, radarCanvas.height);

    if (!liveState || !liveState.room_config) {
      requestAnimationFrame(drawRadarFrame);
      return;
    }

    const cfg = liveState.room_config;
    const topo = liveState.topology_info || { mode_code: 'SINGLE_LINK_CORRIDOR', active_nodes: 1 };
    const cov = liveState.coverage_info || { active_area_m2: 13.3, avg_tx_rx_distance_m: 3.0, lateral_radius_m: 1.8 };
    const modeCode = topo.mode_code;
    const vp = computeRoomViewport(cfg.width_m, cfg.length_m);

    const txPos = normToCanvas(cfg.tx.x, cfg.tx.y);
    const nodesStatus = liveState.nodes_status || {};
    const onlineRxList = cfg.rx_nodes.filter((rx) => nodesStatus[rx.id] && nodesStatus[rx.id].online);

    // 1. Grade Métrica
    radarCtx.save();
    radarCtx.strokeStyle = 'rgba(30, 41, 59, 0.42)';
    radarCtx.lineWidth = 1;

    for (let xm = 0; xm <= cfg.width_m; xm += 1.0) {
      const gx = vp.x + (xm / cfg.width_m) * vp.w;
      radarCtx.beginPath();
      radarCtx.moveTo(gx, vp.y);
      radarCtx.lineTo(gx, vp.y + vp.h);
      radarCtx.stroke();

      radarCtx.fillStyle = '#475569';
      radarCtx.font = '10px monospace';
      radarCtx.textAlign = 'center';
      radarCtx.fillText(`${xm.toFixed(0)}m`, gx, vp.y - 8);
    }

    for (let ym = 0; ym <= cfg.length_m; ym += 1.0) {
      const gy = vp.y + (ym / cfg.length_m) * vp.h;
      radarCtx.beginPath();
      radarCtx.moveTo(vp.x, gy);
      radarCtx.lineTo(vp.x + vp.w, gy);
      radarCtx.stroke();

      radarCtx.fillStyle = '#475569';
      radarCtx.font = '10px monospace';
      radarCtx.textAlign = 'right';
      radarCtx.fillText(`${ym.toFixed(0)}m`, vp.x - 8, gy + 3);
    }
    radarCtx.restore();

    // 2. Heatmap 20x20
    if (liveState.heatmap_grid && liveState.heatmap_grid.length === 20 && modeCode !== 'NO_SENSORS') {
      const imgData = heatmapOffCtx.createImageData(20, 20);
      const dataArr = imgData.data;
      for (let r = 0; r < 20; r++) {
        for (let c = 0; c < 20; c++) {
          const val = liveState.heatmap_grid[r][c] || 0;
          const [R, G, B, A] = heatmapColorRGBA(val);
          const idx = (r * 20 + c) * 4;
          dataArr[idx] = R;
          dataArr[idx + 1] = G;
          dataArr[idx + 2] = B;
          dataArr[idx + 3] = A;
        }
      }
      heatmapOffCtx.putImageData(imgData, 0, 0);

      radarCtx.save();
      radarCtx.imageSmoothingEnabled = true;
      radarCtx.imageSmoothingQuality = 'high';
      radarCtx.drawImage(heatmapOffscreen, vp.x, vp.y, vp.w, vp.h);
      radarCtx.restore();
    }

    // 3. Zona de Abrangência Ativa (Elipse de Fresnel)
    onlineRxList.forEach((rx) => {
      const rxPos = normToCanvas(rx.x, rx.y);
      const st = nodesStatus[rx.id] || { perturbation_score: 0, estimated_distance_m: 3.0, online: true };
      const pScore = st.perturbation_score || 0;
      const distM = st.estimated_distance_m || cov.avg_tx_rx_distance_m || 3.0;

      const dx = rxPos.cx - txPos.cx;
      const dy = rxPos.cy - txPos.cy;
      const distPx = Math.hypot(dx, dy);
      const angle = Math.atan2(dy, dx);
      const centerLineX = (txPos.cx + rxPos.cx) / 2;
      const centerLineY = (txPos.cy + rxPos.cy) / 2;

      const pxPerMeter = vp.w / Math.max(1.0, cfg.width_m);
      const lateralRadiusPx = Math.max(54, Math.min(vp.w * 0.42, (cov.lateral_radius_m || 1.8) * pxPerMeter * 0.78));
      const majorRadiusPx = distPx / 2 + 28;

      radarCtx.save();
      radarCtx.translate(centerLineX, centerLineY);
      radarCtx.rotate(angle);

      radarCtx.beginPath();
      radarCtx.ellipse(0, 0, majorRadiusPx, lateralRadiusPx, 0, 0, Math.PI * 2);
      radarCtx.fillStyle = pScore > 0.35 ? 'rgba(244, 63, 94, 0.08)' : 'rgba(6, 182, 212, 0.07)';
      radarCtx.strokeStyle = pScore > 0.35 ? 'rgba(244, 63, 94, 0.65)' : 'rgba(34, 211, 238, 0.55)';
      radarCtx.lineWidth = 2.0;
      radarCtx.setLineDash([8, 5]);
      radarCtx.fill();
      radarCtx.stroke();
      radarCtx.setLineDash([]);

      radarCtx.beginPath();
      radarCtx.ellipse(0, 0, distPx / 2, lateralRadiusPx * 0.52, 0, 0, Math.PI * 2);
      radarCtx.fillStyle = pScore > 0.35 ? 'rgba(244, 63, 94, 0.07)' : 'rgba(16, 185, 129, 0.06)';
      radarCtx.strokeStyle = 'rgba(110, 231, 183, 0.35)';
      radarCtx.lineWidth = 1.2;
      radarCtx.fill();
      radarCtx.stroke();

      radarCtx.restore();

      if (onlineRxList.length === 1) {
        radarCtx.save();
        radarCtx.fillStyle = '#67e8f9';
        radarCtx.font = 'bold 11px monospace';
        radarCtx.textAlign = 'center';
        radarCtx.fillText(
          `ZONA DE ABRANGÊNCIA ATIVA (~${cov.active_area_m2.toFixed(1)} m² • Distância TX-RX: ${distM.toFixed(1)}m)`,
          centerLineX,
          vp.y + 18
        );
        radarCtx.restore();
      }

      radarCtx.save();
      radarCtx.beginPath();
      radarCtx.moveTo(txPos.cx, txPos.cy);
      radarCtx.lineTo(rxPos.cx, rxPos.cy);
      radarCtx.strokeStyle =
        pScore > 0.42
          ? 'rgba(251, 113, 133, 0.85)'
          : pScore > 0.18
          ? 'rgba(251, 191, 36, 0.75)'
          : 'rgba(34, 211, 238, 0.65)';
      radarCtx.lineWidth = 2.5 + pScore * 3.0;
      radarCtx.stroke();

      const midBeamX = txPos.cx + dx * 0.5;
      const midBeamY = txPos.cy + dy * 0.5;
      radarCtx.fillStyle = 'rgba(15, 23, 42, 0.88)';
      radarCtx.strokeStyle = 'rgba(56, 189, 248, 0.45)';
      radarCtx.lineWidth = 1;
      radarCtx.beginPath();
      radarCtx.roundRect(midBeamX + 14, midBeamY - 10, 86, 18, 4);
      radarCtx.fill();
      radarCtx.stroke();

      radarCtx.fillStyle = '#bae6fd';
      radarCtx.font = 'bold 10px monospace';
      radarCtx.textAlign = 'center';
      radarCtx.fillText(`d ≈ ${distM.toFixed(1)}m`, midBeamX + 57, midBeamY + 3);

      const pulseT = (animPhase * (0.45 + pScore * 0.8)) % 1.0;
      const px = txPos.cx + dx * pulseT;
      const py = txPos.cy + dy * pulseT;
      radarCtx.fillStyle = pScore > 0.4 ? '#fb7185' : '#22d3ee';
      radarCtx.beginPath();
      radarCtx.arc(px, py, 3.2 + pScore * 2, 0, Math.PI * 2);
      radarCtx.fill();
      radarCtx.restore();
    });

    drawTxNode(txPos.cx, txPos.cy, cfg.tx);

    cfg.rx_nodes.forEach((rx) => {
      const rxPos = normToCanvas(rx.x, rx.y);
      const st = nodesStatus[rx.id];
      if (st && st.online) {
        drawRxNode(rxPos.cx, rxPos.cy, rx, st);
      }
    });

    (liveState.people_coordinates || []).forEach((person) => {
      drawTrackedPerson(person);
    });

    drawCSISpectrum();
    requestAnimationFrame(drawRadarFrame);
  }

  function drawTxNode(cx, cy, tx) {
    radarCtx.save();
    const isDragged = isDraggingNode && draggingNodeId === 'tx_router';
    const waveR = 15 + (Math.sin(animPhase * 1.5) + 1) * 5;
    radarCtx.strokeStyle = isDragged ? 'rgba(34, 211, 238, 0.85)' : 'rgba(6, 182, 212, 0.35)';
    radarCtx.lineWidth = isDragged ? 2.5 : 1.5;
    radarCtx.beginPath();
    radarCtx.arc(cx, cy, waveR + (isDragged ? 4 : 0), 0, Math.PI * 2);
    radarCtx.stroke();

    radarCtx.fillStyle = isDragged ? '#06b6d4' : '#0891b2';
    radarCtx.strokeStyle = '#67e8f9';
    radarCtx.lineWidth = isDragged ? 3.0 : 2.2;
    radarCtx.beginPath();
    radarCtx.arc(cx, cy, 11, 0, Math.PI * 2);
    radarCtx.fill();
    radarCtx.stroke();

    radarCtx.fillStyle = '#ffffff';
    radarCtx.font = 'bold 9px monospace';
    radarCtx.textAlign = 'center';
    radarCtx.fillText('TX', cx, cy + 3);

    radarCtx.fillStyle = isDragged ? '#a5f3fc' : '#67e8f9';
    radarCtx.font = 'bold 10px monospace';
    radarCtx.fillText(isDragged ? `TX [ARRASTANDO]` : `${tx.name} [Origem]`, cx, cy - 16);
    radarCtx.restore();
  }

  function drawRxNode(cx, cy, rx, status) {
    const isDragged = isDraggingNode && draggingNodeId === rx.id;
    const pScore = status.perturbation_score || 0;
    const distM = (status.estimated_distance_m || 3.0).toFixed(1);
    radarCtx.save();

    if (pScore > 0.1 || isDragged) {
      radarCtx.fillStyle = isDragged
        ? 'rgba(52, 211, 153, 0.35)'
        : pScore > 0.45
        ? 'rgba(244, 63, 94, 0.25)'
        : 'rgba(16, 185, 129, 0.22)';
      radarCtx.beginPath();
      radarCtx.arc(cx, cy, 13 + (isDragged ? 8 : pScore * 10), 0, Math.PI * 2);
      radarCtx.fill();
    }

    radarCtx.fillStyle = isDragged ? '#059669' : '#059669';
    radarCtx.strokeStyle = isDragged ? '#34d399' : '#6ee7b7';
    radarCtx.lineWidth = isDragged ? 3.0 : 2;
    radarCtx.beginPath();
    radarCtx.roundRect(cx - 11, cy - 11, 22, 22, 6);
    radarCtx.fill();
    radarCtx.stroke();

    radarCtx.fillStyle = '#ffffff';
    radarCtx.font = 'bold 8px monospace';
    radarCtx.textAlign = 'center';
    radarCtx.fillText(rx.id.replace('esp_', 'ESP'), cx, cy + 3);

    radarCtx.fillStyle = isDragged ? '#6ee7b7' : '#a7f3d0';
    radarCtx.font = 'bold 10px monospace';
    const label = isDragged
      ? `${rx.id.toUpperCase()} [POSICIONANDO: ${(rx.x * 100).toFixed(0)}%, ${(rx.y * 100).toFixed(0)}%]`
      : `${rx.id.toUpperCase()} (~${distM}m • ${status.rssi} dBm)`;
    radarCtx.fillText(label, cx, cy + 24);
    radarCtx.restore();
  }

  function drawTrackedPerson(person) {
    const pos = normToCanvas(person.x, person.y);
    const isMoving = person.activity === 'moving';
    const mainColor = isMoving ? '#f43f5e' : '#f59e0b';
    const glowColor = isMoving ? 'rgba(244, 63, 94, 0.28)' : 'rgba(245, 158, 11, 0.24)';

    radarCtx.save();

    const trail = personTrails[person.id] || [];
    if (trail.length > 1) {
      radarCtx.beginPath();
      trail.forEach((pt, idx) => {
        const cPt = normToCanvas(pt.x, pt.y);
        if (idx === 0) radarCtx.moveTo(cPt.cx, cPt.cy);
        else radarCtx.lineTo(cPt.cx, cPt.cy);
      });
      radarCtx.strokeStyle = isMoving ? 'rgba(244, 63, 94, 0.45)' : 'rgba(245, 158, 11, 0.35)';
      radarCtx.lineWidth = 2;
      radarCtx.stroke();
    }

    const confRadius = 20 + (1.0 - person.confidence) * 24;
    radarCtx.fillStyle = glowColor;
    radarCtx.strokeStyle = mainColor;
    radarCtx.lineWidth = 1.4;
    radarCtx.beginPath();
    radarCtx.arc(pos.cx, pos.cy, confRadius, 0, Math.PI * 2);
    radarCtx.fill();
    radarCtx.stroke();

    radarCtx.fillStyle = mainColor;
    radarCtx.strokeStyle = '#ffffff';
    radarCtx.lineWidth = 2.2;
    radarCtx.beginPath();
    radarCtx.arc(pos.cx, pos.cy, 9, 0, Math.PI * 2);
    radarCtx.fill();
    radarCtx.stroke();

    const tagText = `${person.id} • ${isMoving ? 'MOV' : 'PARADO'} (${person.x_m.toFixed(1)}m, ${person.y_m.toFixed(1)}m)`;
    radarCtx.font = 'bold 10px monospace';
    const textWidth = radarCtx.measureText(tagText).width;
    const boxW = textWidth + 14;
    const boxH = 19;
    const boxX = pos.cx - boxW / 2;
    const boxY = pos.cy - confRadius - 22;

    radarCtx.fillStyle = 'rgba(15, 23, 42, 0.94)';
    radarCtx.strokeStyle = mainColor;
    radarCtx.lineWidth = 1.2;
    radarCtx.beginPath();
    radarCtx.roundRect(boxX, boxY, boxW, boxH, 5);
    radarCtx.fill();
    radarCtx.stroke();

    radarCtx.fillStyle = '#f8fafc';
    radarCtx.textAlign = 'center';
    radarCtx.fillText(tagText, pos.cx, boxY + 13);

    radarCtx.restore();
  }

  function drawCSISpectrum() {
    csiCtx.clearRect(0, 0, csiCanvas.width, csiCanvas.height);
    csiCtx.fillStyle = '#050811';
    csiCtx.fillRect(0, 0, csiCanvas.width, csiCanvas.height);

    if (!liveState || !liveState.nodes_status) return;
    const selectedId = selectCsiNode.value || 'esp_1';
    const node = liveState.nodes_status[selectedId] || Object.values(liveState.nodes_status)[0];
    if (!node) return;

    const amps = node.amplitudes_preview || [];
    const phases = node.phases_preview || [];
    const topK = new Set(node.top_k_indices || []);
    const n = amps.length;
    if (!n) return;

    const w = csiCanvas.width;
    const h = csiCanvas.height;
    const padX = 24;
    const padY = 16;
    const plotW = w - padX * 2;
    const plotH = h - padY * 2;

    csiCtx.strokeStyle = 'rgba(51, 65, 85, 0.5)';
    csiCtx.lineWidth = 1;
    csiCtx.beginPath();
    csiCtx.moveTo(padX, padY + plotH / 2);
    csiCtx.lineTo(padX + plotW, padY + plotH / 2);
    csiCtx.stroke();

    if (!node.online) {
      csiCtx.fillStyle = '#64748b';
      csiCtx.font = '11px monospace';
      csiCtx.textAlign = 'center';
      csiCtx.fillText(`${node.node_id} OFFLINE (Sem pacotes CSI nos últimos 3.0s)`, w / 2, h / 2 + 4);
      return;
    }

    // Destaca barras verticais sutis nas Top-18 Subportadoras selecionadas pelo motor SNR
    topK.forEach((idx) => {
      if (idx >= 0 && idx < n) {
        const x = padX + (idx / (n - 1)) * plotW;
        csiCtx.fillStyle = 'rgba(245, 158, 11, 0.14)';
        csiCtx.fillRect(x - 3, padY, 6, plotH);
      }
    });

    csiCtx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = padX + (i / (n - 1)) * plotW;
      const normAmp = Math.min(1.0, Math.max(0.0, amps[i] / 48.0));
      const y = padY + plotH - normAmp * plotH;
      if (i === 0) csiCtx.moveTo(x, y);
      else csiCtx.lineTo(x, y);
    }
    csiCtx.strokeStyle = '#22d3ee';
    csiCtx.lineWidth = 2;
    csiCtx.stroke();

    csiCtx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = padX + (i / (n - 1)) * plotW;
      const normPhase = Math.max(-1.0, Math.min(1.0, phases[i] / 2.2));
      const y = padY + plotH / 2 - normPhase * (plotH * 0.42);
      if (i === 0) csiCtx.moveTo(x, y);
      else csiCtx.lineTo(x, y);
    }
    csiCtx.strokeStyle = '#34d399';
    csiCtx.lineWidth = 1.6;
    csiCtx.stroke();
  }

  // ==========================================================================
  // 5. Eventos de Controle (Modo, Sensibilidade, Calibração 2 Etapas e Ollama)
  // ==========================================================================

  async function setSystemMode({ mode = null, simulatedPeople = null, simulatedNodes = null, sensitivity = null }) {
    ensureAudioContext();
    const payload = {};
    if (mode !== null) payload.mode = mode;
    if (simulatedPeople !== null) payload.simulated_people = simulatedPeople;
    if (simulatedNodes !== null) payload.simulated_nodes = simulatedNodes;
    if (sensitivity !== null) payload.sensitivity = sensitivity;

    await fetch('/api/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  }

  btnModeSim.addEventListener('click', () => setSystemMode({ mode: 'simulation' }));
  btnModeHw.addEventListener('click', () => setSystemMode({ mode: 'hardware' }));

  simNodesBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const nNodes = parseInt(btn.getAttribute('data-sim-nodes'), 10);
      setSystemMode({ mode: 'simulation', simulatedNodes: nNodes });
    });
  });

  quickSimBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const nNodes = parseInt(btn.getAttribute('data-quick-sim'), 10);
      const people = liveState && liveState.simulated_people > 0 ? liveState.simulated_people : 2;
      setSystemMode({ mode: 'simulation', simulatedPeople: people, simulatedNodes: nNodes });
    });
  });

  simCountBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const count = parseInt(btn.getAttribute('data-sim-count'), 10);
      setSystemMode({ mode: 'simulation', simulatedPeople: count });
    });
  });

  sensBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      const sVal = btn.getAttribute('data-sensitivity');
      setSystemMode({ sensitivity: sVal });
    });
  });

  // Botão Imediato: 🧹 Forçar Zero (Aqui está Vazio Agora)
  if (btnInstantZero) {
    btnInstantZero.addEventListener('click', async () => {
      ensureAudioContext();
      Object.keys(personTrails).forEach((k) => delete personTrails[k]);
      playCalibrationCompleteDoubleChime();
      await fetch('/api/calibrate/instant_zero', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
    });
  }

  // Estágio 1: Silêncio da Sala Vazia (15s com Delay 10s para Sair)
  if (btnWizardStage1) {
    btnWizardStage1.addEventListener('click', async () => {
      ensureAudioContext();
      playCountdownBeep(10);
      await fetch('/api/calibrate/wizard', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: 'stage1_silence',
          delay_seconds: 10,
          duration_seconds: 15,
        }),
      });
    });
  }

  // Estágio 2: 1 Pessoa Andando (20s)
  if (btnWizardStage2) {
    btnWizardStage2.addEventListener('click', async () => {
      ensureAudioContext();
      playCalibratingStartBeep();
      await fetch('/api/calibrate/wizard', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: 'stage2_1person',
          delay_seconds: 0,
          duration_seconds: 20,
        }),
      });
    });
  }

  // Estágio 3: 2 Pessoas Andando (15s)
  if (btnWizardStage3) {
    btnWizardStage3.addEventListener('click', async () => {
      ensureAudioContext();
      playCalibratingStartBeep();
      await fetch('/api/calibrate/wizard', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mode: 'stage3_2persons',
          delay_seconds: 0,
          duration_seconds: 15,
        }),
      });
    });
  }

  // Restaurar Posições Padrão dos Sensores (ESP1 Ponta Esq, ESP2 Ponta Dir, ESP3 Centro)
  if (btnResetNodes) {
    btnResetNodes.addEventListener('click', async () => {
      ensureAudioContext();
      playCalibratingStartBeep();
      try {
        const res = await fetch('/api/config/room/reset', { method: 'POST' });
        if (res.ok) {
          const cfg = await res.json();
          if (liveState) liveState.room_config = cfg;
        }
      } catch (e) {
        console.debug('Erro ao resetar nós:', e);
      }
    });
  }

  // ==========================================================================
  // 6. Integração com IA Local via Ollama (100% Offline)
  // ==========================================================================

  function renderMarkdownToHtml(md) {
    if (!md) return '';
    const lines = md.split('\n');
    let html = '';
    let inList = false;
    let inTable = false;

    const formatInline = (str) =>
      str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code>$1</code>');

    for (let i = 0; i < lines.length; i++) {
      const raw = lines[i].trim();

      if (!raw) {
        if (inList) {
          html += '</ul>';
          inList = false;
        }
        if (inTable) {
          html += '</tbody></table>';
          inTable = false;
        }
        continue;
      }

      if (raw.startsWith('|') && raw.endsWith('|')) {
        if (raw.includes('---')) continue;
        const cells = raw
          .slice(1, -1)
          .split('|')
          .map((c) => formatInline(c.trim()));
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

      if (raw.startsWith('# ')) {
        html += `<h1>${formatInline(raw.slice(2))}</h1>`;
      } else if (raw.startsWith('## ')) {
        html += `<h2>${formatInline(raw.slice(3))}</h2>`;
      } else if (raw.startsWith('### ')) {
        html += `<h3>${formatInline(raw.slice(4))}</h3>`;
      } else if (raw.startsWith('> ')) {
        html += `<blockquote>${formatInline(raw.slice(2))}</blockquote>`;
      } else if (raw.startsWith('- ') || /^\d+\.\s/.test(raw)) {
        if (!inList) {
          html += '<ul>';
          inList = true;
        }
        const itemText = raw.replace(/^(-|\d+\.)\s+/, '');
        html += `<li>${formatInline(itemText)}</li>`;
      } else if (raw === '---') {
        html += '<hr class="border-slate-800 my-2" />';
      } else {
        html += `<p>${formatInline(raw)}</p>`;
      }
    }

    if (inList) html += '</ul>';
    if (inTable) html += '</tbody></table>';
    return html;
  }

  async function checkOllamaStatus(preferredModel = null) {
    try {
      const query = preferredModel ? `?model=${encodeURIComponent(preferredModel)}` : '';
      const res = await fetch(`/api/llm/status${query}`);
      const status = await res.json();

      if (status.online) {
        llmStatusDot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse';
        llmProviderBadge.className = 'text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/40';
        llmProviderBadge.textContent = status.badge_text || `LLM Local Online (${status.active_model})`;
      } else {
        llmStatusDot.className = 'w-2.5 h-2.5 rounded-full bg-amber-400';
        llmProviderBadge.className = 'text-[10px] font-mono px-2 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-500/40';
        llmProviderBadge.textContent = status.badge_text || 'Ollama Offline (Fallback Estatístico)';
      }

      if (selectLlmModel && Array.isArray(status.available_models) && status.available_models.length) {
        const installedSet = new Set(status.installed_models || []);
        selectLlmModel.innerHTML = status.available_models
          .map((m) => {
            const isInstalled = installedSet.has(m);
            const suffix = isInstalled ? ' ✓ (Instalado)' : '';
            const selected = m === status.active_model ? 'selected' : '';
            return `<option value="${m}" ${selected}>${m}${suffix}</option>`;
          })
          .join('');
      }
    } catch (err) {
      console.debug('Falha ao consultar status do Ollama:', err);
    }
  }

  if (selectLlmModel) {
    selectLlmModel.addEventListener('change', () => {
      checkOllamaStatus(selectLlmModel.value);
    });
  }

  if (btnRefreshLlm) {
    btnRefreshLlm.addEventListener('click', () => {
      checkOllamaStatus(selectLlmModel ? selectLlmModel.value : null);
    });
  }

  btnGenerateReport.addEventListener('click', async () => {
    btnGenerateReport.disabled = true;
    const chosenModel = selectLlmModel ? selectLlmModel.value : 'qwen2.5:7b';
    btnReportText.textContent = `Analisando via ${chosenModel}...`;
    llmOutputContainer.innerHTML = `<div class="text-cyan-300 font-mono animate-pulse">Sintetizando Relatório Executivo via LLM Local (${chosenModel})...</div>`;

    try {
      const res = await fetch('/api/llm/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: chosenModel }),
      });
      const data = await res.json();
      llmProviderBadge.textContent =
        data.provider === 'ollama-local'
          ? `LLM Local Online (${chosenModel})`
          : data.model || 'Fallback Estatístico Local';
      llmOutputContainer.innerHTML = renderMarkdownToHtml(data.content);
    } catch (err) {
      llmOutputContainer.innerHTML = `<div class="text-rose-400">Erro ao gerar relatório: ${err.message}</div>`;
    } finally {
      btnGenerateReport.disabled = false;
      btnReportText.textContent = 'Gerar Relatório Executivo (LLM)';
    }
  });

  async function askLlmQuestion(questionText) {
    if (!questionText.trim()) return;
    const chosenModel = selectLlmModel ? selectLlmModel.value : 'qwen2.5:7b';
    llmOutputContainer.innerHTML = `<div class="text-cyan-300 font-mono animate-pulse">Consultando LLM Local (${chosenModel}): "${questionText}"...</div>`;

    try {
      const res = await fetch('/api/llm/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: questionText, model: chosenModel }),
      });
      const data = await res.json();
      llmProviderBadge.textContent =
        data.provider === 'ollama-local'
          ? `LLM Local Online (${chosenModel})`
          : data.model || 'Fallback Estatístico Local';
      llmOutputContainer.innerHTML = renderMarkdownToHtml(data.content);
    } catch (err) {
      llmOutputContainer.innerHTML = `<div class="text-rose-400">Erro na consulta à IA: ${err.message}</div>`;
    }
  }

  formAskLlm.addEventListener('submit', (evt) => {
    evt.preventDefault();
    const q = inputAskLlm.value;
    askLlmQuestion(q);
  });

  quickAskChips.forEach((chip) => {
    chip.addEventListener('click', () => {
      const q = chip.getAttribute('data-ask');
      inputAskLlm.value = q;
      askLlmQuestion(q);
    });
  });

  connectWebSocket();
  checkOllamaStatus();
  requestAnimationFrame(drawRadarFrame);
})();
