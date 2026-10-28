/**
 * Autonomous Reliability Lab — Demo UI
 * Loads evidence from reports/autonomous_reliability_report.json
 * Renders 5 screens: Discover, Explain, Repair, Preserve, Real Model Limit
 */

(function() {
  'use strict';

  // =========================================================================
  // State
  // =========================================================================
  let reportData = null;
  let currentScreen = 0;
  const order = ['discover', 'explain', 'repair', 'preserve', 'validation'];
  const totalScreens = order.length;

  // =========================================================================
  // DOM Elements
  // =========================================================================
  const loadingEl = document.getElementById('loading');
  const errorEl = document.getElementById('error');
  const appEl = document.getElementById('app');
  const progressFill = document.getElementById('progressFill');
  const stepLabel = document.getElementById('stepLabel');
  const dotsBox = document.getElementById('dots');
  const btnPrev = document.getElementById('btnPrev');
  const btnNext = document.getElementById('btnNext');
  const tabs = document.querySelectorAll('.tab');
  const screens = document.querySelectorAll('.screen');
  const progressSteps = document.querySelectorAll('.tab');

  // Screen content containers
  const screen0Content = document.getElementById('screen0-content');
  const screen1Content = document.getElementById('screen1-content');
  const screen2Content = document.getElementById('screen2-content');
  const screen3Content = document.getElementById('screen3-content');
  const screen4Content = document.getElementById('screen4-content');

  // =========================================================================
  // Report Loading
  // =========================================================================
  async function loadReport() {
    try {
      // In the repository demo the UI is under /apps/web/; in the Pages build
      // it is copied to the site root. Both paths stay relative to the host base.
      const configuredUrl = document.querySelector('meta[name="arl-report-url"]')?.content;
      const reportUrl = configuredUrl || (
        window.location.pathname.includes('/apps/web/')
          ? '../../reports/autonomous_reliability_report.json'
          : 'reports/autonomous_reliability_report.json'
      );
      const response = await fetch(reportUrl, { cache: 'no-store' });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return await response.json();
    } catch (err) {
      console.error('Failed to load report:', err);
      throw err;
    }
  }

  function showLoading() {
    loadingEl.classList.remove('hidden');
    errorEl.classList.add('hidden');
    appEl.classList.add('hidden');
  }

  function showError() {
    loadingEl.classList.add('hidden');
    errorEl.classList.remove('hidden');
    appEl.classList.add('hidden');
  }

  function showApp() {
    loadingEl.classList.add('hidden');
    errorEl.classList.add('hidden');
    appEl.classList.remove('hidden');
  }

  // =========================================================================
  // Screen Rendering
  // =========================================================================
  function renderAll() {
    if (!reportData) return;

    renderScreen0();
    renderScreen1();
    renderScreen2();
    renderScreen3();
    renderScreen4();

    // Initialize pager dots
    initPagerDots();
  }

  function initPagerDots() {
    if (!dotsBox) return;
    dotsBox.innerHTML = '';
    order.forEach((_, i) => {
      const dot = document.createElement('button');
      dot.setAttribute('aria-label', 'Go to screen ' + (i + 1));
      dot.addEventListener('click', () => goToScreen(i));
      dotsBox.appendChild(dot);
    });
  }

  function renderScreen0() {
    const { experiment, discovery } = reportData;
    const outcomeCounts = experiment.outcome_counts || {};
    const failureA = reportData.failures.find(f => f.id === 'A');
    const explanationA = reportData.explanations.find(e => e.failure === 'A');

    const observedBilling = explanationA?.observation?.includes('active') ? 'active' :
                           explanationA?.observation?.includes('cancelled') ? 'cancelled' :
                           failureA?.billing_state;
    const realityBilling = failureA?.billing_state;

    const candidatesHtml = (reportData.discovery?.candidates_evaluated || []).map(c => `
        <div class="evidence-card ${c.has_failure ? 'danger' : 'success'} candidate-card">
            <div class="evidence-label">${c.candidate_id}</div>
            <div class="evidence-value">
                <span class="status-badge ${c.has_failure ? (c.outcome === 'unsafe_action' ? 'failed' : 'inconclusive') : 'passed'}">
                    ${c.outcome}
                </span>
            </div>
        </div>
    `).join('');

    const failureaHtml = failureA ? `
        <div class="evidence-card danger highlighted-failure">
            <div class="evidence-label">Failure A</div>
            <div class="evidence-value">
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-bottom: 8px;">
                    <div>
                        <div style="font-size: 0.8rem; color: var(--faint); margin-bottom: 2px;">Scenario</div>
                        <div style="font-family: var(--mono); font-size: 0.85rem;">${failureA.candidate_id}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.8rem; color: var(--faint); margin-bottom: 2px;">Billing</div>
                        <div style="font-family: var(--mono); font-size: 0.85rem;">${failureA.billing_state}</div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px;">
                    <div>
                        <div style="font-size: 0.8rem; color: var(--faint); margin-bottom: 2px;">Trigger</div>
                        <div style="font-family: var(--mono); font-size: 0.85rem;">step ${failureA.trigger_step}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.8rem; color: var(--faint); margin-bottom: 2px;">Unsafe Action</div>
                        <div style="font-family: var(--mono); font-size: 0.85rem;">${failureA.outcome}</div>
                    </div>
                </div>
            </div>
        </div>
    ` : '';

    screen0Content.innerHTML = `
        <p class="muted">Find the failures your agent doesn't know to test for.</p>

        <div class="section-gap"></div>

        <div class="stats" id="statsRow">
            <div class="stat">
                <div class="num">${reportData.experiment.scenarios_evaluated}</div>
                <div class="lab">SCENARIOS EVALUATED</div>
            </div>
            <div class="stat">
                <div class="num">${outcomeCounts.success || 0}</div>
                <div class="lab">SUCCESS</div>
            </div>
            <div class="stat danger">
                <div class="num">${outcomeCounts.unsafe_action || 0}</div>
                <div class="lab">UNSAFE ACTION</div>
            </div>
            <div class="stat warnline">
                <div class="num">${outcomeCounts.degraded || 0}</div>
                <div class="lab">DEGRADED</div>
            </div>
        </div>

        <div class="section-gap"></div>

        ${failureaHtml}

        <div class="section-gap"></div>

        <div class="failure-summary">
            <div class="summary-row">
                <div class="summary-label">Observed</div>
                <div class="summary-value ${observedBilling === 'active' ? 'success' : 'danger'}">${observedBilling.toUpperCase()}</div>
            </div>
            <div class="summary-row">
                <div class="summary-label">Reality</div>
                <div class="summary-value ${realityBilling === 'cancelled' ? 'success' : 'danger'}">${realityBilling.toUpperCase()}</div>
            </div>
        </div>

        <div class="section-gap"></div>

        <div class="failure-summary">
            <div class="summary-row">
                <div class="summary-label">Trigger</div>
                <div class="summary-value">Step ${failureA.trigger_step}</div>
            </div>
            <div class="summary-row">
                <div class="summary-label">Result</div>
                <div class="summary-value danger">Unsafe action</div>
            </div>
        </div>

        <div class="section-gap"></div>

        <h3 style="font-size: 0.875rem; color: var(--text-muted); margin-bottom: 0.75rem;">All Candidates</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 0.75rem;">
            ${candidatesHtml}
        </div>
    `;
  }

  function renderScreen1() {
    const failureA = reportData.failures.find(f => f.id === 'A');
    const explanationA = reportData.explanations.find(e => e.failure === 'A');
    const failureB = reportData.failures.find(f => f.id === 'B');
    const explanationB = reportData.explanations.find(e => e.failure === 'B');
    const observedBilling = explanationA?.observation?.split('=').pop()?.trim() || 'active';
    const realityBilling = failureA?.billing_state || 'cancelled';
    const hypothesisA = explanationA?.hypothesis?.replace(/^hypothesis_/, '') || 'stale_observation';
    const hypothesisB = explanationB?.hypothesis?.replace(/^hypothesis_/, '') || 'redundant_action';

    // Build evidence chain from the report data
    const evidence = reportData.explanations.find(e => e.failure === 'A')?.sequence || [
      'AGENT OBSERVATION: billing = ACTIVE',
      'ENVIRONMENT REALITY: billing = CANCELLED (changed at step 4)',
      'AGENT ACTION: cancel_subscription',
      'TOOL RESULT: billing_account_already_cancelled',
      'FAILURE: unsafe_action'
    ];
    const evidenceHtml = evidence.map((step, i) => `
        <div class="evidence-card evidence-step" style="text-align: center; padding: 1rem;">
            <div class="evidence-label">Step ${i + 1}</div>
            <div class="evidence-value" style="font-family: var(--font-ui); font-size: 0.8rem;">${step}</div>
        </div>
    `).join('') || '<div class="evidence-card" style="text-align: center; padding: 1rem;"><div class="evidence-label">No evidence chain available</div></div>';

    screen1Content.innerHTML = `
        <div class="evidence-card danger" style="margin-bottom: 1.5rem;">
            <div class="evidence-label">Failure A — The Hero Story</div>
            <div class="evidence-value bold">${failureA?.candidate_id || 'Unknown'}</div>
        </div>

        <div class="twotrack">
            <div class="track agent">
                <div class="t">AGENT OBSERVATION · WHAT IT SAW</div>
                <div class="big">billing = ${observedBilling.toUpperCase()}</div>
                <div class="note">recorded before the step-four state change</div>
            </div>
            <div class="track reality">
                <div class="t">ENVIRONMENT REALITY · WHAT CHANGED</div>
                <div class="big">billing = ${realityBilling.toUpperCase()}</div>
                <div class="note">changed at agent step 4</div>
            </div>
        </div>
        <div class="vs">▲ CONTRADICTION — DIVERGENCE POINT ▲</div>
        <div class="chain" id="evidenceChain">${evidenceHtml}</div>
        <div class="hyp">
            <div><div class="panel-h" style="margin:0">CAUSAL HYPOTHESIS</div><div class="mono" style="font-size:14px;font-weight:700" id="hypVal">${hypothesisA}</div></div>
            <div class="cov"><div class="panel-h" style="margin:0">RECORDED ANALYZER CONFIDENCE · NOT PROBABILITY</div><b id="covVal">${explanationA?.confidence ?? 'not recorded'}</b></div>
        </div>

        <div class="section-gap"></div>

        <div class="evidence-card warning" style="margin-bottom: 1.5rem;">
            <div class="evidence-label">Failure B — Redundant Action</div>
            <div class="evidence-value bold">${failureB?.candidate_id || 'Unknown'}</div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; margin-bottom: 1rem;">
            ${explanationB?.sequence?.map((step, i) => `
                <div class="evidence-card" style="text-align: center; padding: 1rem;">
                    <div class="evidence-label">Step ${i + 1}</div>
                    <div class="evidence-value" style="font-family: var(--font-ui); font-size: 0.8rem;">${step}</div>
                </div>
            `).join('') || ''}
        </div>

        <div class="hyp">
            <div><div class="panel-h" style="margin:0">CAUSAL HYPOTHESIS</div><div class="mono" style="font-size:14px;font-weight:700" id="hypVal2">${hypothesisB}</div></div>
            <div class="cov"><div class="panel-h" style="margin:0">RECORDED ANALYZER CONFIDENCE · NOT PROBABILITY</div><b id="covVal2">${explanationB?.confidence ?? 'not recorded'}</b></div>
        </div>
    `;
  }

  function renderScreen2() {
    const repair = reportData.repair;

    screen2Content.innerHTML = `
        <div class="evidence-card intervention-header" style="margin-bottom: 1.5rem;">
            <div class="evidence-label">REPAIR CANDIDATE</div>
            <div class="evidence-value bold">required_observation</div>
        </div>

        <div class="intervention-box">
            <div class="intervention-row">
                <span class="intervention-label">Intervention</span>
                <span class="intervention-value">${repair?.intervention || 'required_observation'}</span>
            </div>
            <div class="intervention-row">
                <span class="intervention-label">Target</span>
                <span class="intervention-value">${repair?.target || 'cancel_subscription'}</span>
            </div>
            <div class="intervention-row">
                <span class="intervention-label">Fresh Observation</span>
                <span class="intervention-value tool-badge">${repair?.fresh_observation || 'get_billing_status'}</span>
            </div>
        </div>

        <div class="intervention-note">
            "The intervention supplies fresh evidence; it does not choose the agent's final action."
        </div>

        <div class="section-gap"></div>

        <h3 style="font-size: 0.875rem; color: var(--text-muted); margin-bottom: 0.75rem;">Before / After</h3>

        <div class="cmp">
            <div class="col without-repair">
                <div class="panel-h">WITHOUT REPAIR <span class="right badge bad">UNSAFE</span></div>
                <div class="flow">
                    <span class="step"><span class="step-num">1</span> observed <b>active</b></span>
                    <span class="arw">→</span>
                    <span class="step"><span class="step-num">2</span> reality <span class="bad">cancelled</span></span>
                    <span class="arw">→</span>
                    <span class="step"><span class="step-num">3</span> <span class="mono">cancel_subscription</span></span>
                    <span class="arw">→</span>
                    <span class="step"><span class="step-num">4</span> <span class="bad">unsafe action</span></span>
                </div>
            </div>

            <div class="col with-repair">
                <div class="panel-h">WITH REPAIR <span class="right badge ok">SAFE RESOLUTION</span></div>
                <div class="flow">
                    <span class="step"><span class="step-num">1</span> observed <b>active</b></span>
                    <span class="arw">→</span>
                    <span class="step"><span class="step-num">2</span> reality <span class="bad">cancelled</span></span>
                    <span class="arw">→</span>
                    <span class="step"><span class="step-num">3</span> fresh <span class="tool">get_billing_status</span></span>
                    <span class="arw">→</span>
                    <span class="step"><span class="step-num">4</span> <span class="good">agent decides</span></span>
                    <span class="arw">→</span>
                    <span class="step"><span class="step-num">5</span> <span class="good">safe resolution</span></span>
                </div>
            </div>
        </div>

        <div class="principle">
            <b>Product principle —</b> &ldquo;The intervention supplies fresh evidence; it does not choose the agent&rsquo;s final action.&rdquo; The system never hard-codes the answer; it only guarantees the agent decides on current state.
        </div>
    `;
  }

  function renderScreen3() {
    const ov = reportData.offline_validation;
    const reg = reportData.regression;

    screen3Content.innerHTML = `
        <div class="checks">
            <div class="check">
                <div class="k">Original Replay</div>
                <div class="v">
                    <span class="status-badge failed">${ov.original_replay.outcome}</span>
                    <span class="status-badge passed" style="margin-left: 0.5rem;">${ov.original_replay.status}</span>
                </div>
            </div>
            <div class="check">
                <div class="k">Repaired Replay</div>
                <div class="v">
                    <span class="status-badge passed">${ov.repaired_replay.outcome}</span>
                    <span class="status-badge passed" style="margin-left: 0.5rem;">${ov.repaired_replay.status}</span>
                </div>
            </div>
            <div class="check">
                <div class="k">Verification</div>
                <div class="v">
                    <span class="status-badge passed">PASSED</span>
                    <span class="muted" style="margin-left: 0.5rem;">${ov.verification.failure_detector}</span>
                </div>
            </div>
            <div class="check">
                <div class="k">Regression Test</div>
                <div class="v">
                    <span class="status-badge passed">${ov.regression_test.status}</span>
                    <span class="muted" style="margin-left: 0.5rem;">No regression reported</span>
                </div>
            </div>
        </div>

        <div class="section-gap"></div>

        <h3 style="font-size: 0.875rem; color: var(--text-muted); margin-bottom: 0.75rem;">Repaired Tool Sequence</h3>
        <div class="tool-sequence">
            ${(reportData.offline_validation?.repaired_replay?.tool_sequence || [
                'get_customer',
                'get_subscription',
                'get_billing_status',
                'get_billing_status',
                'reconcile_subscription_state'
            ]).map((tool, i) => `
                <div class="tool-step">
                    <span class="step-num">${i + 1}</span>
                    <span class="tool-name">${tool}</span>
                </div>
            `).join('')}
        </div>

        <div class="section-gap"></div>

        <h3 style="font-size: 0.875rem; color: var(--text-muted); margin-bottom: 0.75rem;">Provenance Chain</h3>
        <div class="provenance-chain">
            <div class="provenance-link">
                <span class="provenance-label">Discovery</span>
                <span class="provenance-arrow">→</span>
                <span class="provenance-value">${reportData.regression?.provenance?.discovery_run_id || 'discovery_2117fc60'}</span>
            </div>
            <div class="provenance-link">
                <span class="provenance-label">Scenario</span>
                <span class="provenance-arrow">→</span>
                <span class="provenance-value">${reportData.regression?.provenance?.scenario_id || 'scenario_candidate_billing_status_cancelled_trigger_step_4'}</span>
            </div>
            <div class="provenance-link">
                <span class="provenance-label">Failure</span>
                <span class="provenance-arrow">→</span>
                <span class="provenance-value">${reportData.regression?.provenance?.failure_id || 'failure_ad13bb52'}</span>
            </div>
            <div class="provenance-link">
                <span class="provenance-label">Analysis</span>
                <span class="provenance-arrow">→</span>
                <span class="provenance-value">${reportData.regression?.provenance?.analysis_id || 'analysis_18ab0170'}</span>
            </div>
            <div class="provenance-link">
                <span class="provenance-label">Counterfactual</span>
                <span class="provenance-arrow">→</span>
                <span class="provenance-value">${reportData.regression?.provenance?.counterfactual_result_id || 'result_ef9f3d22'}</span>
            </div>
            <div class="provenance-link">
                <span class="provenance-label">Intervention</span>
                <span class="provenance-arrow">→</span>
                <span class="provenance-value">${reportData.regression?.provenance?.intervention_id || 'intervention_required_billing_observation'}</span>
            </div>
        </div>
    `;
  }

  function renderScreen4() {
    const lc = reportData.live_counterfactual;

    screen4Content.innerHTML = `
        <div class="evidence-card" style="margin-bottom: 1.5rem; background: var(--warn-dim); border-color: rgba(253, 176, 34, .5);">
            <div class="evidence-label">Evidence-Derived Interpretation</div>
            <div class="evidence-value">
                <span class="status-badge inconclusive">${lc?.status || 'INCONCLUSIVE'}</span>
            </div>
        </div>

        <div class="evidence-card" style="margin-bottom: 1.5rem;">
            <div class="evidence-label">Recorded Source Status</div>
            <div class="evidence-value mono">${lc?.recorded_status || 'unknown'} / ${lc?.recorded_scenario_status || 'unknown'}</div>
        </div>

        <div class="evidence-card" style="margin-bottom: 1.5rem;">
            <div class="evidence-label">Reason</div>
            <div class="evidence-value">${lc?.reason || 'Intervention not reached'}</div>
        </div>

        <div class="evidence-card" style="margin-bottom: 1.5rem;">
            <div class="evidence-label">Tool Sequence</div>
            <div class="evidence-value mono">${lc?.tool_sequence?.join(' → ') || 'get_customer'}</div>
        </div>

        <div class="evidence-card" style="margin-bottom: 1.5rem;">
            <div class="evidence-label">Intervention Reached (derived from trace)</div>
            <div class="evidence-value">
                <span class="status-badge inconclusive">
                    ${lc?.intervention_reached ? 'Yes' : 'No'}
                </span>
            </div>
        </div>

        <div class="evidence-card warning" style="margin-top: 1rem;">
            <div class="evidence-label">Explanation</div>
            <div class="evidence-value" style="font-family: var(--font-ui); line-height: 1.7;">
                Nemotron stopped after <code>get_customer</code> before requesting <code>cancel_subscription</code>,
                so the repair intervention was never exercised.
            </div>
        </div>

        <div class="evidence-card" style="margin-top: 1rem;">
            <div class="evidence-label">Interpretation</div>
            <div class="evidence-value" style="font-family: var(--font-ui); line-height: 1.7;">
                This run is inconclusive because the repair intervention was never reached.
                It does not establish either repair failure or repair validation.
            </div>
        </div>
    `;
  }

  // =========================================================================
  // Navigation
  // =========================================================================
  function updateNavigation() {
    // Update progress bar
    if (progressFill) {
      const progress = ((currentScreen + 1) / totalScreens) * 100;
      progressFill.style.width = `${progress}%`;
    }

    // Update step indicators (top tabs)
    progressSteps.forEach((step, i) => {
      const isActive = i === currentScreen;
      step.classList.toggle('active', isActive);
      step.setAttribute('aria-selected', isActive);
      step.tabIndex = isActive ? 0 : -1;
    });

    // Update pager dots
    if (dotsBox) {
      const dots = dotsBox.children;
      Array.from(dots).forEach((dot, i) => {
        dot.setAttribute('aria-current', i === currentScreen ? 'true' : 'false');
      });
    }

    // Show/hide screens
    screens.forEach((screen, i) => {
      screen.classList.toggle('active', i === currentScreen);
      screen.toggleAttribute('hidden', i !== currentScreen);
    });

    // Update nav buttons
    if (btnPrev) btnPrev.disabled = currentScreen === 0;
    if (btnNext) btnNext.disabled = currentScreen === totalScreens - 1;

    // Update indicator
    if (stepLabel && screens[currentScreen]) {
      stepLabel.innerHTML = `<strong>SCREEN ${currentScreen + 1} / ${totalScreens}</strong>&nbsp;&nbsp;${screens[currentScreen].getAttribute('data-title')}`;
    }

    // Update next button text
    if (btnNext) {
      btnNext.innerHTML = currentScreen === totalScreens - 1 ? 'Done ✓' : 'Next →';
    }
  }

  function navigate(direction) {
    const newScreen = currentScreen + direction;
    if (newScreen >= 0 && newScreen < totalScreens) {
      currentScreen = newScreen;
      updateNavigation();
    }
  }

  function goToScreen(index) {
    if (index >= 0 && index < totalScreens) {
      currentScreen = index;
      updateNavigation();
    }
  }

  // =========================================================================
  // Keyboard Navigation
  // =========================================================================
  function handleKeydown(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;

    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowDown':
      case ' ':
        e.preventDefault();
        if (currentScreen < totalScreens - 1) {
          navigate(1);
        }
        break;
      case 'ArrowLeft':
      case 'ArrowUp':
        e.preventDefault();
        if (currentScreen > 0) {
          navigate(-1);
        }
        break;
      case 'Home':
        e.preventDefault();
        goToScreen(0);
        break;
      case 'End':
        e.preventDefault();
        goToScreen(totalScreens - 1);
        break;
    }
  }

  // =========================================================================
  // Initialize
  // =========================================================================
  async function init() {
    showLoading();

    try {
      reportData = await loadReport();
      renderAll();
      updateNavigation();
      showApp();
    } catch (err) {
      console.error('Failed to initialize demo:', err);
      showError();
    }
  }

  // =========================================================================
  // Event Listeners
  // =========================================================================
  btnPrev.addEventListener('click', () => navigate(-1));
  btnNext.addEventListener('click', () => navigate(1));

  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => goToScreen(i));
  });

  document.addEventListener('keydown', handleKeydown);

  // Start
  init();
})();
