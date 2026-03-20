/* ── AlphaSignal v2 — frontend logic ──────────────────────────────── */

let charts = {};

const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 600 },
  plugins: {
    legend: { display: false },
    tooltip: {
      backgroundColor: '#1d2130',
      borderColor: '#272d3d',
      borderWidth: 1,
      titleColor: '#94a3b8',
      bodyColor: '#e2e8f0',
      padding: 10,
    },
  },
  scales: {
    x: {
      grid: { color: '#1e2436', drawBorder: false },
      ticks: { color: '#64748b', maxTicksLimit: 10, maxRotation: 0, font: { size: 10 } },
    },
    y: {
      grid: { color: '#1e2436', drawBorder: false },
      ticks: { color: '#64748b', font: { size: 10 } },
    },
  },
};

function destroyAll() {
  Object.values(charts).forEach(c => c && c.destroy());
  charts = {};
}

/* ── State transitions ─────────────────────────────────────────────── */
function showOnly(id) {
  ['landingState','loadingState','errorState','resultsState'].forEach(s => {
    document.getElementById(s).classList.toggle('hidden', s !== id);
  });
}

function reset() {
  destroyAll();
  showOnly('landingState');
  document.getElementById('tickerInput').value = '';
}

function quickAnalyze(ticker) {
  document.getElementById('tickerInput').value = ticker;
  runAnalysis();
}

/* ── Tab switching ─────────────────────────────────────────────────── */
function switchTab(tabName) {
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tabName));
  document.querySelectorAll('.tab-content').forEach(tc => tc.classList.toggle('active', tc.id === 'tab-' + tabName));
}

/* ── Loading animation ─────────────────────────────────────────────── */
let loadingTimer = null;

function animateLoading() {
  const steps = ['step1','step2','step3','step4','step5'];
  const msgs = [
    'Fetching market data...',
    'Calculating 18+ indicators...',
    'Training Random Forest + Gradient Boosting...',
    'Running out-of-sample backtest...',
    'Computing AEMI entropy signal...',
    'Almost there...',
  ];

  steps.forEach(id => {
    const el = document.getElementById(id);
    el.className = 'step';
  });

  let msgIdx = 0;
  function cycleMsg() {
    document.getElementById('loadingText').textContent = msgs[msgIdx % msgs.length];
    msgIdx++;
    loadingTimer = setTimeout(cycleMsg, 2200);
  }
  cycleMsg();

  const delays = [0, 1500, 3500, 6000, 8500];
  steps.forEach((id, i) => {
    setTimeout(() => {
      const el = document.getElementById(id);
      if (el) {
        el.className = 'step done';
      }
    }, delays[i]);
  });
}

/* ── Main analysis ─────────────────────────────────────────────────── */
async function runAnalysis() {
  const ticker = document.getElementById('tickerInput').value.trim().toUpperCase();
  if (!ticker) {
    document.getElementById('tickerInput').focus();
    return;
  }

  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  showOnly('loadingState');
  destroyAll();
  if (loadingTimer) clearTimeout(loadingTimer);
  animateLoading();

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ticker,
        period: document.getElementById('periodSelect').value,
        horizon: parseInt(document.getElementById('horizonSelect').value),
      }),
    });

    const data = await resp.json();

    if (!resp.ok || data.error) {
      throw new Error(data.error || 'Unknown server error');
    }

    clearTimeout(loadingTimer);
    renderResults(data);

  } catch (err) {
    clearTimeout(loadingTimer);
    document.getElementById('errorMessage').textContent = err.message;
    showOnly('errorState');
  } finally {
    btn.disabled = false;
  }
}

/* ── Key-press on input ────────────────────────────────────────────── */
document.getElementById('tickerInput').addEventListener('keydown', e => {
  if (e.key === 'Enter') runAnalysis();
});

/* ── Render ─────────────────────────────────────────────────────────── */
function renderResults(d) {
  showOnly('resultsState');
  switchTab('overview');

  // Header
  document.getElementById('resTickerName').textContent = d.ticker;
  document.getElementById('resPrice').textContent = '$' + (d.current_price || 0).toLocaleString('en-US', { minimumFractionDigits: 2 });
  document.getElementById('resHorizon').textContent = d.horizon + '-Day Prediction';

  // Signal hero
  const isBull = d.signal_direction === 'BULLISH';
  const sdEl = document.getElementById('signalDirection');
  sdEl.textContent = d.signal_direction;
  sdEl.className = 'signal-direction ' + (isBull ? 'bullish' : 'bearish');

  document.getElementById('signalProb').textContent =
    d.signal + '% probability of being higher in ' + d.horizon + ' trading days';
  document.getElementById('resAccuracy').textContent = d.accuracy + '%';
  document.getElementById('resStrength').textContent = d.signal_strength + '%';

  const regimeMap = { '-1': 'Bear Trend', '0': 'Ranging', '1': 'Bull Trend' };
  const regimeEl = document.getElementById('resRegime');
  regimeEl.textContent = regimeMap[String(d.regime)] || 'Neutral';
  regimeEl.className = 'sstat-value ' + (d.regime > 0 ? 'green' : d.regime < 0 ? 'red' : 'yellow');

  const fractalEl = document.getElementById('resFractal');
  fractalEl.textContent = d.fractal_dim ?? '--';

  // Hurst Exponent
  const hurstEl = document.getElementById('resHurst');
  const h = d.hurst;
  hurstEl.textContent = h ?? '--';
  hurstEl.className = 'sstat-value ' + (h > 0.55 ? 'green' : h < 0.45 ? 'cyan' : 'yellow');

  // Backtest win rate in hero
  const bt = d.backtest;
  const winRate = bt.total_predictions > 0 ? (bt.correct_predictions / bt.total_predictions * 100).toFixed(1) : '--';
  const wrEl = document.getElementById('resWinRate');
  wrEl.textContent = winRate + '%';
  wrEl.className = 'sstat-value ' + (parseFloat(winRate) > 55 ? 'green' : parseFloat(winRate) < 45 ? 'red' : 'yellow');

  // Confidence-filtered win rate
  const confWinEl = document.getElementById('resConfWin');
  confWinEl.textContent = bt.conf_filtered_accuracy + '%';
  confWinEl.className = 'sstat-value ' + (bt.conf_filtered_accuracy > 55 ? 'green' : bt.conf_filtered_accuracy < 45 ? 'red' : 'yellow');

  // Features selected
  document.getElementById('resFeatures').textContent = (d.selected_features ? d.selected_features.length : '--') + '/' + (d.total_features || '--');

  // Gauge
  drawGauge(d.signal, isBull);
  document.getElementById('gaugeLabel').textContent = d.signal + '%';
  document.getElementById('gaugeLabel').style.color = isBull ? '#22c55e' : '#ef4444';

  // Overview charts
  buildPriceChart(d.price_history);
  buildPredChart(d.dates, d.predictions);
  buildAemiChart(d.price_history);
  buildRsiChart(d.price_history);
  buildMacdChart(d.price_history);
  buildStochChart(d.price_history);
  buildVolChart(d.price_history);
  buildFeatChart(d.feature_importance);

  // Backtest tab
  renderBacktest(d);

  // Indicators tab
  populateIndicators(d);
}

/* ── Gauge canvas ───────────────────────────────────────────────────── */
function drawGauge(pct, isBull) {
  const canvas = document.getElementById('gaugeCanvas');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2, cy = H - 8;
  const r = H - 20;

  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, 0, false);
  ctx.strokeStyle = '#272d3d';
  ctx.lineWidth = 14;
  ctx.lineCap = 'round';
  ctx.stroke();

  const angle = Math.PI + (pct / 100) * Math.PI;
  const grad = ctx.createLinearGradient(cx - r, cy, cx + r, cy);
  grad.addColorStop(0, '#ef4444');
  grad.addColorStop(.5, '#f59e0b');
  grad.addColorStop(1, '#22c55e');

  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, angle, false);
  ctx.strokeStyle = grad;
  ctx.lineWidth = 14;
  ctx.lineCap = 'round';
  ctx.stroke();

  const nx = cx + (r - 20) * Math.cos(angle);
  const ny = cy + (r - 20) * Math.sin(angle);
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(nx, ny);
  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.beginPath();
  ctx.arc(cx, cy, 4, 0, Math.PI * 2);
  ctx.fillStyle = '#e2e8f0';
  ctx.fill();
}

/* ── Overview Chart builders ───────────────────────────────────────── */
function buildPriceChart(ph) {
  const ctx = document.getElementById('priceChart').getContext('2d');
  charts.price = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ph.dates,
      datasets: [
        { label: 'Close', data: ph.close, borderColor: '#6366f1', borderWidth: 2, pointRadius: 0, tension: .3, fill: false, order: 1 },
        { label: 'SMA 20', data: ph.sma20, borderColor: '#06b6d4', borderWidth: 1.5, borderDash: [4,4], pointRadius: 0, fill: false, order: 2 },
        { label: 'SMA 50', data: ph.sma50, borderColor: '#f59e0b', borderWidth: 1.5, borderDash: [4,4], pointRadius: 0, fill: false, order: 3 },
        { label: 'BB Upper', data: ph.bb_upper, borderColor: 'rgba(99,102,241,.25)', borderWidth: 1, pointRadius: 0, fill: '+1', backgroundColor: 'rgba(99,102,241,.04)', order: 4 },
        { label: 'BB Lower', data: ph.bb_lower, borderColor: 'rgba(99,102,241,.25)', borderWidth: 1, pointRadius: 0, fill: false, order: 5 },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: { ...CHART_DEFAULTS.plugins, legend: { display: true, labels: { color: '#64748b', boxWidth: 12, font: { size: 10 } } } },
    },
  });
}

function buildAemiChart(ph) {
  const ctx = document.getElementById('aemiChart').getContext('2d');
  const data = ph.aemi;
  const colors = data.map(v => v === null ? 'transparent' : v > 0 ? 'rgba(34,197,94,.7)' : 'rgba(239,68,68,.7)');
  charts.aemi = new Chart(ctx, {
    type: 'bar',
    data: { labels: ph.dates, datasets: [{ label: 'AEMI', data, backgroundColor: colors, borderWidth: 0, borderRadius: 2 }] },
    options: CHART_DEFAULTS,
  });
}

function buildRsiChart(ph) {
  const ctx = document.getElementById('rsiChart').getContext('2d');
  charts.rsi = new Chart(ctx, {
    type: 'line',
    data: { labels: ph.dates, datasets: [{ label: 'RSI', data: ph.rsi, borderColor: '#818cf8', borderWidth: 2, pointRadius: 0, tension: .3, fill: false }] },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        ...CHART_DEFAULTS.scales,
        y: { ...CHART_DEFAULTS.scales.y, min: 0, max: 100, ticks: { color: '#64748b', font: { size: 10 }, stepSize: 25 } },
      },
    },
  });
}

function buildMacdChart(ph) {
  const ctx = document.getElementById('macdChart').getContext('2d');
  const histColors = ph.macd_hist.map(v => v === null ? 'transparent' : v > 0 ? 'rgba(34,197,94,.5)' : 'rgba(239,68,68,.5)');
  charts.macd = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ph.dates,
      datasets: [
        { type: 'line', label: 'MACD', data: ph.macd, borderColor: '#6366f1', borderWidth: 1.5, pointRadius: 0, tension: .3, fill: false, order: 1 },
        { type: 'line', label: 'Signal', data: ph.macd_signal, borderColor: '#f59e0b', borderWidth: 1.5, pointRadius: 0, tension: .3, fill: false, order: 2 },
        { label: 'Histogram', data: ph.macd_hist, backgroundColor: histColors, borderWidth: 0, borderRadius: 1, order: 3 },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: { ...CHART_DEFAULTS.plugins, legend: { display: true, labels: { color: '#64748b', boxWidth: 10, font: { size: 10 } } } },
    },
  });
}

function buildStochChart(ph) {
  const ctx = document.getElementById('stochChart').getContext('2d');
  charts.stoch = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ph.dates,
      datasets: [
        { label: '%K', data: ph.stoch_k, borderColor: '#6366f1', borderWidth: 1.5, pointRadius: 0, tension: .3, fill: false },
        { label: '%D', data: ph.stoch_d, borderColor: '#f59e0b', borderWidth: 1.5, pointRadius: 0, tension: .3, fill: false },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        ...CHART_DEFAULTS.scales,
        y: { ...CHART_DEFAULTS.scales.y, min: 0, max: 100, ticks: { color: '#64748b', font: { size: 10 }, stepSize: 25 } },
      },
      plugins: { ...CHART_DEFAULTS.plugins, legend: { display: true, labels: { color: '#64748b', boxWidth: 10, font: { size: 10 } } } },
    },
  });
}

function buildPredChart(dates, predictions) {
  const ctx = document.getElementById('predChart').getContext('2d');
  const colors = predictions.map(p => p >= 50 ? 'rgba(34,197,94,.7)' : 'rgba(239,68,68,.7)');
  charts.pred = new Chart(ctx, {
    type: 'bar',
    data: { labels: dates, datasets: [{ label: 'Bullish %', data: predictions, backgroundColor: colors, borderWidth: 0, borderRadius: 2 }] },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        ...CHART_DEFAULTS.scales,
        y: { ...CHART_DEFAULTS.scales.y, min: 0, max: 100, ticks: { color: '#64748b', font: { size: 10 }, callback: v => v + '%' } },
      },
    },
  });
}

function buildVolChart(ph) {
  const ctx = document.getElementById('volChart').getContext('2d');
  const avg = ph.volume.reduce((a, b) => a + b, 0) / ph.volume.length;
  const colors = ph.volume.map(v => v > avg ? 'rgba(99,102,241,.7)' : 'rgba(99,102,241,.3)');
  charts.vol = new Chart(ctx, {
    type: 'bar',
    data: { labels: ph.dates, datasets: [{ label: 'Volume', data: ph.volume, backgroundColor: colors, borderWidth: 0, borderRadius: 2 }] },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        ...CHART_DEFAULTS.scales,
        y: { ...CHART_DEFAULTS.scales.y, ticks: { color: '#64748b', font: { size: 10 }, callback: v => v >= 1e9 ? (v/1e9).toFixed(1)+'B' : v >= 1e6 ? (v/1e6).toFixed(0)+'M' : v } },
      },
    },
  });
}

function buildFeatChart(imp) {
  const labels = Object.keys(imp);
  const values = Object.values(imp).map(v => +(v * 100).toFixed(2));
  const ctx = document.getElementById('featChart').getContext('2d');
  charts.feat = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Importance %',
        data: values,
        backgroundColor: labels.map((l, i) => {
          if (l.startsWith('AEMI')) return 'rgba(6,182,212,.8)';
          if (l.startsWith('Hurst')) return 'rgba(34,197,94,.8)';
          if (l === 'OFI' || l.startsWith('MTF') || l.startsWith('ZScore') || l.startsWith('Autocorr')) return 'rgba(245,158,11,.8)';
          return `rgba(99,102,241,${Math.max(0.2, 0.8 - i * 0.04)})`;
        }),
        borderWidth: 0,
        borderRadius: 2,
      }],
    },
    options: {
      ...CHART_DEFAULTS,
      indexAxis: 'y',
      scales: {
        x: { grid: { color: '#1e2436' }, ticks: { color: '#64748b', font: { size: 10 }, callback: v => v + '%' } },
        y: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 10, family: "'JetBrains Mono', monospace" } } },
      },
    },
  });
}

/* ── Backtest rendering ────────────────────────────────────────────── */
function renderBacktest(d) {
  const bt = d.backtest;
  const cm = bt.confusion_matrix;
  const cv = d.cv_metrics;
  const total = bt.total_predictions;
  const correct = bt.correct_predictions;
  const winRate = total > 0 ? (correct / total * 100).toFixed(1) : '--';

  // Summary cards
  document.getElementById('btTotal').textContent = total;
  document.getElementById('btCorrect').textContent = correct;
  document.getElementById('btWinRate').textContent = winRate + '%';
  document.getElementById('btWinRate').className = 'bt-card-value ' + (parseFloat(winRate) > 55 ? 'green' : parseFloat(winRate) < 45 ? 'red' : 'yellow');

  const avgPrec = cv.precision.reduce((a,b) => a+b, 0) / cv.precision.length;
  const avgRec = cv.recall.reduce((a,b) => a+b, 0) / cv.recall.length;
  const avgF1 = cv.f1.reduce((a,b) => a+b, 0) / cv.f1.length;
  document.getElementById('btPrecision').textContent = avgPrec.toFixed(1) + '%';
  document.getElementById('btRecall').textContent = avgRec.toFixed(1) + '%';
  document.getElementById('btF1').textContent = avgF1.toFixed(1) + '%';

  document.getElementById('btMaxWin').textContent = bt.streaks.max_win;
  document.getElementById('btMaxLoss').textContent = bt.streaks.max_loss;

  // Confidence-filtered stats
  document.getElementById('btConfTotal').textContent = bt.conf_filtered_total || 0;
  document.getElementById('btConfCorrect').textContent = bt.conf_filtered_correct || 0;
  const cfwr = bt.conf_filtered_accuracy || 0;
  document.getElementById('btConfWinRate').textContent = cfwr + '%';
  document.getElementById('btConfWinRate').className = 'bt-card-value ' + (cfwr > 55 ? 'green' : cfwr < 45 ? 'red' : 'yellow');

  // Confusion matrix
  document.getElementById('cmTP').textContent = cm.tp;
  document.getElementById('cmFN').textContent = cm.fn;
  document.getElementById('cmFP').textContent = cm.fp;
  document.getElementById('cmTN').textContent = cm.tn;

  // Accuracy chart (predicted vs actual scatter)
  buildBtAccuracyChart(bt);
  buildBtEquityChart(bt);
  buildBtRollingChart(bt);
  buildBtConfidenceChart(bt);

  // CV bars
  renderCVMetricBars('cvAccBars', cv.accuracy);
  renderCVMetricBars('cvPrecBars', cv.precision);
  renderCVMetricBars('cvRecBars', cv.recall);
  renderCVMetricBars('cvF1Bars', cv.f1);
}

function buildBtAccuracyChart(bt) {
  const ctx = document.getElementById('btAccuracyChart').getContext('2d');
  // Color each bar: green if prediction was correct, red if wrong
  const barColors = bt.correct.map(c => c ? 'rgba(34,197,94,.6)' : 'rgba(239,68,68,.6)');
  // Show probability, with correct/incorrect coloring
  charts.btAccuracy = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: bt.dates,
      datasets: [
        {
          label: 'Prediction Confidence',
          data: bt.probabilities,
          backgroundColor: barColors,
          borderWidth: 0,
          borderRadius: 1,
        },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: {
        ...CHART_DEFAULTS.plugins,
        legend: { display: false },
        tooltip: {
          ...CHART_DEFAULTS.plugins.tooltip,
          callbacks: {
            label: function(context) {
              const i = context.dataIndex;
              const pred = bt.predictions[i] === 1 ? 'UP' : 'DOWN';
              const actual = bt.actuals[i] === 1 ? 'UP' : 'DOWN';
              const correct = bt.correct[i] ? 'CORRECT' : 'WRONG';
              return [
                'Confidence: ' + bt.probabilities[i] + '%',
                'Predicted: ' + pred,
                'Actual: ' + actual,
                correct,
              ];
            }
          }
        }
      },
      scales: {
        ...CHART_DEFAULTS.scales,
        y: {
          ...CHART_DEFAULTS.scales.y,
          min: 0, max: 100,
          ticks: { color: '#64748b', font: { size: 10 }, callback: v => v + '%' },
        },
      },
    },
  });
}

function buildBtEquityChart(bt) {
  const ctx = document.getElementById('btEquityChart').getContext('2d');
  charts.btEquity = new Chart(ctx, {
    type: 'line',
    data: {
      labels: bt.dates,
      datasets: [
        {
          label: 'Model Strategy',
          data: bt.equity_curve,
          borderColor: '#6366f1',
          borderWidth: 2,
          pointRadius: 0,
          tension: .3,
          fill: true,
          backgroundColor: 'rgba(99,102,241,.08)',
        },
        {
          label: 'Buy & Hold',
          data: bt.bh_equity_curve,
          borderColor: '#64748b',
          borderWidth: 1.5,
          borderDash: [5,5],
          pointRadius: 0,
          tension: .3,
          fill: false,
        },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: {
        ...CHART_DEFAULTS.plugins,
        legend: {
          display: true,
          labels: { color: '#94a3b8', boxWidth: 12, font: { size: 11 } },
        },
        tooltip: {
          ...CHART_DEFAULTS.plugins.tooltip,
          callbacks: {
            label: function(context) {
              return context.dataset.label + ': $' + context.parsed.y.toFixed(2);
            }
          }
        }
      },
      scales: {
        ...CHART_DEFAULTS.scales,
        y: {
          ...CHART_DEFAULTS.scales.y,
          ticks: { color: '#64748b', font: { size: 10 }, callback: v => '$' + v.toFixed(0) },
        },
      },
    },
  });
}

function buildBtRollingChart(bt) {
  const ctx = document.getElementById('btRollingChart').getContext('2d');
  const colors = bt.rolling_accuracy.map(v => v >= 55 ? 'rgba(34,197,94,.7)' : v >= 45 ? 'rgba(245,158,11,.7)' : 'rgba(239,68,68,.7)');
  charts.btRolling = new Chart(ctx, {
    type: 'line',
    data: {
      labels: bt.dates,
      datasets: [{
        label: 'Rolling Accuracy %',
        data: bt.rolling_accuracy,
        borderColor: '#818cf8',
        borderWidth: 2,
        pointRadius: 0,
        tension: .3,
        fill: true,
        backgroundColor: 'rgba(129,140,248,.08)',
      }],
    },
    options: {
      ...CHART_DEFAULTS,
      scales: {
        ...CHART_DEFAULTS.scales,
        y: {
          ...CHART_DEFAULTS.scales.y,
          min: 0, max: 100,
          ticks: { color: '#64748b', font: { size: 10 }, callback: v => v + '%' },
        },
      },
    },
  });
}

function buildBtConfidenceChart(bt) {
  const ctx = document.getElementById('btConfidenceChart').getContext('2d');
  // Histogram of confidence levels, split by correct/incorrect
  const bins = [0, 30, 40, 45, 50, 55, 60, 70, 100];
  const binLabels = bins.slice(0, -1).map((b, i) => b + '-' + bins[i+1] + '%');
  const correctCounts = new Array(bins.length - 1).fill(0);
  const wrongCounts = new Array(bins.length - 1).fill(0);

  for (let i = 0; i < bt.probabilities.length; i++) {
    const p = bt.probabilities[i];
    // Map probability to effective confidence (distance from 50%)
    const conf = p;
    for (let j = 0; j < bins.length - 1; j++) {
      if (conf >= bins[j] && conf < bins[j+1]) {
        if (bt.correct[i]) correctCounts[j]++;
        else wrongCounts[j]++;
        break;
      }
    }
    // Handle 100%
    if (conf >= 100) {
      if (bt.correct[i]) correctCounts[correctCounts.length - 1]++;
      else wrongCounts[wrongCounts.length - 1]++;
    }
  }

  charts.btConfidence = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: binLabels,
      datasets: [
        { label: 'Correct', data: correctCounts, backgroundColor: 'rgba(34,197,94,.6)', borderWidth: 0, borderRadius: 2 },
        { label: 'Wrong', data: wrongCounts, backgroundColor: 'rgba(239,68,68,.6)', borderWidth: 0, borderRadius: 2 },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: {
        ...CHART_DEFAULTS.plugins,
        legend: { display: true, labels: { color: '#94a3b8', boxWidth: 10, font: { size: 10 } } },
      },
      scales: {
        x: { grid: { color: '#1e2436' }, ticks: { color: '#64748b', font: { size: 10 } }, stacked: true },
        y: { grid: { color: '#1e2436' }, ticks: { color: '#64748b', font: { size: 10 } }, stacked: true },
      },
    },
  });
}

/* ── CV metric bars ────────────────────────────────────────────────── */
function renderCVMetricBars(containerId, scores) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  scores.forEach((s, i) => {
    const wrap = document.createElement('div');
    wrap.className = 'cv-bar-wrap';
    wrap.innerHTML = `
      <div class="cv-pct">${s}%</div>
      <div class="cv-bar" style="height:${s * 0.7}%;"></div>
      <div class="cv-fold">F${i + 1}</div>
    `;
    container.appendChild(wrap);
  });
}

/* ── Indicator cards ────────────────────────────────────────────────── */
function populateIndicators(d) {
  // AEMI
  const aemi = d.aemi;
  document.getElementById('indAemi').textContent = aemi !== null ? aemi.toFixed(5) : '--';
  document.getElementById('indAemi').className = 'ind-value ' + (aemi > 0 ? 'green' : aemi < 0 ? 'red' : '');
  document.getElementById('indAemiHint').textContent =
    aemi > 0.001 ? 'High-confidence bullish' :
    aemi < -0.001 ? 'High-confidence bearish' :
    'Low predictability - wait for clarity';

  // RSI
  const rsi = d.rsi;
  document.getElementById('indRsi').textContent = rsi ?? '--';
  document.getElementById('indRsi').className = 'ind-value ' + colorClass(rsi, 70, 30);
  setBar('indRsiBar', rsi, 0, 100, rsi > 70 ? '#ef4444' : rsi < 30 ? '#22c55e' : '#6366f1');
  document.getElementById('indRsiHint').textContent = rsi > 70 ? 'Overbought' : rsi < 30 ? 'Oversold' : 'Neutral';

  // MACD
  const macd = d.macd;
  document.getElementById('indMacd').textContent = macd ?? '--';
  document.getElementById('indMacd').className = 'ind-value ' + (macd > 0 ? 'green' : 'red');
  document.getElementById('indMacdHint').textContent = macd > 0 ? 'Bullish crossover zone' : 'Bearish crossover zone';

  // Stochastic
  const stk = d.stoch_k, std = d.stoch_d;
  document.getElementById('indStoch').textContent = stk !== null ? stk.toFixed(1) + ' / ' + (std || 0).toFixed(1) : '--';
  document.getElementById('indStoch').className = 'ind-value ' + colorClass(stk, 80, 20);
  setBar('indStochBar', stk, 0, 100, stk > 80 ? '#ef4444' : stk < 20 ? '#22c55e' : '#6366f1');
  document.getElementById('indStochHint').textContent = stk > 80 ? 'Overbought' : stk < 20 ? 'Oversold' : 'Neutral';

  // Williams %R
  const wr = d.williams_r;
  document.getElementById('indWilliams').textContent = wr ? wr.toFixed(1) : '--';
  document.getElementById('indWilliams').className = 'ind-value ' + (wr !== null ? (wr > -20 ? 'red' : wr < -80 ? 'green' : '') : '');
  document.getElementById('indWilliamsHint').textContent = wr > -20 ? 'Overbought' : wr < -80 ? 'Oversold' : 'Neutral';

  // CCI
  const cci = d.cci;
  document.getElementById('indCci').textContent = cci ? cci.toFixed(0) : '--';
  document.getElementById('indCci').className = 'ind-value ' + (cci > 100 ? 'green' : cci < -100 ? 'red' : '');
  document.getElementById('indCciHint').textContent = cci > 100 ? 'Bullish momentum' : cci < -100 ? 'Bearish momentum' : 'Neutral zone';

  // ADX
  const adx = d.adx;
  document.getElementById('indAdx').textContent = adx ? adx.toFixed(1) : '--';
  document.getElementById('indAdx').className = 'ind-value ' + (adx > 25 ? 'cyan' : '');
  setBar('indAdxBar', adx, 0, 60, adx > 25 ? '#06b6d4' : '#6366f1');
  document.getElementById('indAdxHint').textContent = adx > 50 ? 'Very strong trend' : adx > 25 ? 'Trending' : 'Weak/no trend';

  // MFI
  const mfi = d.mfi;
  document.getElementById('indMfi').textContent = mfi ? mfi.toFixed(1) : '--';
  document.getElementById('indMfi').className = 'ind-value ' + colorClass(mfi, 80, 20);
  setBar('indMfiBar', mfi, 0, 100, mfi > 80 ? '#ef4444' : mfi < 20 ? '#22c55e' : '#6366f1');
  document.getElementById('indMfiHint').textContent = mfi > 80 ? 'Overbought (money flow)' : mfi < 20 ? 'Oversold (money flow)' : 'Neutral money flow';

  // ROC
  const roc = d.roc;
  document.getElementById('indRoc').textContent = roc ? roc.toFixed(2) + '%' : '--';
  document.getElementById('indRoc').className = 'ind-value ' + (roc > 0 ? 'green' : roc < 0 ? 'red' : '');
  document.getElementById('indRocHint').textContent = roc > 5 ? 'Strong upward momentum' : roc < -5 ? 'Strong downward momentum' : 'Moderate momentum';

  // BB Position
  const bb = d.bb_position;
  document.getElementById('indBB').textContent = bb !== null ? bb.toFixed(1) + '%' : '--';
  document.getElementById('indBB').className = 'ind-value ' + colorClass(bb, 90, 10);
  setBar('indBBBar', bb, 0, 100, bb > 90 ? '#ef4444' : bb < 10 ? '#22c55e' : '#6366f1');
  document.getElementById('indBBHint').textContent = bb > 90 ? 'Near upper band' : bb < 10 ? 'Near lower band' : 'Mid-band';

  // Volume ratio
  const vr = d.volume_ratio;
  document.getElementById('indVol').textContent = vr ? vr.toFixed(2) + 'x' : '--';
  document.getElementById('indVol').className = 'ind-value ' + (vr > 1.5 ? 'cyan' : '');
  document.getElementById('indVolHint').textContent = vr > 1.5 ? 'Strong volume confirmation' : vr < 0.7 ? 'Low volume - weak signal' : 'Average volume';

  // ATR
  const atr = d.atr;
  document.getElementById('indAtr').textContent = atr ? '$' + atr.toFixed(2) : '--';
  document.getElementById('indAtrHint').textContent = 'Average true range (volatility)';

  // ── NEW INDICATORS ──

  // Hurst Exponent
  const hurst = d.hurst;
  document.getElementById('indHurst').textContent = hurst !== null ? hurst.toFixed(3) : '--';
  document.getElementById('indHurst').className = 'ind-value ' + (hurst > 0.55 ? 'green' : hurst < 0.45 ? 'cyan' : 'yellow');
  setBar('indHurstBar', hurst, 0, 1, hurst > 0.55 ? '#22c55e' : hurst < 0.45 ? '#06b6d4' : '#f59e0b');
  document.getElementById('indHurstHint').textContent =
    hurst > 0.6 ? 'Trending regime - momentum works' :
    hurst < 0.4 ? 'Mean-reverting - contrarian works' :
    hurst < 0.45 ? 'Slight mean-reversion tendency' :
    hurst > 0.55 ? 'Slight trending tendency' :
    'Random walk - low predictability';

  // Z-Score
  const zs = d.zscore;
  document.getElementById('indZscore').textContent = zs !== null ? zs.toFixed(2) : '--';
  document.getElementById('indZscore').className = 'ind-value ' + (zs > 2 ? 'red' : zs < -2 ? 'green' : '');
  document.getElementById('indZscoreHint').textContent =
    zs > 2 ? 'Extremely overbought - expect pullback' :
    zs < -2 ? 'Extremely oversold - expect bounce' :
    zs > 1 ? 'Above mean' : zs < -1 ? 'Below mean' : 'Near mean';

  // Order Flow Imbalance
  const ofi = d.ofi;
  document.getElementById('indOfi').textContent = ofi !== null ? ofi.toFixed(3) : '--';
  document.getElementById('indOfi').className = 'ind-value ' + (ofi > 0.2 ? 'green' : ofi < -0.2 ? 'red' : '');
  document.getElementById('indOfiHint').textContent =
    ofi > 0.3 ? 'Strong buying pressure' :
    ofi < -0.3 ? 'Strong selling pressure' :
    'Balanced order flow';

  // MTF Convergence
  const mtf = d.mtf_convergence;
  document.getElementById('indMtf').textContent = mtf !== null ? mtf.toFixed(2) : '--';
  document.getElementById('indMtf').className = 'ind-value ' + (mtf > 0.5 ? 'green' : mtf < -0.5 ? 'red' : 'yellow');
  setBar('indMtfBar', (mtf + 1) / 2, 0, 1, mtf > 0.5 ? '#22c55e' : mtf < -0.5 ? '#ef4444' : '#f59e0b');
  document.getElementById('indMtfHint').textContent =
    mtf > 0.75 ? 'All timeframes bullish' :
    mtf < -0.75 ? 'All timeframes bearish' :
    'Mixed signals across timeframes';

  // Autocorrelation
  const ac = d.autocorr;
  document.getElementById('indAutocorr').textContent = ac !== null ? ac.toFixed(3) : '--';
  document.getElementById('indAutocorr').className = 'ind-value ' + (ac > 0.1 ? 'green' : ac < -0.1 ? 'cyan' : '');
  document.getElementById('indAutocorrHint').textContent =
    ac > 0.15 ? 'Positive serial corr - momentum persists' :
    ac < -0.15 ? 'Negative serial corr - reversals likely' :
    'Near zero - no serial dependence';

  // Volatility Regime
  const vlr = d.vol_regime;
  document.getElementById('indVolRegime').textContent = vlr !== null ? (vlr * 100).toFixed(0) + 'th %ile' : '--';
  setBar('indVolRegimeBar', vlr, 0, 1, vlr > 0.8 ? '#ef4444' : vlr < 0.2 ? '#22c55e' : '#6366f1');
  document.getElementById('indVolRegimeHint').textContent =
    vlr > 0.8 ? 'High volatility regime - larger moves' :
    vlr < 0.2 ? 'Low volatility regime - calm market' :
    'Normal volatility';
}

function colorClass(val, highThresh, lowThresh) {
  if (val === null || val === undefined) return '';
  if (val > highThresh) return 'red';
  if (val < lowThresh) return 'green';
  return '';
}

function setBar(id, val, min, max, color) {
  const el = document.getElementById(id);
  if (!el || val === null || val === undefined) return;
  const pct = Math.min(100, Math.max(0, ((val - min) / (max - min)) * 100));
  el.style.width = pct + '%';
  el.style.background = color;
}
