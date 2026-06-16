/* static/js/dashboard.js */

const GREEN = '#1D9E75';
const RED   = '#E24B4A';
const BLUE  = '#3266ad';

// ── Slider display ────────────────────────────────────────────────────────────
const formatters = {
  income:         v => '₹' + Number(v).toLocaleString('en-IN'),
  debt:           v => '₹' + Number(v).toLocaleString('en-IN'),
  credit_score:   v => v,
  on_time_pct:    v => v + '%',
  utilization:    v => v + '%',
  history_years:  v => v + ' yr',
  delinquencies:  v => v,
};

document.querySelectorAll('input[type=range]').forEach(el => {
  const outId = el.id + 'V';
  const fmt   = formatters[el.id] || (v => v);
  el.addEventListener('input', () => {
    document.getElementById(outId).textContent = fmt(el.value);
  });
});

// ── Predict ───────────────────────────────────────────────────────────────────
async function runPredict() {
  const btn = document.getElementById('predict-btn');
  btn.textContent = 'Predicting…';
  btn.disabled = true;

  const payload = {
    income:        document.getElementById('income').value,
    debt:          document.getElementById('debt').value,
    credit_score:  document.getElementById('credit_score').value,
    on_time_pct:   document.getElementById('on_time_pct').value,
    utilization:   document.getElementById('utilization').value,
    history_years: document.getElementById('history_years').value,
    delinquencies: document.getElementById('delinquencies').value,
    model:         document.getElementById('model-select').value,
  };

  try {
    const res  = await fetch('/api/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();
    renderResult(data, payload);
  } catch (err) {
    document.getElementById('result-area').innerHTML =
      `<p style="color:${RED}">Error: ${err.message}. Make sure the Flask server is running and models are trained.</p>`;
  } finally {
    btn.textContent = 'Predict';
    btn.disabled = false;
  }
}

function renderResult(data, payload) {
  const worthy = data.label === 1;
  const color  = worthy ? GREEN : RED;
  const prob   = data.probability;

  const factors = [
    { label: 'Credit score',       val: payload.credit_score,  threshold: 700,  higherGood: true  },
    { label: 'On-time payments',   val: payload.on_time_pct,   threshold: 80,   higherGood: true  },
    { label: 'Credit utilization', val: payload.utilization,   threshold: 30,   higherGood: false },
    { label: 'Delinquencies',      val: payload.delinquencies, threshold: 0,    higherGood: false },
    { label: 'History (years)',    val: payload.history_years, threshold: 5,    higherGood: true  },
  ];

  const factorHtml = factors.map(f => {
    const v = parseFloat(f.val);
    const positive = f.higherGood ? v >= f.threshold : v <= f.threshold;
    return `
      <div class="factor-row">
        <span class="dot" style="background:${positive ? GREEN : RED}"></span>
        ${f.label}: <strong>${formatters[Object.keys(formatters).find(k => f.label.toLowerCase().includes(k.replace(/_/g,' ').toLowerCase()))] 
          ? formatters[Object.keys(formatters).find(k => f.label.toLowerCase().includes(k.replace(/_/g,' ').toLowerCase()))](v) 
          : v}</strong>
      </div>`;
  }).join('');

  document.getElementById('result-area').innerHTML = `
    <div class="verdict-block">
      <div class="verdict-label">Prediction — ${document.getElementById('model-select').options[document.getElementById('model-select').selectedIndex].text}</div>
      <div class="verdict-name" style="color:${color}">${data.verdict}</div>
      <div class="verdict-prob" style="color:${color}">${prob}%</div>
      <div class="verdict-sub">creditworthiness probability</div>
    </div>
    <div class="prob-bar-wrap">
      <div class="prob-bar" style="width:${prob}%; background:${color};"></div>
    </div>
    <div class="factors">
      <div class="factors-title">Key factors</div>
      ${factorHtml}
    </div>
  `;
}

// ── Model metrics table ───────────────────────────────────────────────────────
async function loadMetrics() {
  try {
    const res  = await fetch('/api/models');
    const data = await res.json();
    const rows = Object.entries(data).map(([name, m], i) => `
      <tr ${i === 0 ? 'class="best"' : ''}>
        <td>${name.replace(/_/g, ' ')} ${i === 0 ? '<span class="chip">best</span>' : ''}</td>
        <td>${m.accuracy.toFixed(1)}%</td>
        <td>${m.precision.toFixed(1)}%</td>
        <td>${m.recall.toFixed(1)}%</td>
        <td>${m.f1.toFixed(1)}%</td>
        <td>${m.roc_auc.toFixed(3)}</td>
      </tr>`).join('');
    document.getElementById('metrics-body').innerHTML = rows;
  } catch (err) {
    document.getElementById('metrics-body').innerHTML =
      `<tr><td colspan="6" style="text-align:center;color:${RED}">Could not load metrics — is the server running?</td></tr>`;
  }
}

loadMetrics();
