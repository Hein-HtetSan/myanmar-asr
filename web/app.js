'use strict';
const byId = id => document.getElementById(id);
let runs = [];
const number = value => typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString(undefined, {maximumFractionDigits: 4}) : 'Not logged';
function textElement(tag, text) { const element = document.createElement(tag); element.textContent = text; return element; }
function fillRows(target, rows) {
  target.replaceChildren(...rows.map(values => { const row = document.createElement('tr'); values.forEach(value => row.append(textElement('td', value))); return row; }));
}
function drawHistory(run) {
  const metric = byId('metric-select').value;
  const points = (run.metric_history[metric] || []).filter(point => Number.isFinite(point.value));
  fillRows(byId('history'), points.map(point => [point.step, number(point.value), new Date(point.timestamp).toISOString()]));
  const chart = byId('chart'); chart.replaceChildren();
  if (!points.length) { chart.append(textElement('p', 'No finite metric points logged.')); return; }
  const xs = points.map(point => point.step), ys = points.map(point => point.value);
  const extent = values => values.reduce(([low,high],value) => [Math.min(low,value),Math.max(high,value)], [Infinity,-Infinity]);
  const [minX,maxX] = extent(xs), [minY,maxY] = extent(ys);
  const ns = 'http://www.w3.org/2000/svg'; const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 920 280'); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', `${metric}: ${points.length} logged points, values ${number(minY)} to ${number(maxY)}`);
  const line = document.createElementNS(ns, 'polyline');
  line.setAttribute('points', points.map(point => `${60 + (point.step-minX)/(maxX-minX || 1)*830},${230-(point.value-minY)/(maxY-minY || 1)*200}`).join(' '));
  line.setAttribute('fill', 'none'); line.setAttribute('stroke', '#155e50'); line.setAttribute('stroke-width', '2'); svg.append(line);
  const labels = [[5,30,number(maxY)],[5,230,number(minY)],[60,265,`Step ${minX}`],[760,265,`Step ${maxX}`]];
  for (const [x,y,value] of labels) { const label=document.createElementNS(ns,'text'); label.setAttribute('x',x); label.setAttribute('y',y); label.setAttribute('font-size','12'); label.setAttribute('fill','#586b67'); label.textContent=value; svg.append(label); }
  chart.append(svg);
}
function renderRun() {
  const run = runs[Number(byId('run-select').value)];
  byId('run').hidden = false; byId('run-name').textContent = run.name; byId('run-status').textContent = run.status;
  byId('run-id').textContent = run.run_id;
  byId('metrics').replaceChildren(...Object.entries(run.latest_metrics).map(([key,value]) => {
    const box=textElement('div',''); box.className='metric'; box.append(textElement('small',key),textElement('strong',number(value))); return box;
  }));
  byId('config').textContent = run.configuration ? JSON.stringify(run.configuration,null,2) : 'This run has no configuration/run.json artifact. No configuration was inferred.';
  byId('result-artifacts').textContent = JSON.stringify(run.result_artifacts || {},null,2);
  byId('artifact-heading').textContent = `Artifact inventory · ${run.artifacts.length} files`;
  fillRows(byId('artifacts'),run.artifacts.map(item=>[item.path,item.size_bytes ?? 'Unknown']));
  byId('metric-select').replaceChildren(...Object.keys(run.metric_history).map(key => { const option=textElement('option',key); option.value=key; return option; }));
  drawHistory(run);
}
byId('run-select').addEventListener('change',renderRun);
byId('metric-select').addEventListener('change',()=>drawHistory(runs[Number(byId('run-select').value)]));
(async()=>{
  try {
    const response=await fetch('results.json'); if(!response.ok) throw new Error('Results request failed');
    const data=await response.json(); if(data.schema_version!==1 || !Array.isArray(data.runs)) throw new Error('Unsupported results schema');
    runs=data.runs;
    byId('updated').textContent=data.generated_at ? `Snapshot: ${new Date(data.generated_at).toISOString()} · ${runs.length} selected runs` : 'Awaiting the first reviewed MLflow export';
    if(!runs.length) { byId('message').textContent='No experiment results have been published yet. The automatic workflow publishes completed runs after their result artifacts are available. No placeholder accuracy scores are shown.'; return; }
    byId('message').hidden=true;
    byId('run-select').replaceChildren(...runs.map((run,index)=>{const option=textElement('option',run.name);option.value=index;return option;}));
    byId('run-select').disabled=false; renderRun();
  } catch(error) { byId('message').hidden=false; byId('message').textContent='The results snapshot could not be loaded. Please retry or download the JSON to inspect it.'; }
})();
