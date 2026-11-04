// MRI upload, inside the workspace (the one page): scan -> KneeScan reader (MONAI tear ensemble, MedSAM ACL outline)
// -> readings -> scaffold builder. Server side: /api/mri/upload and /api/mri/job (tools/mri_bridge.py).
import { api } from './imaging.js';
import { esc } from './panels.js';

const STEP_LABELS = { upload: 'Send the scan to the MRI reader', analyze: 'MRI reader analyses the scan', results: 'Fetch the readings',
  convert: 'Convert readings to modeler parameters', save: 'Save the design for the scaffold builder' };
const fmt = (v, d = 1) => v == null ? '—' : (+v).toFixed(d);

export class MriUpload {
  constructor(ws) {
    this.ws = ws; this.file = null; this.side = null; this.busy = false; this.job = null; this.t0 = 0;
    this.el = document.createElement('dialog'); this.el.id = 'mriDlg';
    this.el.innerHTML = `
      <div class="dlg">
        <div class="mhead"><h3>Upload a knee MRI</h3><button type="button" data-x="close" title="Close">×</button></div>
        <p class="note">The scan goes to the MRI reader (KneeScan): it finds the ACL, checks it for a tear (MONAI ensemble) and outlines and
          measures it (MedSAM). Its readings go to the scaffold builder, which builds the model from them. <i>Research prototype, not for clinical use.</i></p>
        <p class="note" data-x="replaces" hidden>This builds a new design from the scan; the current design stays saved under its link.</p>
        <section class="mgrid">
          <div>
            <h4>MRI reader <span class="pill" data-x="pill">checking…</span></h4>
            <div class="mrow"><input data-x="url" spellcheck="false" aria-label="MRI reader address"><button type="button" data-x="check">Check</button></div>
            <p class="note" data-x="readerMsg"></p>
            <p class="note" data-x="setup" hidden>Set it up once: in a terminal in the project folder run <code>python tools/setup_kneescan.py</code>
              (it asks for a Hugging Face token for the model weights), then restart <code>python tools/server.py</code>.</p>
            <p class="note">The workspace server starts KneeScan on this computer by itself. To use another one instead
              (e.g. a Colab <code>…gradio.live</code> link), paste its address and press Check.</p>
          </div>
          <div>
            <h4>Scan</h4>
            <div class="mdrop" data-x="drop" tabindex="0" role="button" aria-label="Choose an MRI file"><b data-x="fileName">Choose or drop an MRI file</b>
              <span class="note">.zip of a DICOM series, .dcm, .nii / .nii.gz, KneeMRI .pck or MRNet .npy</span></div>
            <input data-x="file" type="file" accept=".zip,.dcm,.nii,.gz,.npy,.pck" hidden>
            <div class="mrow"><span>Knee side</span><span role="group" aria-label="Knee side"><button type="button" class="seg" data-side="right">Right</button><button type="button" class="seg" data-side="left">Left</button></span>
              <span class="note">the reader cannot tell left from right</span></div>
            <label class="mchk"><input data-x="bones" type="checkbox" checked> Measure femur → fibular head (experimental, adds about a minute;
              files without pixel spacing, .npy and .pck, get their millimetres from this step)</label>
          </div>
        </section>
        <div class="mrow"><button type="button" class="primary" data-x="go" disabled>Analyze and send to the scaffold builder</button><span class="note" data-x="hint"></span></div>
        <section data-x="prog" hidden>
          <h4>Progress <span class="mono muted" data-x="elapsed"></span></h4>
          <div data-x="upWrap"><div class="note" data-x="upTxt">uploading…</div><div class="mbar"><div data-x="bar"></div></div></div>
          <ul class="msteps" data-x="steps"></ul>
          <p class="note mono" data-x="stepMsg"></p>
          <div class="merr" data-x="err" hidden></div>
        </section>
        <section data-x="res" hidden>
          <h4>Readings from the MRI reader <span class="mono muted" data-x="resMeta"></span></h4>
          <div class="mcards" data-x="cards"></div>
          <ul class="mwarn" data-x="warn"></ul>
          <h4>Sent to the scaffold builder</h4>
          <div class="mlen" data-x="lenWarn" hidden></div>
          <div class="mtable"><table class="t" data-x="map"></table></div>
          <div class="mrow mend">
            <button type="button" data-x="dlReadings">Download KneeScan results</button>
            <button type="button" data-x="dlDoc">Download acl-mri-input JSON</button>
            <button type="button" class="primary" data-x="build">Build the scaffold model</button>
          </div>
        </section>
      </div>`;
    document.body.appendChild(this.el);
    this.x = k => this.el.querySelector(`[data-x="${k}"]`);
    this.bind();
  }

  bind() {
    const x = this.x;
    x('close').onclick = () => this.el.close();
    x('check').onclick = () => this.checkReader();
    x('url').onkeydown = e => { if (e.key === 'Enter') this.checkReader(); };
    x('drop').onclick = () => x('file').click();
    x('drop').onkeydown = e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); x('file').click(); } };
    x('file').onchange = e => this.pick(e.target.files[0]);
    ['dragover', 'dragenter'].forEach(ev => x('drop').addEventListener(ev, e => { e.preventDefault(); x('drop').classList.add('over'); }));
    ['dragleave', 'drop'].forEach(ev => x('drop').addEventListener(ev, e => { e.preventDefault(); x('drop').classList.remove('over'); }));
    x('drop').addEventListener('drop', e => this.pick(e.dataTransfer.files[0]));
    this.el.querySelectorAll('button[data-side]').forEach(b => b.onclick = () => {
      this.side = b.dataset.side; this.el.querySelectorAll('button[data-side]').forEach(o => o.classList.toggle('on', o === b)); this.refresh();
    });
    x('go').onclick = () => this.start();
    x('build').onclick = () => this.buildModel();
    x('dlReadings').onclick = () => this.job && this.download('kneescan_' + this.job.id + '.json', this.job.result.kneescan);
    x('dlDoc').onclick = () => this.job && this.download('acl-mri-input_' + this.job.id + '.json', this.job.result.document);
  }

  open() {
    this.x('replaces').hidden = !this.ws.app.S.state;
    this.x('url').value = this.ws.readerUrl || '';
    if (!this.el.open) this.el.showModal();
    this.checkReader();
  }

  async checkReader() {
    const url = this.x('url').value.trim();
    if (url) this.ws.readerUrl = url;
    this.x('pill').className = 'pill busy'; this.x('pill').textContent = 'checking…';
    await this.ws.checkReader();                     // one reader address for the whole workspace (AI segmentation uses it too)
    if (!url && this.ws.readerUrl) this.x('url').value = this.ws.readerUrl;
  }

  // Reader status, including what the server's own KneeScan launcher is doing (tools/reader_launcher.py).
  renderReader() {
    const r = this.ws.reader || {}, L = r.launcher || {}, st = L.state;
    const [cls, txt] = r.ok ? ['ok', 'connected'] : st === 'starting' ? ['warn', 'starting…'] : st === 'not_installed' ? ['fail', 'not set up']
      : st === 'exited' ? ['fail', 'stopped'] : ['fail', 'not reachable'];
    this.x('pill').className = 'pill ' + cls; this.x('pill').textContent = txt;
    let msg = r.ok ? 'KneeScan is answering at ' + r.url : (r.error || '');
    if (st === 'not_installed') msg = 'KneeScan is not set up on this computer yet: ' +
      ((L.missing || []).map(m => m.split(' (')[0]).join(' and ') || 'parts are') + ' missing.';
    if (st === 'exited' && (L.log_tail || []).length) msg += ' Last lines: ' + L.log_tail.join(' | ');
    this.x('readerMsg').textContent = msg;
    this.x('setup').hidden = !(st === 'not_installed' || st === 'exited');
    this.refresh();
  }

  pick(f) {
    if (!f) return;
    this.file = f; this.x('fileName').textContent = f.name + '  ·  ' + (f.size / 1048576).toFixed(1) + ' MB'; this.refresh();
  }

  refresh() {
    const ok = this.ws.reader && this.ws.reader.ok;
    const why = !this.file ? 'choose a scan' : !this.side ? 'choose the knee side' : !ok ? 'the MRI reader is not reachable' : '';
    this.x('go').disabled = this.busy || !!why; this.x('hint').textContent = this.busy ? '' : why;
  }

  steps(list) { this.x('steps').innerHTML = list.map(s => `<li class="${s.state}"><span class="dot"></span>${esc(STEP_LABELS[s.id] || s.label)}</li>`).join(''); }

  start() {
    const x = this.x;
    this.busy = true; this.job = null; this.refresh();
    x('res').hidden = true; x('prog').hidden = false; x('err').hidden = true; x('upWrap').hidden = false; x('stepMsg').textContent = '';
    this.steps([{ label: 'Upload the scan to the workspace server', state: 'running' }]);
    this.t0 = Date.now(); clearInterval(this.tick);
    this.tick = setInterval(() => { x('elapsed').textContent = Math.round((Date.now() - this.t0) / 1000) + ' s'; }, 500);
    const q = new URLSearchParams({ name: this.file.name, side: this.side, bones: x('bones').checked ? '1' : '0', reader: this.ws.readerUrl || '' });
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/mri/upload?' + q.toString());
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    xhr.upload.onprogress = e => { if (e.lengthComputable) { const p = 100 * e.loaded / e.total; x('bar').style.width = p + '%'; x('upTxt').textContent = 'uploading… ' + p.toFixed(0) + '%'; } };
    xhr.onload = () => {
      let j = {}; try { j = JSON.parse(xhr.responseText); } catch (e) { /* ignore */ }
      if (xhr.status !== 200) return this.fail(j.error || 'upload failed (HTTP ' + xhr.status + ')');
      x('upWrap').hidden = true; this.poll(j.job);
    };
    xhr.onerror = () => this.fail('upload failed: the workspace server did not answer');
    xhr.send(this.file);
  }

  fail(msg) {
    this.busy = false; clearInterval(this.tick); this.refresh();
    this.x('err').hidden = false; this.x('err').textContent = '✖ ' + msg;
  }

  async poll(id) {
    let j;
    try { j = await api('/api/mri/job?id=' + encodeURIComponent(id)); } catch (e) { return this.fail(e.message); }
    this.steps(j.steps);
    this.x('stepMsg').textContent = j.status === 'running' ? j.message : '';
    if (j.status === 'running') return setTimeout(() => this.poll(id), 1000);
    if (j.status === 'error') return this.fail(j.error);
    this.busy = false; clearInterval(this.tick); this.refresh();
    this.job = j; this.render(j.result);
  }

  render(r) {
    const x = this.x, rd = r.readings;
    x('resMeta').textContent = (r.input && r.input.file ? r.input.file + ' · ' : '') + (r.seconds != null ? 'reader took ' + r.seconds + ' s' : '');
    const torn = rd.tear_status.value === 'torn';
    const L = rd.acl_length, W = rd.acl_width, loc = rd.acl_location, ff = rd.femur_fibula;
    const check = v => v.passed_checks === false ? ' <span class="pill tiny warn">check outline</span>' : '';
    const sizeSub = v => v.value_mm != null ? `${esc(v.value_px)} px · ${esc(v.mm_source || '')}` : esc(v.error || 'no millimetre value');
    const card = (title, big, sub, cls) => `<div class="mcard"><h5>${esc(title)}</h5><div class="big ${cls || ''}">${big}</div><div class="sub">${sub || ''}</div></div>`;
    x('cards').innerHTML = [
      card('ACL tear', torn ? 'Torn' : (rd.tear_status.value === 'intact' ? 'Intact' : '—'), `tear probability ${fmt(rd.tear_status.probability, 2)} (cut-off ${fmt(rd.tear_status.threshold, 2)})`, torn ? 'torn' : 'ok'),
      card('Tear level', esc(rd.tear_level.value || '—') + (torn ? ' <span class="pill tiny warn">low reliability</span>' : ''), torn ? 'partial vs complete is the weakest reading' : 'no tear predicted'),
      card('ACL length', L.value_mm != null ? fmt(L.value_mm) + ' mm' + check(L) : '—', sizeSub(L)),
      card('ACL width (mid-substance)', W.value_mm != null ? fmt(W.value_mm) + ' mm' + check(W) : '—', sizeSub(W)),
      card('ACL location', loc.slice != null ? `Slice ${loc.slice + 1}${loc.slices ? ' of ' + loc.slices : ''}` : '—', loc.box_px ? `box x ${loc.box_px[0]}–${loc.box_px[2]}, y ${loc.box_px[1]}–${loc.box_px[3]} px · score ${fmt(loc.score, 2)}` : ''),
      card('Femur → fibular head', ff && ff.value_mm != null ? fmt(ff.value_mm) + ' mm <span class="pill tiny warn">experimental</span>' + check(ff) : '—',
        ff ? (ff.value_mm != null ? `${esc(ff.value_px)} px · slice ${ff.slice + 1} · ${esc(ff.mm_source || '')}` : esc(ff.error || '')) : 'not measured (step was off)'),
    ].join('');
    // Reader caveats remain in the saved provenance/QC data, but are not used
    // as publication-derived blockers in the exploratory build workflow.
    x('warn').innerHTML = '';
    const pill = s => `<span class="pill tiny ${s === 'applied' ? 'ok' : s === 'recorded' ? '' : 'warn'}">${esc(s)}</span>`;
    const val = row => {
      if (row.reading === 'ACL location') { const l = row.value || {}; return l.slice != null ? 'slice ' + (l.slice + 1) : '—'; }
      return row.value == null ? '—' : esc(typeof row.value === 'number' ? fmt(row.value) : row.value) + (row.units ? ' ' + row.units : '');
    };
    const sent = row => row.sent == null ? '' : '<br><span class="mono muted">= ' + esc(Array.isArray(row.sent) ? '[' + row.sent.join(', ') + ']' : row.sent) + '</span>';
    x('map').innerHTML = '<tr><th>Reading</th><th>Value</th><th>Scaffold builder parameter</th><th>Result</th><th>Note</th></tr>' + r.mapping.map(row =>
      `<tr><td>${esc(row.reading)}</td><td class="mono">${val(row)}</td><td class="mono">${row.param ? esc(row.param) + sent(row) : '—'}</td><td>${pill(row.result)}</td><td class="muted">${esc(row.note || '')}</td></tr>`).join('');
    x('lenWarn').hidden = true; x('lenWarn').innerHTML = '';
    x('res').hidden = false;
    x('build').focus();
  }

  async buildModel() {
    const sid = this.job && this.job.result.state_id;
    if (!sid) return;
    try {
      const j = await api('/api/v2/state?id=' + encodeURIComponent(sid));
      this.ws.app.loadState(j.state, sid);
      if (this.job.result.imaging_case) {
        await this.ws.loadCases();
        await this.ws.openCase(this.job.result.imaging_case, true);
      }
      if (!this.ws.layout.visibleViews().includes('scaffold')) this.ws.openKind('scaffold', 'right');
      this.el.close();
      this.ws.toast('Scaffold model built from the MRI readings (' + (j.state.case.label || sid) + ')');
    } catch (e) { this.fail('could not open the design: ' + e.message); }
  }

  download(name, obj) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([JSON.stringify(obj, null, 1)], { type: 'application/json' })); a.download = name; a.click();
  }
}
