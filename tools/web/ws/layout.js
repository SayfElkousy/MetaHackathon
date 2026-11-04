// Tabbed split-pane layout. A view (MPR plane, scaffold viewer, QC panel, …) is a persistent DOM element that is
// re-parented when tabs move, so canvases, WebGL contexts and scroll positions survive layout changes.
// Tree: { t: 'split', dir: 'row' | 'col', sizes: [fractions], kids: [...] } | { t: 'pane', id, tabs: [viewId], active }

let paneSeq = 0;
const newPane = (tabs) => ({ t: 'pane', id: 'p' + (++paneSeq) + Math.random().toString(36).slice(2, 5), tabs, active: tabs[0] || null });

export class Layout {
  constructor(host, opts) {
    this.host = host; this.o = opts; this.tree = newPane([]); this.max = null; this.activeView = null;
    this.holder = document.createElement('div'); this.holder.style.display = 'none'; document.body.appendChild(this.holder);
  }
  // ---------------------------------------------------------------- queries
  panes(n = this.tree, out = []) { if (n.t === 'pane') out.push(n); else n.kids.forEach(k => this.panes(k, out)); return out; }
  paneOf(viewId) { return this.panes().find(p => p.tabs.includes(viewId)) || null; }
  pane(id) { return this.panes().find(p => p.id === id) || null; }
  views() { return this.panes().flatMap(p => p.tabs); }
  visibleViews() { return (this.max ? [this.pane(this.max)].filter(Boolean) : this.panes()).map(p => p.active).filter(Boolean); }
  snapshot() { return JSON.parse(JSON.stringify({ tree: this.tree, max: this.max, active: this.activeView })); }
  restoreSnapshot(s) { this.tree = s.tree; this.max = s.max; this.activeView = s.active; this.render(); }
  // ---------------------------------------------------------------- mutations
  setTree(tree) { this.tree = tree; this.max = null; this.normalize(); this.render(); }
  static grid(rows) {
    // rows: [[viewId, …], …]  -> row of panes per line, lines stacked vertically
    const line = r => r.length === 1 ? newPane([r[0]].flat()) : { t: 'split', dir: 'row', sizes: r.map(() => 1 / r.length), kids: r.map(v => newPane([v].flat())) };
    return rows.length === 1 ? line(rows[0]) : { t: 'split', dir: 'col', sizes: rows.map(() => 1 / rows.length), kids: rows.map(line) };
  }
  open(viewId, where) {
    const have = this.paneOf(viewId);
    if (have) { have.active = viewId; this.activate(viewId); this.render(); return; }
    where = where || 'active';
    const act = this.paneOf(this.activeView) || this.panes()[0];
    if (where === 'active' && act) { act.tabs.push(viewId); act.active = viewId; }
    else if (where === 'right' || where === 'down' || where === 'left' || where === 'up') this.splitAt(act || this.panes()[0], viewId, where);
    else { const p = this.pane(where); if (p) { p.tabs.push(viewId); p.active = viewId; } else this.panes()[0].tabs.push(viewId); }
    this.activeView = viewId; this.normalize(); this.render(); this.o.onActivate && this.o.onActivate(viewId);
  }
  splitAt(pane, viewId, edge) {
    const dir = edge === 'left' || edge === 'right' ? 'row' : 'col';
    const fresh = newPane([viewId]);
    const replace = (n, parent, idx) => {
      if (n === pane) {
        const before = edge === 'left' || edge === 'up';
        const node = { t: 'split', dir, sizes: [0.5, 0.5], kids: before ? [fresh, pane] : [pane, fresh] };
        if (!parent) this.tree = node; else parent.kids[idx] = node;
        return true;
      }
      return n.t === 'split' && n.kids.some((k, i) => replace(k, n, i));
    };
    replace(this.tree, null, 0);
  }
  moveView(viewId, targetPaneId, edge) {
    const from = this.paneOf(viewId), to = this.pane(targetPaneId);
    if (!from || !to) return;
    if (from === to && (!edge || edge === 'center')) return;
    from.tabs = from.tabs.filter(v => v !== viewId);
    if (from.active === viewId) from.active = from.tabs[from.tabs.length - 1] || null;
    if (!edge || edge === 'center') { to.tabs.push(viewId); to.active = viewId; }
    else this.splitAt(to, viewId, edge);
    this.activeView = viewId; this.normalize(); this.render(); this.o.onActivate && this.o.onActivate(viewId);
  }
  close(viewId) {
    const p = this.paneOf(viewId);
    if (!p) return;
    p.tabs = p.tabs.filter(v => v !== viewId);
    if (p.active === viewId) p.active = p.tabs[p.tabs.length - 1] || null;
    if (this.max === p.id && !p.tabs.length) this.max = null;
    this.normalize(); this.render();
    this.o.onClose && this.o.onClose(viewId);
    if (this.activeView === viewId) { const v = this.visibleViews()[0]; if (v) this.activate(v); }
  }
  activate(viewId) {
    if (!viewId) return;
    this.activeView = viewId;
    const p = this.paneOf(viewId);
    if (p && p.active !== viewId) { p.active = viewId; this.render(); }
    this.host.querySelectorAll('.wpane').forEach(el => el.classList.toggle('active', el.dataset.pane === (p && p.id)));
    this.o.onActivate && this.o.onActivate(viewId);
  }
  maximize(viewId) {
    const p = this.paneOf(viewId || this.activeView); if (!p) return;
    this.max = p.id; this.render(); this.o.onChange && this.o.onChange();
  }
  restore() { this.max = null; this.render(); this.o.onChange && this.o.onChange(); }
  normalize() {
    const fix = (n) => {
      if (n.t === 'pane') return n.tabs.length ? n : null;
      const kids = [], sizes = [];
      n.kids.forEach((k, i) => { const f = fix(k); if (f) { kids.push(f); sizes.push(n.sizes[i] || 1 / n.kids.length); } });
      if (!kids.length) return null;
      if (kids.length === 1) return kids[0];
      const tot = sizes.reduce((a, b) => a + b, 0);
      n.kids = kids; n.sizes = sizes.map(s => s / tot);
      return n;
    };
    this.tree = fix(this.tree) || newPane([]);
  }
  // ---------------------------------------------------------------- render
  render() {
    // park every view element, then rebuild the frame and re-attach the active ones
    for (const v of this.views()) { const el = this.o.viewEl(v); if (el && el.parentNode !== this.holder) this.holder.appendChild(el); }
    this.host.innerHTML = '';
    const node = this.max ? this.pane(this.max) : this.tree;
    if (!node) { this.max = null; return this.render(); }
    this.host.appendChild(this.build(node));
    if (this.max) {
      const b = document.createElement('button'); b.className = 'restoreBtn'; b.textContent = 'Restore layout';
      b.title = 'Return to the previous pane arrangement (Esc)'; b.onclick = () => this.restore(); this.host.appendChild(b);
    }
    const ap = this.paneOf(this.activeView);
    this.host.querySelectorAll('.wpane').forEach(el => el.classList.toggle('active', el.dataset.pane === (ap && ap.id)));
    this.o.onChange && this.o.onChange();
    this.o.afterRender && this.o.afterRender();
  }
  build(n) {
    if (n.t === 'pane') return this.buildPane(n);
    const el = document.createElement('div'); el.className = 'wsplit dir-' + n.dir;
    n.kids.forEach((k, i) => {
      const c = this.build(k); c.style.flex = `${n.sizes[i]} 1 0`; el.appendChild(c);
      if (i < n.kids.length - 1) {
        const g = document.createElement('div'); g.className = 'wgutter'; g.title = 'Drag to resize';
        g.onpointerdown = (e) => this.dragGutter(e, el, n, i);
        el.appendChild(g);
      }
    });
    return el;
  }
  dragGutter(e, el, n, i) {
    e.preventDefault();
    const rect = el.getBoundingClientRect(), total = n.dir === 'row' ? rect.width : rect.height;
    const start = n.dir === 'row' ? e.clientX : e.clientY, a0 = n.sizes[i], b0 = n.sizes[i + 1];
    const kids = [...el.children].filter(c => !c.classList.contains('wgutter'));
    const move = (ev) => {
      const d = ((n.dir === 'row' ? ev.clientX : ev.clientY) - start) / total;
      const a = Math.max(0.08, Math.min(a0 + b0 - 0.08, a0 + d));
      n.sizes[i] = a; n.sizes[i + 1] = a0 + b0 - a;
      kids[i].style.flex = `${n.sizes[i]} 1 0`; kids[i + 1].style.flex = `${n.sizes[i + 1]} 1 0`;
    };
    const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); this.o.onChange && this.o.onChange(); };
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
  }
  buildPane(p) {
    const el = document.createElement('section'); el.className = 'wpane'; el.dataset.pane = p.id;
    const bar = document.createElement('div'); bar.className = 'wtabs'; bar.setAttribute('role', 'tablist');
    for (const v of p.tabs) {
      const t = document.createElement('div'); t.className = 'wtab' + (v === p.active ? ' on' : ''); t.setAttribute('role', 'tab');
      t.draggable = true; t.dataset.view = v; t.title = (this.o.viewTip && this.o.viewTip(v)) || '';
      const lab = document.createElement('span'); lab.className = 'wtabl'; lab.textContent = this.o.viewTitle(v);
      const x = document.createElement('button'); x.className = 'wtabx'; x.textContent = '×'; x.title = 'Close this view (the data stays loaded)';
      x.onclick = (e) => { e.stopPropagation(); this.close(v); };
      t.append(lab, x);
      t.onclick = () => { p.active = v; this.activate(v); this.render(); };
      t.ondragstart = (e) => { e.dataTransfer.setData('text/x-view', v); e.dataTransfer.effectAllowed = 'move'; document.body.classList.add('dragging-tab'); };
      t.ondragend = () => document.body.classList.remove('dragging-tab');
      t.oncontextmenu = (e) => { e.preventDefault(); this.o.tabMenu && this.o.tabMenu(v, e); };
      bar.appendChild(t);
    }
    const sp = document.createElement('span'); sp.className = 'grow'; bar.appendChild(sp);
    const mk = (txt, tip, fn) => { const b = document.createElement('button'); b.className = 'wpbtn'; b.textContent = txt; b.title = tip; b.onclick = fn; return b; };
    if (this.o.addMenu) bar.appendChild(mk('+', 'Open a view in this pane', (e) => this.o.addMenu(p.id, e)));
    bar.appendChild(mk('⇥', 'Split: open the active view in a new pane to the right', () => p.active && this.moveView(p.active, p.id, 'right')));
    bar.appendChild(this.max === p.id ? mk('⤡', 'Restore the previous layout', () => this.restore()) : mk('⤢', 'Maximize this pane', () => this.maximize(p.active)));
    const body = document.createElement('div'); body.className = 'wbody';
    const ve = p.active && this.o.viewEl(p.active);
    if (ve) body.appendChild(ve);
    const drop = document.createElement('div'); drop.className = 'wdrop';
    body.appendChild(drop);
    const zone = (e) => {
      const r = body.getBoundingClientRect(), x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
      if (x < 0.22) return 'left'; if (x > 0.78) return 'right'; if (y < 0.22) return 'up'; if (y > 0.78) return 'down'; return 'center';
    };
    const over = (e) => { if (![...e.dataTransfer.types].includes('text/x-view')) return; e.preventDefault(); drop.dataset.zone = zone(e); drop.classList.add('on'); };
    body.ondragover = over; bar.ondragover = (e) => { if ([...e.dataTransfer.types].includes('text/x-view')) { e.preventDefault(); drop.dataset.zone = 'center'; drop.classList.add('on'); } };
    body.ondragleave = bar.ondragleave = () => drop.classList.remove('on');
    const dropIt = (e, z) => { e.preventDefault(); drop.classList.remove('on'); const v = e.dataTransfer.getData('text/x-view'); if (v) this.moveView(v, p.id, z); };
    body.ondrop = (e) => dropIt(e, zone(e)); bar.ondrop = (e) => dropIt(e, 'center');
    el.onpointerdown = () => { if (p.active && this.activeView !== p.active) this.activate(p.active); };
    el.append(bar, body);
    return el;
  }
}
