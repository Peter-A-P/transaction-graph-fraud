/* The page's figures, drawn from data.json, which `consort site` writes from the repository's
   committed results. No number on the page is typed by hand: every figure and every number
   in a finding below is read from that file. No framework and no build step. */
"use strict";

const NS = "http://www.w3.org/2000/svg";
const SETS = {
  "elliptic": "Elliptic",
  "ellipticpp": "Elliptic++",
  "ibm-HI-Small": "IBM HI-Small",
  "ibm-LI-Small": "IBM LI-Small",
};
const SET_NOTE = {
  "elliptic": "real, transactions",
  "ellipticpp": "real, wallets",
  "ibm-HI-Small": "synthetic",
  "ibm-LI-Small": "synthetic",
};
const ARM = {
  "R": "Row only",
  "G": "Graph features",
  "N": "Graph neural network",
  "TabFM-R": "TabFM, row",
  "TabFM-G": "TabFM, graph features",
  "R-retrain": "Row, retrained each step",
  "TabFM-R-swap": "TabFM, examples swapped",
};
const FROZEN = ["R", "G", "N", "TabFM-R", "TabFM-G"];
const PATTERNS = {
  "FAN-IN": "Fan-in", "FAN-OUT": "Fan-out", "CYCLE": "Cycle", "SCATTER-GATHER": "Scatter-gather",
  "GATHER-SCATTER": "Gather-scatter", "BIPARTITE": "Bipartite", "STACK": "Stack",
  "RANDOM": "Random", "NONE": "No named pattern",
};

let D = null;

/* ---------------------------------------------------------------- small helpers */

const $ = id => document.getElementById(id);
const f2 = v => v.toFixed(2);
const f3 = v => v.toFixed(3);
// A transfer's place in a model's ranking of all test transfers: the share scored at or above
// it. The raw scores are not calibrated and differ in range from model to model (the graph
// neural network puts most transfers above 0.99), so the page shows the rank, which is what
// decides the 1% alert queue and reads the same for every model.
const topLabel = share => `top ${+(share * 100).toPrecision(2)}%`;
// The rank bar on a log scale, from the whole test set (empty) to the top 0.01% (full); the
// 1% queue line sits at the middle. Sized by class, in steps of 5%.
const rankWidth = share => Math.round(Math.min(1, Math.max(0, Math.log10(1 / share) / 4)) * 20) * 5;
const sgn = (v, f) => (v >= 0 ? "+" : "-") + f(Math.abs(v));
const pct = v => `${Math.round(v * 100)}%`;
const pts = v => `${sgn(v * 100, x => x.toFixed(1))} points`;
const ci = (t, f) => `95% CI ${f(t[1])} to ${f(t[2])}`;
const clear = t => t[1] > 0;
const hopWord = h => `${h} hop${h === 1 ? "" : "s"}`;
const armKey = a => (a.startsWith("N-") ? "N" : a);

function el(name, attrs, parent) {
  const e = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

/* HTML built as nodes: every string reaches the page as text, never as markup. */
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false || kid === "") continue;
    e.append(typeof kid === "string" || typeof kid === "number" ? String(kid) : kid);
  }
  return e;
}
const B = (...kids) => h("strong", {}, ...kids);
const P = (...kids) => h("p", {}, ...kids);
function put(id, ...nodes) {
  $(id).replaceChildren(...nodes.flat().filter(n => n !== null && n !== undefined && n !== false && n !== ""));
}

function txt(parent, x, y, s, cls, attrs) {
  const t = el("text", {x, y, class: cls || "tick", ...(attrs || {})}, parent);
  t.textContent = s;
  return t;
}

function frame(id, H, label) {
  const box = $(id);
  box.replaceChildren();
  const W = Math.max(300, Math.floor(box.clientWidth));
  const s = el("svg", {viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": label}, box);
  return {s, W, H, narrow: W < 620};
}

const tip = () => $("tip");
/* A chart mark's card: lines() returns its lines, the first shown in bold. The pointer opens it
   where it is; keyboard focus opens it beside the mark, so every mark can be read without a mouse. */
function hover(node, lines, focusable = true) {
  const fillTip = () => {
    const [head, ...rest] = lines();
    tip().replaceChildren(h("b", {}, head), ...rest.map(line => h("span", {class: "tip-line"}, line)));
    tip().classList.add("on");
  };
  const place = (cx, cy) => {
    const t = tip();
    const x = Math.min(cx + 16, window.innerWidth - t.offsetWidth - 10);
    const y = cy + 16 + t.offsetHeight > window.innerHeight ? cy - t.offsetHeight - 12 : cy + 16;
    t.style.left = `${Math.max(8, x)}px`;
    t.style.top = `${y}px`;
  };
  node.addEventListener("pointermove", ev => { fillTip(); place(ev.clientX, ev.clientY); });
  node.addEventListener("pointerleave", () => tip().classList.remove("on"));
  if (!focusable) return;
  if (!node.hasAttribute("tabindex")) node.setAttribute("tabindex", "0");
  if (!node.hasAttribute("aria-label")) node.setAttribute("aria-label", lines().join(". "));
  node.addEventListener("focus", () => {
    fillTip();
    const r = node.getBoundingClientRect();
    place(r.left + Math.min(r.width, 40), r.top + r.height / 2);
  });
  node.addEventListener("blur", () => tip().classList.remove("on"));
}

function seg(id, options, current, onPick) {
  const box = $(id);
  box.replaceChildren();
  for (const [key, label] of options) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.setAttribute("aria-pressed", String(key === current));
    b.addEventListener("click", () => {
      for (const o of box.children) o.setAttribute("aria-pressed", "false");
      b.setAttribute("aria-pressed", "true");
      onPick(key);
    });
    box.appendChild(b);
  }
}

function legend(id, arms) {
  put(id, arms.map(a => h("span", {class: "legend-item"}, h("span", {class: `swatch solid a-sw a-${a}`}), ARM[a] || a)));
}

/* The arm's headline run on a dataset: temporal, seed 0, the best validation AUPRC, ties to
   fewer hops; arm N's variant is chosen the same way. The README's tables use this rule. */
function headline(ds, arm) {
  const pool = D.runs.filter(r => r.dataset === ds && r.kind === "temporal" &&
    (r.arm === arm || (arm === "N" && r.arm.startsWith("N-"))));
  pool.sort((a, b) => ((b.val_auprc != null) - (a.val_auprc != null)) ||
    ((b.val_auprc || 0) - (a.val_auprc || 0)) || (a.hops - b.hops));
  return pool[0] || null;
}

const run = (ds, arm, hops, kind = "temporal") =>
  D.runs.find(r => r.dataset === ds && r.arm === arm && r.hops === hops && r.kind === kind) || null;

/* Rows grouped under a label: on a wide screen the group's name sits in its own column, on a
   narrow one it gets a line above its rows. Returns the height used. */
function layout(groups, narrow, rowH = 26, gap = 16, top = 8) {
  let y = top;
  for (const g of groups) {
    if (narrow && g.label) { g.ly = y + 12; y += 20; }
    g.y0 = y;
    for (const it of g.items) { it.cy = y + rowH / 2; y += rowH; }
    g.y1 = y;
    if (!narrow) g.ly = g.y0 + rowH / 2 + 4;
    y += gap;
  }
  return y;
}

function xAxis(s, x, ticks, y0, y1, fmt, title, W) {
  for (const t of ticks) {
    el("line", {x1: x(t), x2: x(t), y1: y0, y2: y1, class: "gridline"}, s);
    txt(s, x(t), y1 + 15, fmt(t), "tick", {"text-anchor": "middle"});
  }
  if (title) txt(s, W ? W / 2 : x(ticks[Math.floor(ticks.length / 2)]), y1 + 34, title, "axis-title", {"text-anchor": "middle"});
}

function rowLabels(s, groups, narrow, L) {
  for (const g of groups) {
    if (g.label) {
      txt(s, 0, g.ly, g.label, "group-label");
      if (g.note && !narrow) txt(s, 0, g.ly + 15, g.note, "group-note");
    }
    for (const it of g.items) txt(s, L - 12, it.cy + 4, narrow && it.short ? it.short : it.label, "arm-label", {"text-anchor": "end"});
  }
}

/* ---------------------------------------------------------------- the hero */

function hero() {
  const ibmG = headline("ibm-HI-Small", "G"), ibmR = headline("ibm-HI-Small", "R");
  const gateG = D.gate.find(g => g.run.startsWith("ibm-HI-Small.") && g.run.includes(".G."));
  $("sub-g").textContent = pct(ibmG.recall_at_1pct_budget[0]);
  $("sub-r").textContent = pct(ibmR.recall_at_1pct_budget[0]);
  put("hero-recall", h("span", {class: "fig-graph"}, pct(ibmG.recall_at_1pct_budget[0])), " ",
    h("span", {class: "fig-vs"}, "vs"), " ", h("span", {class: "fig-row"}, pct(ibmR.recall_at_1pct_budget[0])));
  $("hero-recall-note").textContent = `IBM HI-Small, synthetic bank transfers: ${ci(ibmG.recall_at_1pct_budget, pct)} with graph features.` +
    (gateG ? ` Project 03's release gate passes it at ${pts(gateG.difference[0])}.` : "");

  const ppG = headline("ellipticpp", "G"), ppR = headline("ellipticpp", "R");
  $("hero-real").textContent = sgn(ppG.delta_auprc[0], f2);
  $("hero-real-note").textContent = `Elliptic++, real wallets: ${f3(ppR.auprc[0])} to ${f3(ppG.auprc[0])}, paired on the same wallets, ${ci(ppG.delta_auprc, v => sgn(v, f2))}.`;

  const sets = Object.keys(SETS).filter(ds => headline(ds, "N") && headline(ds, "G"));
  const wins = sets.filter(ds => headline(ds, "N").auprc[0] > headline(ds, "G").auprc[0]);
  const below = sets.filter(ds => headline(ds, "N").delta_auprc && headline(ds, "N").delta_auprc[2] < 0);
  $("hero-gnn").textContent = `${wins.length} of ${sets.length}`;
  $("hero-gnn-note").textContent = `Graph features led on every dataset` +
    (below.length ? `, and on ${below.map(ds => SETS[ds]).join(" and ")} the network fell below the row model it was meant to beat.` : ".");

  const live = D.serving.filter(r => r.centre === "counters" && r.hops === 1);
  if (live.length) {
    const lo = Math.min(...live.map(r => r.p50_ms)), hi = Math.max(...live.map(r => r.p50_ms));
    const m = live[0].machine;
    $("hero-ms").textContent = `${Math.round(lo)} to ${Math.round(hi)} ms`;
    $("hero-ms-note").textContent = `One hop, across three stores (in memory, DuckDB, Neo4j), 1,000 calls each on a rented ${m.name.replace("RunPod SECURE NVIDIA GeForce ", "")} at US$${m.usd_per_hour}/h (${m.price_date}).`;
  }

  const tR = headline("elliptic", "R"), rR = run("elliptic", "R", 0, "random");
  $("hero-leak").textContent = `${f2(rR.auprc[0])} vs ${f2(tR.auprc[0])}`;
  $("hero-leak-note").textContent = `Elliptic, the same row model: AUPRC on a random split against a split by time. The random split's number is the one most papers report.`;
}

/* ---------------------------------------------------------------- the ring */

let ringArm = "G", ringSel = null;

function ringSummary() {
  const r = D.ring, scored = r.transfers.filter(t => t.scored);
  const caught = a => scored.filter(t => t.arms[a] && t.arms[a].flagged).length;
  return {scored, caught};
}

function drawRing() {
  const r = D.ring, box = $("ring");
  box.replaceChildren();
  const W = Math.max(300, Math.floor(box.clientWidth)), H = Math.round(Math.min(460, Math.max(330, W * 0.78)));
  const s = el("svg", {viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "The ring's accounts and transfers"}, box);
  // Layered left to right by the longest path from an account the ring never pays into, so
  // the money reads in the direction it moved; a cycle has no such account and goes round.
  const names = r.accounts.map(a => a.name), level = Object.fromEntries(names.map(n => [n, 0]));
  const into = new Set(r.transfers.map(t => t.to));
  const acyclic = names.some(n => !into.has(n));
  if (acyclic) for (let k = 0; k < names.length; k++)
    for (const t of r.transfers) if (t.from !== t.to) level[t.to] = Math.max(level[t.to], level[t.from] + 1);
  const depth = Math.max(...Object.values(level));
  const pos = {};
  if (acyclic && depth < names.length) {
    for (let l = 0; l <= depth; l++) {
      const col = names.filter(n => level[n] === l);
      col.forEach((n, i) => { pos[n] = [26 + l * (W - 52) / Math.max(depth, 1), 20 + (i + 0.5) * (H - 40) / col.length]; });
    }
  } else {
    names.forEach((n, i) => {
      const a = -Math.PI / 2 + 2 * Math.PI * i / names.length;
      pos[n] = [W / 2 + (Math.min(W, H) / 2 - 30) * Math.cos(a), H / 2 + (H / 2 - 30) * Math.sin(a)];
    });
  }
  const deg = Object.fromEntries(names.map(n => [n, 0]));
  for (const t of r.transfers) { deg[t.from]++; deg[t.to]++; }
  const defs = el("defs", {}, s);
  for (const [id, cls] of [["on", `arrow-on a-${ringArm}`], ["off", "arrow-off"], ["un", "arrow-un"]]) {
    const m = el("marker", {id: `ar-${id}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 5.5, markerHeight: 5.5, orient: "auto-start-reverse"}, defs);
    el("path", {d: "M0,0 L10,5 L0,10 z", class: cls}, m);
  }
  const layer = el("g", {}, s), hits = el("g", {}, s);
  r.transfers.forEach((t, i) => {
    const [x1, y1] = pos[t.from], [x2, y2] = pos[t.to];
    const dx = x2 - x1, dy = y2 - y1, d = Math.hypot(dx, dy) || 1;
    const sx = x1 + dx / d * 13, sy = y1 + dy / d * 13, ex = x2 - dx / d * 15, ey = y2 - dy / d * 15;
    const bend = 0.05 * d, mx = (sx + ex) / 2 - dy / d * bend, my = (sy + ey) / 2 + dx / d * bend;
    const a = t.arms[ringArm], on = t.scored && a && a.flagged;
    const path = `M${sx},${sy} Q${mx},${my} ${ex},${ey}`;
    el("path", {d: path, class: `edge a-${ringArm}${!t.scored ? " unscored" : on ? " on" : ""}${ringSel === i ? " sel" : ""}`,
      "marker-end": `url(#ar-${!t.scored ? "un" : on ? "on" : "off"})`}, layer);
    const hit = el("path", {d: path, class: "edge-hit", tabindex: 0, role: "button",
      "aria-label": `${t.from} to ${t.to}, US$${Math.round(t.usd).toLocaleString()}`}, hits);
    hover(hit, () => [`${t.from} to ${t.to}`, `US$${Math.round(t.usd).toLocaleString()}, hour ${t.hours} of the ring`,
      ...(t.scored ? FROZEN.filter(k => t.arms[k]).map(k => `${ARM[k]}: ${topLabel(t.arms[k].top)}${t.arms[k].flagged ? ", flagged" : ""}`)
        : ["Not scored: before the test days, or in the generator's tail"])], false);
    const pick = () => { ringSel = i; drawRing(); showTransfer(i); };
    hit.addEventListener("click", pick);
    hit.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); pick(); } });
  });
  for (const n of names) {
    const [x, y] = pos[n];
    el("circle", {cx: x, cy: y, r: deg[n] > 4 ? 14 : 11, class: `node${deg[n] > 4 ? " hub" : ""}`}, s);
    txt(s, x, y + 3.5, n, "node-label");
  }
}

function totals() {
  const {scored, caught} = ringSummary();
  return [h("p", {class: "eyebrow"}, "Scored transfers each model's 1% alert queue reaches"),
    h("div", {class: "caught"}, ["R", "G", "N"].map(a => h("div", {class: `stat a-${a}`},
      h("span", {class: "figure"}, `${caught(a)} of ${scored.length}`), h("span", {class: "caption"}, ARM[a]))))];
}

// The side panel opens on the first transfer graph features flagged, so that a case file is
// on the screen before anyone clicks; every other transfer is one click away.
function ringOverview() {
  const first = D.ring.transfers.findIndex(t => D.ring.cases.some(c => c.transfer === `${t.from} to ${t.to}` && c.hours === t.hours));
  if (first >= 0) { ringSel = first; showTransfer(first); return; }
  put("ring-side", totals(), P("Select a transfer to see where every model ranks it."));
}

function showTransfer(i) {
  const t = D.ring.transfers[i];
  const out = [h("p", {class: "eyebrow"}, `Transfer ${i + 1} of ${D.ring.transfers.length}`),
    h("h3", {}, `${t.from} to ${t.to}`),
    h("p", {class: "note"}, `US$${Math.round(t.usd).toLocaleString()}, hour ${t.hours} of the ring, day ${t.day} of the data${t.laundering ? "; laundering, by the generator's record" : ""}`)];
  if (!t.scored) {
    out.push(P("Not scored: it falls before the test days or in the generator's last days, which no model is graded on."));
  } else {
    // Score bars are sized by class, so the page needs no style attribute.
    for (const a of FROZEN.filter(k => t.arms[k])) {
      out.push(h("div", {class: `score-row a-${a}`},
        h("span", {}, ARM[a], t.arms[a].flagged ? h("span", {class: "flag-pill"}, "flagged") : null),
        h("span", {class: "score-bar rank"}, h("span", {class: `w-${rankWidth(t.arms[a].top)}`})),
        h("span", {class: "num"}, topLabel(t.arms[a].top))));
    }
    out.push(h("p", {class: "note"}, "Where each model ranks this transfer among all test transfers, on a log scale; the line marks the 1% alert queue. Each model's raw score has its own range, so the rank is what compares."));
    const flagged = FROZEN.filter(a => t.arms[a] && t.arms[a].flagged).map(a => ARM[a]);
    out.push(h("p", {class: "note"}, flagged.length ? `In the alert queue of: ${flagged.join(", ")}.` : "In no model's alert queue."));
  }
  const c = D.ring.cases.find(c => c.transfer === `${t.from} to ${t.to}` && c.hours === t.hours);
  if (c) {
    out.push(h("p", {class: "eyebrow"}, "Case file, graph features"),
      h("p", {class: "note"}, `Of ${c.neighbours} accounts next to this transfer, the ones whose removal drops its score most, each rescored through the exported model.`),
      ...c.top.map(n => h("div", {class: "case-row"},
        h("span", {}, h("span", {class: "who"}, n.account), " ",
          h("span", {class: "why"}, n.in_ring ? "in this ring" : n.in_any_laundering ? "in another laundering transfer" : "no laundering on record")),
        h("span", {class: "num"}, `${n.score_drop >= 0 ? "-" : "+"}${f3(Math.abs(n.score_drop))}`))));
  }
  out.push(h("p", {class: "note"}, "Select any arrow in the ring for its transfer."));
  put("ring-side", totals(), out);
}

function ringText() {
  const r = D.ring, {scored, caught} = ringSummary();
  const days = Math.ceil(Math.max(...r.transfers.map(t => t.hours)) / 24);
  $("ring-intro").textContent = `A scatter-gather ring planted by IBM's generator: money leaves one account, ` +
    `is split across ${r.accounts.length - 2} go-betweens and gathers again in one, ${r.transfers.length} transfers over ${days} days. ` +
    `Each transfer on its own looks ordinary. It was chosen by a rule that reads no score, the attempt of this shape with the most ` +
    `transfers on the test days, so it was not picked to flatter any model. Its accounts are named by their place in the ring.`;
  put("ring-verdict", P(B(`The row model's alert queue reaches ${caught("R")} of the ring's ${scored.length} scored transfers; graph features reach ${caught("G")}.`),
    ` The graph neural network reaches ${caught("N")}. Each transfer the row model sees is a modest payment between two accounts. ` +
    "Graph features see that the receiving account is paid by many accounts that were themselves just paid by the same one. " +
    "One ring proves nothing on its own; the charts below measure the same thing over every laundering transfer in the test period."));
}

/* ---------------------------------------------------------------- does the graph help */

let mainView = "auprc";

function drawMain() {
  const view = mainView;
  const groups = Object.keys(SETS).map(ds => ({
    ds, label: SETS[ds], note: SET_NOTE[ds],
    items: FROZEN.filter(a => headline(ds, a) && (view !== "gain" || a !== "R")).map(a => {
      const r = headline(ds, a);
      return {a, r, label: ARM[a] + (a === "R" || a.startsWith("TabFM") ? "" : `, ${hopWord(r.hops)}`), short: a};
    }),
  }));
  const metric = it => view === "auprc" ? it.r.auprc : view === "gain" ? it.r.delta_auprc : it.r.recall_at_1pct_budget;
  const {s, W, narrow} = frame("main", 10, "");
  const L = narrow ? 92 : 320, R = narrow ? 52 : 70;
  const H = layout(groups, narrow) + 36;
  s.setAttribute("viewBox", `0 0 ${W} ${H}`); s.setAttribute("height", H);
  s.setAttribute("aria-label", "Each model's score on each dataset, with 95% intervals");
  let lo = 0, hi = 1;
  if (view === "gain") {
    const all = groups.flatMap(g => g.items.map(metric).filter(Boolean));
    lo = Math.min(-0.1, Math.floor(Math.min(...all.map(v => v[1])) * 10) / 10);
    hi = Math.ceil(Math.max(...all.map(v => v[2])) * 10) / 10;
  }
  const x = v => L + (v - lo) / (hi - lo) * (W - L - R);
  const ticks = [];
  const step = view === "gain" ? (hi - lo > 0.6 ? 0.2 : 0.1) : (narrow ? 0.5 : 0.25);
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) ticks.push(Math.round(t * 100) / 100);
  xAxis(s, x, ticks, 0, H - 30, v => (view === "gain" ? sgn(v, f2) : view === "recall" ? pct(v) : f2(v)), "", W);
  if (view === "gain") el("line", {x1: x(0), x2: x(0), y1: 0, y2: H - 30, class: "zero"}, s);
  rowLabels(s, groups, narrow, L);
  for (const g of groups) {
    const base = headline(g.ds, "R");
    if (view !== "gain" && base) {
      const v = metric({r: base})[0];
      el("line", {x1: x(v), x2: x(v), y1: g.y0 + 2, y2: g.y1 - 2, class: "ref-tick"}, s);
    }
    if (view === "recall" && base && base.ceiling_at_1pct_budget) {
      const c = base.ceiling_at_1pct_budget[0];
      if (c < 0.999) {
        el("line", {x1: x(c), x2: x(c), y1: g.y0, y2: g.y1, class: "budget"}, s);
        txt(s, x(c) + 4, g.y0 + 9, "ceiling", "budget-label");
      }
    }
    for (const it of g.items) {
      const v = metric(it);
      const row = el("g", {class: `a-${it.a}`}, s);
      el("rect", {x: L, y: it.cy - 13, width: W - L, height: 26, class: "hit"}, row);
      if (!v) continue;
      el("line", {x1: x(v[1]), x2: x(v[2]), y1: it.cy, y2: it.cy, class: "whisker"}, row);
      el("circle", {cx: x(v[0]), cy: it.cy, r: 6, class: "dot"}, row);
      txt(row, W - 2, it.cy + 4, view === "gain" ? sgn(v[0], f3) : view === "recall" ? pct(v[0]) : f3(v[0]),
        `value${it.a === "G" ? " strong" : ""}`, {"text-anchor": "end"});
      hover(row, () => [`${g.label}: ${ARM[it.a]}${it.a === "R" ? "" : `, ${hopWord(it.r.hops)}`}`,
        `AUPRC ${f3(it.r.auprc[0])}, ${ci(it.r.auprc, f3)}`,
        ...(it.r.delta_auprc ? [`Against the row model, same items: ${sgn(it.r.delta_auprc[0], f3)}, ${ci(it.r.delta_auprc, v => sgn(v, f3))}`] : []),
        ...(it.r.recall_at_1pct_budget ? [`Recall in the 1% alert queue: ${pct(it.r.recall_at_1pct_budget[0])}`] : [])]);
    }
  }
  legend("main-legend", FROZEN.filter(a => view !== "gain" || a !== "R"));
  if (view !== "gain") $("main-legend").append(h("span", {class: "legend-item"}, h("span", {class: "swatch tick sw-faint"}), "grey tick: the row model, the line to beat"));
}

function mainText() {
  const G = Object.keys(SETS).map(ds => ({ds, r: headline(ds, "G")})).filter(o => o.r && o.r.delta_auprc);
  const all = G.every(o => clear(o.r.delta_auprc));
  const d = ds => G.find(o => o.ds === ds).r.delta_auprc;
  const N = Object.keys(SETS).map(ds => ({ds, r: headline(ds, "N"), g: headline(ds, "G")})).filter(o => o.r);
  const nBelow = N.filter(o => o.r.delta_auprc[2] < 0).map(o => SETS[o.ds]);
  const tg = headline("elliptic", "TabFM-G"), tgp = headline("ellipticpp", "TabFM-G"), tpr = headline("ellipticpp", "TabFM-R");
  put("main-verdict",
    P(B(`Graph features beat the row model on ${all ? "all" : "some"} ${G.length} datasets, each paired interval clear of zero.`),
      ` The gain is ${sgn(d("ibm-HI-Small")[0], f2)} AUPRC on IBM HI-Small and ${sgn(d("ibm-LI-Small")[0], f2)} on LI-Small, where the generator plants ` +
      `laundering as shapes, ${sgn(d("ellipticpp")[0], f2)} on Elliptic++'s real wallets, and ${sgn(d("elliptic")[0], f3)} on Elliptic, whose own ` +
      "features already summarise each transaction's neighbours."),
    P(B("The graph neural network never beat the graph features."),
      ` ${nBelow.length ? `On ${nBelow.join(" and ")} it scored below the plain row model. ` : ""}It learns ` +
      "from the same neighbourhoods but needs a GPU to train, and it moves more from one random seed to the next than the gaps between the models."),
    P(B("The foundation model, never trained on this data, is the surprise on Elliptic:"),
      ` handed the graph features it scores ${f3(tg.auprc[0])}, the best of any model there. On Elliptic++ it does not travel: on the row alone it is ${sgn(tpr.delta_auprc[0], f2)} against the ` +
      `row model, and handed the graph features it scores ${f3(tgp.auprc[0])}, short of the trees' ${f3(headline("ellipticpp", "G").auprc[0])}.`));
  const e = headline("elliptic", "R"), pp = headline("ellipticpp", "R");
  put("main-assumption", P(B("Why the alert-queue view barely separates the models on the real sets."),
    ` A 1% queue holds fewer accounts than there is fraud: at most ${pct(e.ceiling_at_1pct_budget[0])} of Elliptic's illicit transactions and ${pct(pp.ceiling_at_1pct_budget[0])} ` +
    "of Elliptic++'s wallets can fit in it, and every model comes close to that ceiling. AUPRC, which reads the whole ranking, is where they differ. " +
    "On IBM's bank transfers the queue can hold all of the laundering, and there the graph's lead shows in the queue too."));
}

/* ---------------------------------------------------------------- shapes */

function drawPatterns() {
  const heads = Object.fromEntries(["R", "G", "N"].map(a => [a, headline("ibm-HI-Small", a)]));
  const keys = Object.keys(PATTERNS).filter(p => heads.G.patterns[p] && heads.R.patterns[p]);
  // The named shapes by the gain graph features bring, then the generator's random laundering
  // and the transfers with no named shape, which are not shapes at all.
  const named = keys.filter(p => p !== "NONE" && p !== "RANDOM").sort((a, b) =>
    (heads.G.patterns[b][0] - heads.R.patterns[b][0]) - (heads.G.patterns[a][0] - heads.R.patterns[a][0]));
  const order = [...named, ...keys.filter(p => p === "RANDOM"), ...keys.filter(p => p === "NONE")];
  const groups = [{items: order.map(p => ({p, label: PATTERNS[p], short: p === "NONE" ? "No shape" : PATTERNS[p]}))}];
  const shapes = named.length;
  const {s, W, narrow} = frame("patterns", 10, "");
  const L = narrow ? 112 : 170, R = narrow ? 70 : 150;
  const H = layout(groups, false, 32, 0, 6) + 30;
  s.setAttribute("viewBox", `0 0 ${W} ${H}`); s.setAttribute("height", H);
  s.setAttribute("aria-label", "Share of each laundering shape's transfers in the 1% alert queue, row model against graph features");
  const x = v => L + v * (W - L - R);
  xAxis(s, x, narrow ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1], 0, H - 26, pct, "", W);
  rowLabels(s, groups, narrow, L);
  for (const it of groups[0].items) {
    const v = a => heads[a] && heads[a].patterns[it.p];
    const row = el("g", {}, s);
    if (it === groups[0].items[shapes]) el("line", {x1: 0, x2: W, y1: it.cy - 16, y2: it.cy - 16, class: "divider"}, s);
    el("rect", {x: L, y: it.cy - 16, width: W - L, height: 32, class: "hit"}, row);
    el("line", {x1: x(v("R")[0]), x2: x(v("G")[0]), y1: it.cy, y2: it.cy, class: "gain"}, row);
    // Each end's 95% interval, just above (row model) and below (graph features) the bar.
    el("line", {x1: x(v("R")[1]), x2: x(v("R")[2]), y1: it.cy - 9, y2: it.cy - 9, class: "whisker a-R"}, row);
    el("line", {x1: x(v("G")[1]), x2: x(v("G")[2]), y1: it.cy + 9, y2: it.cy + 9, class: "whisker a-G"}, row);
    for (const a of ["R", "N", "G"]) if (v(a)) el("circle", {cx: x(v(a)[0]), cy: it.cy, r: a === "G" ? 7 : 5.5, class: `dot a-${a}`}, row);
    txt(row, W - 2, it.cy + 4, narrow ? sgn((v("G")[0] - v("R")[0]) * 100, n => n.toFixed(0)) :
      `${pct(v("R")[0])} to ${pct(v("G")[0])}`, "value strong", {"text-anchor": "end"});
    hover(row, () => [it.label, ...["R", "G", "N"].filter(v).map(a => `${ARM[a]}: ${pct(v(a)[0])}, ${ci(v(a), pct)}`)]);
  }
  legend("pattern-legend", ["R", "G", "N"]);
  const top = named.slice(0, 2);
  const none = heads.G.patterns.NONE, noneR = heads.R.patterns.NONE;
  put("pattern-verdict",
    P(B(`Graph features reach more of every named shape than the row model, most of all ${PATTERNS[top[0]].toLowerCase()} ` +
      `(${pct(heads.R.patterns[top[0]][0])} to ${pct(heads.G.patterns[top[0]][0])}) and ${PATTERNS[top[1]].toLowerCase()} ` +
      `(${pct(heads.R.patterns[top[1]][0])} to ${pct(heads.G.patterns[top[1]][0])})`),
      ", the shapes where many accounts meet at one. A row model cannot see that account: each payment into it is unremarkable, " +
      "but the number of them is not. The graph neural network sits between the two on most shapes."),
    none && noneR ? P(`Laundering that belongs to no named shape is the hardest for everyone: the queue reaches ${pct(noneR[0])} of it on the row and ${pct(none[0])} with graph features.`) : null);
}

/* ---------------------------------------------------------------- hops */

let hopSet = "ellipticpp";

function hopSeries(ds) {
  return {
    G: [0, 1, 2, 3].map(h => (h === 0 ? run(ds, "R", 0) : run(ds, "G", h))),
    N: [0, 1, 2, 3].map(h => run(ds, "N-sage", h)),
  };
}

function drawHops() {
  const ds = hopSet, ser = hopSeries(ds);
  const {s, W, narrow} = frame("hops", 300, `AUPRC by hop count on ${SETS[ds]}`);
  const H = 300, L = 46, R = narrow ? 40 : 120, T = 14, B = 46;
  const vals = Object.values(ser).flat().filter(Boolean).flatMap(r => r.auprc);
  const ymax = Math.min(1, Math.ceil((Math.max(...vals) + 0.02) * 10) / 10);
  const x = h => L + 10 + h * (W - L - R - 20) / 3, y = v => T + (1 - v / ymax) * (H - T - B);
  const step = ymax > 0.5 ? 0.2 : ymax > 0.2 ? 0.1 : 0.05;
  for (let t = 0; t <= ymax + 1e-9; t += step) {
    el("line", {x1: L, x2: W - R + 10, y1: y(t), y2: y(t), class: "gridline"}, s);
    txt(s, L - 8, y(t) + 4, t.toFixed(2), "tick", {"text-anchor": "end"});
  }
  const fetch = D.serving.filter(r => r.dataset === ds && r.centre === "counters");
  for (let h = 0; h <= 3; h++) {
    txt(s, x(h), H - B + 18, h === 0 ? "row only" : hopWord(h), "tick", {"text-anchor": "middle"});
    const f = fetch.filter(r => r.hops === h);
    if (f.length) txt(s, x(h), H - B + 32, `live ${Math.round(Math.min(...f.map(r => r.p50_ms)))} to ${Math.round(Math.max(...f.map(r => r.p50_ms)))} ms`, "row-note", {"text-anchor": "middle"});
  }
  for (const [arm, runs] of Object.entries(ser)) {
    const p = runs.map((r, h) => r && {h, v: r.auprc}).filter(Boolean);
    if (!p.length) continue;
    const g = el("g", {class: `a-${arm}`}, s);
    const band = p.map(q => `${x(q.h)},${y(q.v[2])}`).concat(p.slice().reverse().map(q => `${x(q.h)},${y(q.v[1])}`));
    el("polygon", {points: band.join(" "), class: "band"}, g);
    el("polyline", {points: p.map(q => `${x(q.h)},${y(q.v[0])}`).join(" "), class: "series"}, g);
    for (const q of p) {
      el("circle", {cx: x(q.h), cy: y(q.v[0]), r: 5.5, class: "dot"}, g);
      const hit = el("circle", {cx: x(q.h), cy: y(q.v[0]), r: 16, class: "hit"}, g);
      hover(hit, () => [`${SETS[ds]}, ${q.h === 0 ? "row only" : hopWord(q.h)}`, arm === "G" && q.h === 0 ? ARM.R : ARM[arm], `AUPRC ${f3(q.v[0])}, ${ci(q.v, f3)}`]);
    }
    if (!narrow) {
      const last = p[p.length - 1];
      txt(g, x(last.h) + 12, y(last.v[0]) + 4, `${arm === "G" ? "Graph features" : "Graph network"} ${f2(last.v[0])}`, "value strong");
    }
  }
  legend("hop-legend", ["G", "N"]);
}

function hopText() {
  const e = hopSeries("elliptic").G, p = hopSeries("ellipticpp").G, i = hopSeries("ibm-HI-Small").G;
  const f = h => D.serving.filter(r => r.dataset === "ibm-HI-Small" && r.centre === "counters" && r.hops === h);
  const ratio = Math.max(...f(2).map(r => r.p50_ms / f(1).find(q => q.store === r.store).p50_ms));
  put("hop-verdict", P(B("One hop is where most of the gain is, and on the bank transfers it is all of it."),
    ` On IBM HI-Small graph features go from ${f3(i[0].auprc[0])} on the row to ${f3(i[1].auprc[0])} at one hop and ${f3(i[2].auprc[0])} at two, ` +
    `while the second hop costs up to ${ratio.toFixed(1)} times as long to fetch. Elliptic++'s wallets keep gaining to two hops ` +
    `(${f3(p[1].auprc[0])} to ${f3(p[2].auprc[0])}); Elliptic, whose features already look one hop out, barely moves after the first ` +
    `(${f3(e[1].auprc[0])} to ${f3(e[3].auprc[0])} at three). The graph neural network gets no better with depth on any set.`));
}

/* ---------------------------------------------------------------- the shift */

let shiftSet = "elliptic";

function drawShift() {
  const ds = shiftSet;
  const arms = [...FROZEN, "R-retrain", "TabFM-R-swap"].map(a => ({a, r: headline(ds, a)})).filter(o => o.r && o.r.from_43);
  const {s, W, narrow} = frame("shift", 360, `AUPRC before and after step 43 on ${SETS[ds]}`);
  const H = 360, T = 30, B = 22, L = narrow ? 40 : 60, R = narrow ? 150 : 270;
  const x0 = L + 20, x1 = W - R;
  const y = v => T + (1 - v) * (H - T - B);
  for (let t = 0; t <= 1.0001; t += 0.2) {
    el("line", {x1: L, x2: x1 + 8, y1: y(t), y2: y(t), class: "gridline"}, s);
    txt(s, L - 8, y(t) + 4, t.toFixed(1), "tick", {"text-anchor": "end"});
  }
  txt(s, x0, 14, "Before step 43", "axis-title", {"text-anchor": "middle"});
  txt(s, x1, 14, "From step 43", "axis-title", {"text-anchor": "middle"});
  el("line", {x1: x0, x2: x0, y1: T - 6, y2: H - B, class: "axis-line"}, s);
  el("line", {x1: x1, x2: x1, y1: T - 6, y2: H - B, class: "axis-line"}, s);
  // End labels, spread so that none overlaps the next.
  const ends = arms.map(o => ({o, y: y(o.r.from_43[0])})).sort((a, b) => a.y - b.y);
  for (let k = 1; k < ends.length; k++) ends[k].y = Math.max(ends[k].y, ends[k - 1].y + 15);
  const over = ends.length ? ends[ends.length - 1].y - (H - B + 4) : 0;
  if (over > 0) for (const e of ends) e.y -= over;
  // Each model's 95% interval after the shutdown, a short vertical line just right of its dot,
  // stepped so that none hides another; the labels start after the last of them.
  const step = narrow ? 3 : 4, labelX = x1 + 10 + arms.length * step;
  arms.forEach((o, k) => {
    const g = el("g", {class: `a-${o.a}`}, s);
    el("line", {x1: x1 + 6 + k * step, x2: x1 + 6 + k * step, y1: y(o.r.from_43[1]), y2: y(o.r.from_43[2]), class: "whisker shift"}, g);
  });
  for (const {o, y: ly} of ends) {
    const g = el("g", {class: `a-${o.a}`}, s);
    const recover = o.a.endsWith("-retrain") || o.a.endsWith("-swap");
    el("line", {x1: x0, x2: x1, y1: y(o.r.before_43[0]), y2: y(o.r.from_43[0]), class: `series${recover ? " dashed" : ""}`}, g);
    el("circle", {cx: x0, cy: y(o.r.before_43[0]), r: 5, class: "dot"}, g);
    el("circle", {cx: x1, cy: y(o.r.from_43[0]), r: 5.5, class: "dot"}, g);
    txt(g, labelX, ly + 4, narrow ? `${o.a} ${f2(o.r.from_43[0])}` : `${ARM[o.a]} ${f2(o.r.from_43[0])}`, `value${recover || o.a === "G" ? " strong" : ""}`);
    const hit = el("line", {x1: x0, x2: x1, y1: y(o.r.before_43[0]), y2: y(o.r.from_43[0]), stroke: "transparent", "stroke-width": 14}, g);
    hover(hit, () => [ARM[o.a], `Before step 43: ${f3(o.r.before_43[0])}, ${ci(o.r.before_43, f3)}`, `From step 43: ${f3(o.r.from_43[0])}, ${ci(o.r.from_43, f3)}`]);
  }
}

function shiftText() {
  const h = a => headline("elliptic", a);
  const frozen = FROZEN.filter(a => h(a));
  const frozenMax = Math.max(...frozen.filter(a => a !== "TabFM-G").map(a => h(a).from_43[0]));
  const re = h("R-retrain"), sw = h("TabFM-R-swap"), tg = h("TabFM-G");
  const ppG = headline("ellipticpp", "G");
  put("shift-verdict",
    P(B("After the shutdown every frozen model collapses on Elliptic, graph or not:"),
      ` AUPRC falls from about ${f1(h("R").before_43[0])} to ${f2(frozenMax)} or less. The one exception is the foundation model handed the graph features, which ` +
      `holds ${f2(tg.from_43[0])}.`),
    P(B("Fresh labels are what recover it."),
      ` Retraining the row model before each test step on the four latest labelled steps brings it to ${f2(re.from_43[0])}. ` +
      `Swapping the foundation model's examples for the same rows brings it to ${f2(sw.from_43[0])}, and that swap costs no training at all. ` +
      `On Elliptic++'s wallets the fall is gentler, and graph features keep ${f2(ppG.from_43[0])} of their ${f2(ppG.before_43[0])}.`));
  put("shift-assumption", P(B("The test after the shutdown is small."),
    ` Elliptic's seven steps after it hold 169 illicit transactions, so every interval here is wide: the swap's ${ci(sw.from_43, f2)} overlaps retraining's ${ci(re.from_43, f2).replace("95% CI ", "")}. ` +
    "It is the better recovery on this test, not a proven one."));
}
const f1 = v => v.toFixed(1);

/* ---------------------------------------------------------------- the usual split */

function drawLeak() {
  const groups = Object.keys(SETS).map(ds => ({
    ds, label: SETS[ds], note: SET_NOTE[ds],
    items: ["R", "G"].map(a => ({a, t: headline(ds, a), r: run(ds, a, a === "R" ? 0 : 1, "random"),
      w: a === "G" ? run(ds, "G-window", 1) : null, label: ARM[a], short: a})).filter(o => o.t && o.r),
  })).filter(g => g.items.length);
  const {s, W, narrow} = frame("leak", 10, "");
  const L = narrow ? 70 : 280, R = narrow ? 50 : 140;
  const H = layout(groups, narrow, 28) + 30;
  s.setAttribute("viewBox", `0 0 ${W} ${H}`); s.setAttribute("height", H);
  s.setAttribute("aria-label", "AUPRC on a time-ordered split against a random split");
  const x = v => L + v * (W - L - R);
  xAxis(s, x, narrow ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1], 0, H - 26, f2, "", W);
  rowLabels(s, groups, narrow, L);
  for (const g of groups) for (const it of g.items) {
    const t = it.t.auprc_all_days[0], r = it.r.auprc[0];
    const row = el("g", {}, s);
    el("rect", {x: L, y: it.cy - 14, width: W - L, height: 28, class: "hit"}, row);
    el("line", {x1: x(t), x2: x(r), y1: it.cy, y2: it.cy, class: "inflate"}, row);
    el("circle", {cx: x(t), cy: it.cy, r: 6, class: "dot split-t"}, row);
    el("circle", {cx: x(r), cy: it.cy, r: 6, class: "dot split-r"}, row);
    if (it.w) el("circle", {cx: x(it.w.auprc_all_days[0]), cy: it.cy, r: 6, class: "dot split-w"}, row);
    txt(row, W - 2, it.cy + 4, narrow ? sgn(r - t, f2) : `${f2(t)} to ${f2(r)}`, "value strong", {"text-anchor": "end"});
    hover(row, () => [`${g.label}: ${ARM[it.a]}`, `Time-ordered: ${f3(t)}, ${ci(it.t.auprc_all_days, f3)}`, `Random: ${f3(r)}, ${ci(it.r.auprc, f3)}`,
      ...(it.w ? [`Features that could see later days: ${f3(it.w.auprc_all_days[0])}`] : [])]);
  }
  const e = headline("elliptic", "R"), er = run("elliptic", "R", 0, "random");
  const p = headline("ellipticpp", "R"), pr = run("ellipticpp", "R", 0, "random");
  const iG = headline("ibm-HI-Small", "G"), iw = run("ibm-HI-Small", "G-window", 1);
  put("leak-verdict",
    P(B("On the real Bitcoin data a random split makes every model look nearly perfect."),
      ` The row model scores ${f2(er.auprc[0])} on Elliptic and ${f2(pr.auprc[0])} on Elliptic++ when accounts are split at random, against ` +
      `${f2(e.auprc_all_days[0])} and ${f2(p.auprc_all_days[0])} when it must score a later period. A random split lets the model learn from the ` +
      "very weeks it is tested on, and it hides the drift a bank lives with."),
    P("IBM's synthetic sets go the other way, because the generator's last days, nearly all laundering, fall in the time-ordered " +
      "test. There the leak that mattered was one this build found in itself: graph features computed with sight of later days " +
      `scored ${f2(iw.auprc_all_days[0])} against the honest ${f2(iG.auprc_all_days[0])}. It was fixed, and a leakage test now runs on every build.`));
}

/* ---------------------------------------------------------------- explanations */

function drawFaith() {
  const rows = D.faithfulness.filter(r => r.dataset === "ibm-HI-Small").sort((a, b) => a.k - b.k);
  const groups = [{items: rows.map(r => ({r, label: `Top ${r.k} neighbour${r.k === 1 ? "" : "s"}`, short: `Top ${r.k}`}))}];
  const {s, W, narrow} = frame("faith", 10, "");
  const L = narrow ? 70 : 170, R = narrow ? 60 : 190;
  const H = layout(groups, false, 40, 0, 6) + 30;
  s.setAttribute("viewBox", `0 0 ${W} ${H}`); s.setAttribute("height", H);
  s.setAttribute("aria-label", "How often the explanation's neighbours include a laundering account, against random neighbours");
  const x = v => L + v * (W - L - R);
  xAxis(s, x, narrow ? [0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1], 0, H - 26, pct, "", W);
  rowLabels(s, groups, false, L);
  for (const it of groups[0].items) {
    const r = it.r, row = el("g", {}, s);
    el("rect", {x: L, y: it.cy - 20, width: W - L, height: 40, class: "hit"}, row);
    el("line", {x1: x(r.random[1]), x2: x(r.random[2]), y1: it.cy + 6, y2: it.cy + 6, class: "whisker rand"}, row);
    el("circle", {cx: x(r.random[0]), cy: it.cy + 6, r: 5, class: "dot rand"}, row);
    el("line", {x1: x(r.explained[1]), x2: x(r.explained[2]), y1: it.cy - 6, y2: it.cy - 6, class: "whisker pick"}, row);
    el("circle", {cx: x(r.explained[0]), cy: it.cy - 6, r: 6, class: "dot pick"}, row);
    txt(row, W - 2, it.cy + 4, narrow ? pts(r.lift[0]).replace(" points", "") : `lift ${pts(r.lift[0])}`, `value${clear(r.lift) ? " strong" : ""}`, {"text-anchor": "end"});
    hover(row, () => [`${it.label}, ${r.graded} flagged transfers graded`, `Explanation: ${pct(r.explained[0])}, ${ci(r.explained, pct)}`,
      `Random: ${pct(r.random[0])}, ${ci(r.random, pct)}`, `Lift: ${pts(r.lift[0])}, ${ci(r.lift, v => pts(v).replace(" points", ""))}`]);
  }
  const k1 = rows.find(r => r.k === 1);
  $("faith-caption").textContent = `IBM HI-Small, a random ${k1.flagged} of graph features' alert queue; graded where a transfer has more than k neighbours and at least one known laundering account among them. Intervals over hours. Elliptic's flagged transactions mostly have one or two neighbours, too few to grade.`;
  const k3 = rows.find(r => r.k === 3);
  put("faith-verdict", P(B("The explanation is a better-than-random pointer, not a reliable one."),
    ` The single neighbour it ranks first is an account the generator used for laundering ${pct(k1.explained[0])} of the time, against ${pct(k1.random[0])} for a ` +
    `neighbour picked at random: a lift of ${pts(k1.lift[0])} (95% CI ${(k1.lift[1] * 100).toFixed(0)} to ${(k1.lift[2] * 100).toFixed(0)}). ` +
    `${k3 && !clear(k3.lift) ? "At three neighbours the interval spans zero. " : ""}An investigator gets a place to start, and the ` +
    "case file presents it as no more than that."));
}

/* ---------------------------------------------------------------- the gate */

function drawGate() {
  const name = g => {
    const ds = Object.keys(SETS).find(k => g.run.startsWith(k + "."));
    const arm = armKey(g.run.split(".")[2]);
    const hops = Number(g.run.split(".")[3].slice(1));
    return {ds, arm, label: SETS[ds], sub: `${ARM[arm]}, ${hopWord(hops)}`};
  };
  const rows = D.gate.map(g => ({g, ...name(g)}))
    .sort((a, b) => Object.keys(SETS).indexOf(a.ds) - Object.keys(SETS).indexOf(b.ds) || (a.arm > b.arm ? 1 : -1));
  // Two decisions on Elliptic's graph features: keep the one for the headline run.
  const keep = rows.filter(r => !(r.arm === "G" && headline(r.ds, "G") && !r.g.run.includes(`.h${headline(r.ds, "G").hops}.`)));
  const groups = [{items: keep.map(r => ({...r, label: `${r.label}: ${r.arm === "G" ? "graph features" : "graph network"}`, short: `${r.ds.replace("ibm-", "")} ${r.arm}`}))}];
  const {s, W, narrow} = frame("gate", 10, "");
  const L = narrow ? 110 : 260, R = narrow ? 50 : 90;
  const H = layout(groups, false, 30, 0, 24) + 34;
  s.setAttribute("viewBox", `0 0 ${W} ${H}`); s.setAttribute("height", H);
  s.setAttribute("aria-label", "Recall in the 1% alert queue, challenger minus row model, with the gate's verdict");
  const lo = Math.min(-0.06, ...keep.map(r => r.g.difference[1])), hi = Math.max(0.05, ...keep.map(r => r.g.difference[2]));
  const x = v => L + (v - Math.floor(lo * 20) / 20) / (Math.ceil(hi * 20) / 20 - Math.floor(lo * 20) / 20) * (W - L - R);
  const ticks = [];
  for (let t = Math.floor(lo * 20) / 20; t <= hi + 1e-9; t += 0.05) ticks.push(Math.round(t * 100) / 100);
  xAxis(s, x, ticks, 20, H - 30, v => sgn(v * 100, n => n.toFixed(0)), "", W);
  const m = -keep[0].g.margin;
  el("line", {x1: x(m), x2: x(m), y1: 6, y2: H - 30, class: "budget"}, s);
  txt(s, x(m) - 4, 14, `margin, ${(m * 100).toFixed(0)} points`, "budget-label", {"text-anchor": "end"});
  el("line", {x1: x(0), x2: x(0), y1: 20, y2: H - 30, class: "zero"}, s);
  rowLabels(s, groups, false, L);
  for (const it of groups[0].items) {
    const d = it.g.difference, row = el("g", {class: `a-${it.arm}`}, s);
    el("rect", {x: L, y: it.cy - 15, width: W - L, height: 30, class: "hit"}, row);
    el("rect", {x: Math.min(x(0), x(d[0])), y: it.cy - 8, width: Math.max(2, Math.abs(x(d[0]) - x(0))), height: 16, rx: 3, class: "bar gate"}, row);
    el("line", {x1: x(d[1]), x2: x(d[2]), y1: it.cy, y2: it.cy, class: "whisker cost"}, row);
    txt(row, W - 2, it.cy + 4, it.g.verdict, `verdict-label ${it.g.verdict}`, {"text-anchor": "end"});
    hover(row, () => [`${SETS[it.ds]}: ${it.sub}`, `Recall in the 1% queue: ${pct(it.g.baseline_recall)} for the row model, ${pct(it.g.candidate_recall)} for this one`,
      `Difference ${pts(d[0])}, ${ci(d, v => pts(v).replace(" points", ""))}`, `${it.g.items.toLocaleString()} illicit items, paired; ${it.g.date}`]);
  }
  const ibm = keep.filter(r => r.ds.startsWith("ibm-") && r.arm === "G");
  const n = keep.filter(r => r.arm === "N" && r.ds.startsWith("ibm-"));
  put("gate-verdict",
    P(B("On the bank transfers the gate passes graph features with a wide margin:"),
      ` ${ibm.map(r => `${pts(r.g.difference[0])} of laundering reached on ${SETS[r.ds]}`).join(" and ")}, every interval far above the line. ` +
      `The graph network passes too, at ${n.map(r => pts(r.g.difference[0])).join(" and ")}.`),
    P("On the real sets the 1% queue is smaller than the fraud, so every model fills it with the same sure cases and the gate " +
      "has little to separate. Where it blocks, it cannot rule out a 3-point drop; it did not see one."));
}

/* ---------------------------------------------------------------- live cost */

let costView = "counters";

function drawCost() {
  const rows = D.serving.filter(r => r.centre === costView && r.dataset === "ibm-HI-Small");
  const stores = ["memory", "duckdb", "neo4j"], SN = {memory: "In memory", duckdb: "DuckDB", neo4j: "Neo4j"};
  const groups = stores.map(st => ({label: SN[st], items: [1, 2].map(h => rows.find(r => r.store === st && r.hops === h))
    .filter(Boolean).map(r => ({r, label: hopWord(r.hops), short: hopWord(r.hops)}))})).filter(g => g.items.length);
  const {s, W, narrow} = frame("cost", 10, "");
  const L = narrow ? 70 : 200, R = narrow ? 64 : 190;
  const H = layout(groups, narrow, 26, 12, 22) + 30;
  s.setAttribute("viewBox", `0 0 ${W} ${H}`); s.setAttribute("height", H);
  s.setAttribute("aria-label", "Live latency per score, median and 99th percentile, by store and hop count");
  const top = Math.pow(10, Math.ceil(Math.log10(Math.max(...rows.map(r => r.p99_ms)))));
  const x = v => L + Math.log10(Math.max(v, 1)) / Math.log10(top) * (W - L - R);
  const ticks = [];
  for (let t = 1; t <= top; t *= 10) ticks.push(t);
  xAxis(s, x, ticks, 18, H - 26, v => `${v.toLocaleString()} ms`, "", W);
  el("line", {x1: x(50), x2: x(50), y1: 4, y2: H - 26, class: "budget"}, s);
  txt(s, x(50) + 4, 13, "a 50 ms decision budget", "budget-label");
  rowLabels(s, groups, narrow, L);
  for (const g of groups) for (const it of g.items) {
    const r = it.r, row = el("g", {}, s);
    el("rect", {x: L, y: it.cy - 13, width: W - L, height: 26, class: "hit"}, row);
    el("rect", {x: L, y: it.cy - 8, width: Math.max(2, x(r.p50_ms) - L), height: 16, rx: 3, class: `bar cost h${r.hops}`}, row);
    el("line", {x1: x(r.p50_ms), x2: x(r.p99_ms), y1: it.cy, y2: it.cy, class: "whisker cost"}, row);
    el("line", {x1: x(r.p99_ms), x2: x(r.p99_ms), y1: it.cy - 5, y2: it.cy + 5, class: "whisker cost"}, row);
    txt(row, W - 2, it.cy + 4, narrow ? `${r.p50_ms.toFixed(1)} ms` : `${r.p50_ms.toFixed(1)} ms, US$${r.usd_per_million.toFixed(2)} per million`, "value", {"text-anchor": "end"});
    hover(row, () => [`${SN[r.store]}, ${hopWord(r.hops)}`, `Median ${r.p50_ms.toFixed(1)} ms, 99th percentile ${r.p99_ms.toFixed(1)} ms`,
      `${Math.round(r.mean_edges_fetched).toLocaleString()} transfers fetched per score, on average`, `US$${r.usd_per_million.toFixed(2)} per million scores, the machine busy half the hour`]);
  }
  const m = rows[0].machine;
  $("cost-caption").textContent = `IBM HI-Small, ${rows[0].calls.toLocaleString()} calls per row after warm-up, one worker, on ${m.name} at US$${m.usd_per_hour} an hour (${m.price_date}). Log scale. The features are identical whichever way an account's own counts are found; only the fetch differs.`;
  const c = D.serving.filter(r => r.centre === "counters"), w = D.serving.filter(r => r.centre === "history");
  const worst = rs => rs.reduce((a, b) => (b.p99_ms > a.p99_ms ? b : a));
  const cw = worst(c.filter(r => r.hops === 1)), ww = worst(w);
  put("cost-verdict",
    P(B("One hop with running counters fits a live decision in every store tested:"),
      ` a median of ${Math.round(Math.min(...c.filter(r => r.hops === 1).map(r => r.p50_ms)))} to ${Math.round(Math.max(...c.filter(r => r.hops === 1).map(r => r.p50_ms)))} ms and ` +
      "a few dollars per million scores. Fetching each account's whole history instead looks similar at the median and fails " +
      `at the tail: ${SN[ww.store]}'s 99th percentile reaches ${(ww.p99_ms / 1000).toFixed(1)} seconds, against ${Math.round(cw.p99_ms)} ms with counters. ` +
      "Which store matters less than keeping the counts an account already has, instead of fetching them on every payment."),
    P("The foundation model is not in this chart: its weights' licence rules out serving it, and it scored hundreds of " +
      "milliseconds a row on a GPU."));
}

/* ---------------------------------------------------------------- limits, rejected, table */

function limits() {
  const sd = D.seeds.filter(r => r.dataset.startsWith("elliptic"));
  const spread = r => Math.max(...Object.values(r.auprc)) - Math.min(...Object.values(r.auprc));
  const pp = headline("ellipticpp", "G");
  const k1 = D.faithfulness.find(r => r.dataset === "ibm-HI-Small" && r.k === 1);
  const cards = [
    ["The graph network moves with its seed", `Rerun with two more random seeds, its AUPRC moves by up to ${f2(Math.max(...sd.map(spread)))} on the real sets, as much as some gaps between the models. It stays below graph features at every seed.`],
    ["The biggest gains are on synthetic data", `IBM's generator plants laundering as shapes on purpose. On the real wallets of Elliptic++ the gain is ${sgn(pp.delta_auprc[0], f2)}, but its interval runs from ${f2(pp.delta_auprc[1])} to ${f2(pp.delta_auprc[2])}.`],
    ["The test after the shutdown is small", `The seven steps after Elliptic's step 43 hold 169 illicit transactions, so every interval in the shift chart is wide.`],
    ["Explanations are only somewhat better than random", `The top neighbour is a laundering account ${pts(k1.lift[0])} more often than a random one, and at three neighbours the difference is not measurable.`],
  ];
  put("limit-cards", cards.map(([head, body]) => h("article", {class: "guide-card limit"}, h("h3", {}, head), P(body))));
  const t = run("elliptic", "G+nlr", 1), r = run("elliptic", "G+nlr", 1, "random");
  $("rejected-text").textContent = "It is guilt by association in one column, and published graph fraud results often lean on it. Here it " +
    `costs graph features ${f3(-t.delta_auprc[0])} AUPRC on Elliptic's time-ordered split (95% CI ${f3(-t.delta_auprc[2])} to ${f3(-t.delta_auprc[1])}): a new transaction's neighbours sit in its own time step, whose labels are not known yet, ` +
    `so the column is empty exactly where it is needed. On a random split it adds ${sgn(r.delta_auprc[0], f3)}, an interval that touches zero, because a random split ` +
    "hands the model the test step's labels. The lift it shows in the literature comes from the split, not the column.";
}

function allTable() {
  const head = ["Dataset", "Model", "Hops", "AUPRC", "Against the row model", "Recall, 1% false positives", "Recall, 1% alert queue"];
  const thead = h("thead", {}, h("tr", {}, head.map((name, i) => h("th", i > 1 ? {class: "num"} : {}, name))));
  const tbody = h("tbody", {});
  for (const ds of Object.keys(SETS)) {
    FROZEN.filter(a => headline(ds, a)).forEach((a, i) => {
      const r = headline(ds, a);
      const cell = (t, f) => h("td", {class: "num"}, t ? [f(t[0]), h("br"), h("span", {class: "interval"}, `${f(t[1])} to ${f(t[2])}`)] : null);
      const variant = a === "N" ? [h("br"), h("span", {class: "interval"}, r.arm.slice(2).replace(/\+/g, ", ").replace("sage", "GraphSAGE").replace("gat", "GAT").replace("rmp", "reverse messages").replace("ego", "ego ids"))] : null;
      tbody.append(h("tr", i === 0 ? {class: "group-start"} : {},
        h("td", {}, i === 0 ? SETS[ds] : ""), h("td", {class: "arm"}, ARM[a], variant),
        h("td", {class: "num"}, String(r.hops)), cell(r.auprc, f3), cell(r.delta_auprc, v => sgn(v, f3)),
        cell(r.recall_at_1pct_fpr, f3), cell(r.recall_at_1pct_budget, f3)));
    });
  }
  put("all-table", thead, tbody, h("caption", {}, "Seed 0, time-ordered split, 95% intervals over time steps or hours. IBM rows leave out the generator's last days."));
}

/* ---------------------------------------------------------------- start */

function drawAll() {
  drawRing(); drawMain(); drawPatterns(); drawHops(); drawShift(); drawLeak(); drawFaith(); drawGate(); drawCost();
}

function boot(data) {
  D = data;
  hero();
  ringText(); ringOverview();
  seg("ring-arms", ["R", "G", "N"].map(a => [a, ARM[a]]), ringArm, a => {
    ringArm = a; drawRing(); if (ringSel != null) showTransfer(ringSel);
  });
  seg("main-view", [["auprc", "Precision-recall AUC"], ["gain", "Gain over the row model"], ["recall", "Share caught in a 1% alert queue"]],
    mainView, v => { mainView = v; drawMain(); });
  seg("hop-sets", Object.entries(SETS), hopSet, ds => { hopSet = ds; drawHops(); });
  seg("shift-sets", [["elliptic", "Elliptic"], ["ellipticpp", "Elliptic++"]], shiftSet, ds => { shiftSet = ds; drawShift(); });
  seg("cost-view", [["counters", "Own counts kept as running counters"], ["history", "Own counts fetched from the whole history"]],
    costView, v => { costView = v; drawCost(); });
  mainText(); hopText(); shiftText(); limits(); allTable();
  drawAll();
  let width = window.innerWidth;
  window.addEventListener("resize", () => {
    if (window.innerWidth === width) return;
    width = window.innerWidth;
    drawAll();
  });
  $("build-stamp").textContent = `Built ${D.built} from commit ${D.commit}. Every figure on this page is read from data.json, which the repository's site command writes from its committed results.`;
}

fetch("data.json")
  .then(r => r.json())
  .then(boot)
  .catch(err => {
    document.querySelector(".hero .lead").after(
      h("p", {class: "warning"}, `The figures could not be loaded (${String(err)}). The numbers are in the repository's README.`));
  });
