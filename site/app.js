"use strict";

const SVGNS = "http://www.w3.org/2000/svg";
const TZ = "America/New_York";
const LOW_WATER = 52.41; // driest 10 % of days 1979-2021
const CATS = [["major", "Major flood"], ["moderate", "Moderate flood"], ["minor", "Minor flood"], ["action", "Action stage"]];
const CAT_LABEL = { normal: "Normal", action: "Action stage", minor: "Minor flood", moderate: "Moderate flood", major: "Major flood", unknown: "No reading" };
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function el(tag, attrs = {}, parent = null, text = null) {
  const svgTags = new Set(["svg", "g", "rect", "line", "path", "text", "circle", "clipPath", "defs", "title", "tspan"]);
  const node = svgTags.has(tag) ? document.createElementNS(SVGNS, tag) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) node.setAttribute(k, v);
  if (text !== null) node.textContent = text;
  if (parent) parent.appendChild(node);
  return node;
}

const day = (s) => { const [y, m, d] = s.split("-").map(Number); return new Date(y, m - 1, d, 12); }; // local noon of a calendar day
// The gauge's day (USGS daily mean) runs midnight to midnight Eastern Standard Time, i.e. 05Z to 05Z.
const gaugeDay = (s) => { const [y, m, d] = s.split("-").map(Number); const a = new Date(Date.UTC(y, m - 1, d, 5)); return [a, new Date(a.getTime() + 864e5)]; };
const ft = (x, nd = 2) => (x === null || x === undefined || Number.isNaN(x) ? "—" : x.toFixed(nd));
const fmtDay = (d, opts = { weekday: "short", month: "short", day: "numeric" }) => d.toLocaleDateString("en-US", opts);
const weekday = (d) => d.toLocaleDateString("en-US", { weekday: "long" });

function category(stage, stages) {
  if (stage === null || stage === undefined) return "unknown";
  for (const [k] of CATS) if (stage >= stages[k]) return k;
  return "normal";
}

async function getJSON(path) {
  const r = await fetch(path, { cache: "no-cache" });
  if (!r.ok) throw new Error(`${path}: HTTP ${r.status}`);
  return r.json();
}

/* ---------------- reading + headline ---------------- */

function trend24h(hourly) {
  if (!hourly || hourly.length < 2) return null;
  const last = hourly[hourly.length - 1];
  const t = Date.parse(last.time) - 24 * 3600e3;
  let prev = null;
  for (const h of hourly) if (Date.parse(h.time) <= t) prev = h;
  if (!prev || last.stage === null || prev.stage === null) return null;
  return last.stage - prev.stage;
}

function renderReading(fc, obs) {
  const cur = fc.current || {};
  document.getElementById("stage-value").textContent = ft(cur.stage_ft);
  const chip = document.getElementById("category");
  chip.dataset.cat = cur.category || "unknown";
  chip.querySelector(".chip-text").textContent = CAT_LABEL[cur.category] || "No reading";

  const d = trend24h(obs && obs.hourly);
  const trend = document.getElementById("trend");
  if (d === null) trend.textContent = "";
  else if (Math.abs(d) < 0.03) trend.textContent = "Steady over the last 24 hours";
  else trend.textContent = `${d > 0 ? "Rising" : "Falling"} ${Math.abs(d).toFixed(2)} ft in 24 hours`;

  if (cur.time) {
    const t = new Date(cur.time);
    document.getElementById("as-of").textContent =
      `Gauge reading ${t.toLocaleString("en-US", { timeZone: TZ, weekday: "short", month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" })}`;
  }
  document.getElementById("headline").textContent = headline(fc);
}

function headline(fc) {
  const rows = (fc.forecast || []).filter((r) => r.p50 !== null);
  if (!rows.length) return "No forecast is available right now.";
  const st = fc.flood_stages;
  const peak = rows.reduce((a, b) => (b.p50 > a.p50 ? b : a));
  const low = rows.reduce((a, b) => (b.p50 < a.p50 ? b : a));
  const last = rows[rows.length - 1];
  const firstOver = rows.find((r) => r.p50 >= st.action);
  if (firstOver) {
    return `Expected to reach ${CAT_LABEL[category(peak.p50, st)].toLowerCase()} levels by ${weekday(day(firstOver.date))}, peaking near ${ft(peak.p50, 1)} ft on ${weekday(day(peak.date))}.`;
  }
  const risky = rows.find((r) => r.p90 !== null && r.p90 >= st.action);
  if (risky) {
    return `Expected to stay below flood stage, but the upper end of the likely range reaches action stage (${st.action} ft) by ${weekday(day(risky.date))}. Expected peak ${ft(peak.p50)} ft.`;
  }
  if (low.p50 < LOW_WATER) {
    return `Low water: expected to drop to ${ft(low.p50)} ft by ${weekday(day(low.date))}, lower than on 9 in 10 days on record.`;
  }
  const below = (st.action - peak.p50).toFixed(1);
  const first = rows[0];
  if (peak !== first && peak !== last && peak.p50 - first.p50 > 0.05) {
    return `Below flood stage all week. Expected to crest at ${ft(peak.p50)} ft on ${weekday(day(peak.date))}, ${below} ft under action stage, then ease to ${ft(last.p50)} ft.`;
  }
  const change = last.p50 - first.p50;
  const dir = Math.abs(change) < 0.1 ? "holding near" : change > 0 ? "rising to" : "falling to";
  return `Below flood stage all week, ${dir} ${ft(last.p50)} ft by ${weekday(day(last.date))}. The highest expected level is ${below} ft under action stage.`;
}

function renderWeek(fc) {
  const body = document.getElementById("week-body");
  body.replaceChildren();
  for (const r of fc.forecast || []) {
    const tr = el("tr", {}, body);
    el("td", {}, tr, fmtDay(day(r.date)));
    const exp = el("td", {}, tr, ft(r.p50));
    if (r.category && r.category !== "normal") addFlag(exp, r.category);
    const rng = el("td", {}, tr, `${ft(r.p10)}–${ft(r.p90)}`);
    if (r.p90_category && r.p90_category !== "normal" && r.p90_category !== r.category) addFlag(rng, r.p90_category);
    el("td", {}, tr, r.rain_mm === null || r.rain_mm === undefined ? "—" : `${(r.rain_mm / 25.4).toFixed(2)} in`);
  }
}

function addFlag(cell, cat) {
  const f = el("span", { class: "flag", style: `background: var(--cat-${cat})`, "aria-hidden": "true" }, cell);
  f.title = CAT_LABEL[cat];
  el("span", { class: "sr-only" }, cell, ` (${CAT_LABEL[cat]})`);
}

/* ---------------- staff gauge ---------------- */

function renderGauge(fc) {
  const svg = document.getElementById("gauge");
  svg.replaceChildren();
  const st = fc.flood_stages;
  const cur = fc.current && fc.current.stage_ft;
  const rows = (fc.forecast || []).filter((r) => r.p50 !== null);
  const lo10 = Math.min(...rows.map((r) => r.p10 ?? r.p50), cur ?? 99);
  const hi90 = Math.max(...rows.map((r) => r.p90 ?? r.p50), cur ?? 0);
  const lo = Math.min(54, Math.floor(lo10 - 0.8));
  const hi = Math.max(63, Math.ceil(hi90 + 0.6));

  const W = 300, H = 700, top = 10, bot = 10;
  const bx = 96, bw = 76; // enamel board
  const ppf = (H - top - bot) / (hi - lo); // px per foot
  const y = (v) => top + (hi - v) * ppf;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  el("title", {}, svg, `Staff gauge: ${ft(cur)} ft now; action ${st.action}, minor ${st.minor}, moderate ${st.moderate}, major ${st.major} ft.`);
  const defs = el("defs", {}, svg);
  el("rect", { x: bx, y: top, width: bw, height: H - top - bot, rx: 3 }, el("clipPath", { id: "board-clip" }, defs));

  // flood category bands beside the board, labelled at their lower edge
  const bands = [["action", st.action, st.minor], ["minor", st.minor, st.moderate], ["moderate", st.moderate, st.major], ["major", st.major, hi]];
  for (const [k, a, b] of bands) {
    el("rect", { x: bx + bw + 8, y: y(b) + 1, width: 8, height: y(a) - y(b) - 2, rx: 2, fill: css(`--cat-${k}`) }, svg);
    el("text", { x: bx + bw + 24, y: y(a) - 5, "font-size": 13, fill: css("--ink-2") }, svg, `${CAT_LABEL[k].replace(/ (flood|stage)$/, "")} ${a}`);
  }
  if (LOW_WATER > lo) {
    el("line", { x1: bx + bw + 8, x2: bx + bw + 16, y1: y(LOW_WATER), y2: y(LOW_WATER), stroke: css("--muted"), "stroke-width": 2 }, svg);
    el("text", { x: bx + bw + 24, y: y(LOW_WATER) + 4, "font-size": 12, fill: css("--muted") }, svg, `low water ${LOW_WATER.toFixed(1)}`);
  }

  // board with USGS-style E graduations: one E per tenth (arms 0.02 ft at .00/.04/.08), alternating sides
  const g = el("g", { "clip-path": "url(#board-clip)", "shape-rendering": "crispEdges" }, svg);
  el("rect", { x: bx, y: top, width: bw, height: H - top - bot, fill: css("--board") }, g);
  const ink = css("--board-ink"), arm = 0.02 * ppf, ew = bw / 2 - 8;
  for (let tenth = Math.round(lo * 10); tenth < hi * 10; tenth++) {
    const base = y(tenth / 10), left = tenth % 2 === 0;
    const ex = left ? bx + 5 : bx + bw - 5 - ew;
    const spineX = left ? ex : ex + ew - 4;
    el("rect", { x: spineX, y: base - 0.1 * ppf, width: 4, height: 0.1 * ppf, fill: ink }, g);
    for (const k of [0, 0.04, 0.08]) el("rect", { x: ex, y: base - (k + 0.02) * ppf, width: ew, height: arm, fill: ink }, g);
  }
  for (let f = lo + 1; f < hi; f++) {
    el("text", { x: bx - 10, y: y(f) + 5, "text-anchor": "end", "font-size": 15, "font-weight": 600, fill: css("--ink-2") }, svg, String(f));
  }

  // water to the current reading, drifting surface, pointer + value
  if (cur !== null && cur !== undefined) {
    const wy = y(cur);
    el("rect", { x: bx, y: wy, width: bw, height: H - bot - wy, fill: css("--water") }, g);
    const wave = el("path", { class: "ripple", fill: "none", stroke: css("--water-edge"), "stroke-width": 2, "shape-rendering": "auto" }, g);
    let d = `M ${bx - 24} ${wy}`;
    for (let xx = bx - 24; xx < bx + bw + 24; xx += 12) d += " q 6 -3 12 0";
    wave.setAttribute("d", d);
    const right = cur < st.action - 0.4;
    const px = right ? bx + bw + 4 : bx - 4;
    el("path", { d: `M ${px} ${wy} l ${right ? 10 : -10} -6 v 12 z`, fill: css("--ink") }, svg);
    if (right) el("text", { x: px + 16, y: wy + 5, "font-size": 14, "font-weight": 600, fill: css("--ink") }, svg, `${ft(cur)} now`);
  }

  // the week's likely range as a bracket, peak expected level as a dot
  if (rows.length) {
    const peak = rows.reduce((a, b) => (b.p50 > a.p50 ? b : a));
    const k = 24;
    el("line", { x1: k, x2: k, y1: y(hi90), y2: y(lo10), stroke: css("--fc"), "stroke-width": 2, "stroke-linecap": "round" }, svg);
    for (const v of [hi90, lo10]) el("line", { x1: k, x2: k + 8, y1: y(v), y2: y(v), stroke: css("--fc"), "stroke-width": 2, "stroke-linecap": "round" }, svg);
    el("circle", { cx: k, cy: y(peak.p50), r: 5, fill: css("--fc"), stroke: css("--page"), "stroke-width": 2 }, svg);
    const lbl = el("text", { x: 6, y: y(lo10) + 20, "font-size": 12, fill: css("--ink-2") }, svg);
    el("tspan", { x: 6, dy: 0 }, lbl, "next");
    el("tspan", { x: 6, dy: 14 }, lbl, "7 days");
  }
}

/* ---------------- hydrograph ---------------- */

function renderHydro(fc, obs) {
  const host = document.getElementById("hydro");
  host.replaceChildren();
  const W = Math.max(300, host.clientWidth);
  const H = Math.round(Math.min(400, Math.max(240, W * 0.42)));
  const m = { l: 44, r: W < 560 ? 16 : 112, t: 12, b: 28 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, tabindex: 0, role: "img",
    "aria-label": "Creek level over the last 30 days and the 7-day forecast with its likely range" }, host);

  const st = fc.flood_stages;
  const rows = (fc.forecast || []).filter((r) => r.p50 !== null);
  const now = fc.current && fc.current.time ? new Date(fc.current.time) : new Date();
  const start = new Date(now.getTime() - 30 * 864e5);
  const daily = (obs.daily || []).filter((r) => r.stage !== null && day(r.date) >= start);
  const hourly = (obs.hourly || []).filter((r) => r.stage !== null).map((r) => ({ t: new Date(r.time), v: r.stage }));
  const hourlyStart = hourly.length ? hourly[0].t : now;
  const obsLine = daily.filter((r) => day(r.date) < hourlyStart).map((r) => ({ t: day(r.date), v: r.stage })).concat(hourly);

  // Each forecast is that day's expected average, so it is drawn as a step spanning the gauge's day, not a point.
  const fcSteps = rows.map((r) => { const [a, b] = gaugeDay(r.date); return { a, b, v: r.p50, lo: r.p10 ?? r.p50, hi: r.p90 ?? r.p50 }; });

  const vals = obsLine.map((p) => p.v).concat(fcSteps.flatMap((p) => [p.lo, p.hi]));
  let ymin = Math.min(...vals), ymax = Math.max(...vals, st.action);
  const pad = Math.max(0.15, (ymax - ymin) * 0.06);
  ymin -= pad; ymax += pad;
  const t0 = start.getTime(), t1 = (fcSteps.length ? fcSteps[fcSteps.length - 1].b : now).getTime() + 3 * 3600e3;
  const x = (t) => m.l + (t.getTime() - t0) / (t1 - t0) * (W - m.l - m.r);
  const y = (v) => m.t + (ymax - v) / (ymax - ymin) * (H - m.t - m.b);

  // grid + y ticks
  const span = ymax - ymin, step = span > 6 ? 1 : span > 2.5 ? 0.5 : 0.25;
  for (let v = Math.ceil(ymin / step) * step; v <= ymax; v += step) {
    el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: css("--grid"), "stroke-width": 1 }, svg);
    el("text", { x: m.l - 8, y: y(v) + 4, "text-anchor": "end" }, svg, v.toFixed(step < 0.5 ? 2 : step < 1 ? 1 : 0));
  }
  el("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: css("--axis"), "stroke-width": 1 }, svg);
  // x ticks every 7 days counting back from today
  for (let k = 0; k <= 6; k++) {
    const t = new Date(now.getTime() - (28 - 7 * k) * 864e5);
    if (t.getTime() < t0 || t.getTime() > t1) continue;
    el("text", { x: x(t), y: H - m.b + 18, "text-anchor": "middle" }, svg, fmtDay(t, { month: "short", day: "numeric" }));
  }

  // flood stages in range
  for (const [k, name] of CATS.slice().reverse()) {
    const v = st[k];
    if (v > ymax) continue;
    el("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: css(`--cat-${k}`), "stroke-width": 1.5 }, svg);
    if (m.r > 60) el("text", { x: W - m.r + 8, y: y(v) + 4, class: "flood-label" }, svg, `${name.replace(" flood", "")} ${v} ft`);
    else el("text", { x: m.l + 4, y: y(v) - 5, class: "flood-label" }, svg, `${name.replace(" flood", "")} ${v} ft`);
  }

  // now divider
  el("line", { x1: x(now), x2: x(now), y1: m.t, y2: H - m.b, stroke: css("--axis"), "stroke-width": 1 }, svg);
  el("text", { x: x(now) + 5, y: H - m.b - 6 }, svg, "now");

  const path = (pts, f) => pts.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(f(p)).toFixed(1)}`).join("");
  const gap = 1.5; // px between neighbouring days, so each step reads as one day
  for (const s of fcSteps) {
    const xa = x(s.a) + gap, xb = x(s.b) - gap;
    el("rect", { x: xa, y: y(s.hi), width: Math.max(1, xb - xa), height: Math.max(1, y(s.lo) - y(s.hi)), fill: css("--fc-wash") }, svg);
    el("line", { x1: xa, x2: xb, y1: y(s.v), y2: y(s.v), stroke: css("--fc"), "stroke-width": 2.5, "stroke-linecap": "round" }, svg);
  }
  if (obsLine.length > 1) {
    el("path", { d: path(obsLine, (p) => p.v), fill: "none", stroke: css("--obs"), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    const last = obsLine[obsLine.length - 1];
    el("circle", { cx: x(last.t), cy: y(last.v), r: 4, fill: css("--obs"), stroke: css("--surface"), "stroke-width": 2 }, svg);
  }

  // hover / keyboard layer: snap to the nearest calendar day
  const days = new Map();
  for (const r of obs.daily || []) if (r.stage !== null && day(r.date) >= start) days.set(r.date, { obs: r.stage });
  for (const r of rows) days.set(r.date, Object.assign(days.get(r.date) || {}, { p50: r.p50, p10: r.p10, p90: r.p90 }));
  // the day in progress has no daily mean yet: show the latest reading and the average of today's readings so far
  for (const [k, d] of days) {
    const [a, b] = gaugeDay(k);
    if (d.obs !== undefined || now < a || now >= b) continue;
    const today = hourly.filter((h) => h.t >= a && h.t < b);
    if (today.length) {
      d.latest = today[today.length - 1];
      d.soFar = today.reduce((s, h) => s + h.v, 0) / today.length;
    }
  }
  const keys = [...days.keys()].sort();
  const cross = el("line", { y1: m.t, y2: H - m.b, stroke: css("--ink-2"), "stroke-width": 1, visibility: "hidden" }, svg);
  const tip = document.getElementById("tooltip");
  let idx = keys.length - 1;
  const show = (i, clientX, clientY) => {
    idx = Math.max(0, Math.min(keys.length - 1, i));
    const k = keys[idx], d = days.get(k);
    const cx = x(d.p50 !== undefined ? new Date(gaugeDay(k)[0].getTime() + 12 * 3600e3) : day(k));
    cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("visibility", "visible");
    tip.replaceChildren();
    el("div", { class: "tt-date" }, tip, fmtDay(day(k), { weekday: "short", month: "short", day: "numeric", year: "numeric" }));
    if (d.obs !== undefined) row(tip, "--obs", ft(d.obs), "observed daily mean");
    if (d.latest) {
      const at = d.latest.t.toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", timeZone: "America/New_York", timeZoneName: "short" });
      row(tip, "--obs", ft(d.latest.v), `latest reading, ${at}`);
      row(tip, "--obs", ft(d.soFar), "average so far today");
    }
    if (d.p50 !== undefined) { row(tip, "--fc", ft(d.p50), "expected daily average"); row(tip, null, `${ft(d.p10)}–${ft(d.p90)}`, "likely range"); }
    tip.hidden = false;
    const r = svg.getBoundingClientRect();
    const px = clientX ?? r.left + cx * (r.width / W), py = clientY ?? r.top + r.height * 0.3;
    const tw = tip.offsetWidth;
    tip.style.left = `${Math.min(window.innerWidth - tw - 8, Math.max(8, px + 14 + tw > window.innerWidth ? px - tw - 14 : px + 14))}px`;
    tip.style.top = `${Math.max(8, py - tip.offsetHeight - 12)}px`;
  };
  const hide = () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); };
  svg.addEventListener("pointermove", (e) => {
    const r = svg.getBoundingClientRect();
    const px = (e.clientX - r.left) * (W / r.width);
    let best = 0, bd = Infinity;
    keys.forEach((k, i) => { const dd = Math.abs(x(day(k)) - px); if (dd < bd) { bd = dd; best = i; } });
    show(best, e.clientX, e.clientY);
  });
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("blur", hide);
  svg.addEventListener("focus", () => show(idx));
  svg.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { show(idx - 1); e.preventDefault(); }
    if (e.key === "ArrowRight") { show(idx + 1); e.preventDefault(); }
    if (e.key === "Escape") hide();
  });
}

function row(parent, color, value, label) {
  const r = el("div", { class: "tt-row" }, parent);
  el("span", { class: "tt-key", style: color ? `background: var(${color})` : "background: var(--fc-wash); height: 8px" }, r);
  el("b", {}, r, value);
  el("span", {}, r, label);
}

/* ---------------- tables ---------------- */

function renderObsTable(obs) {
  const body = document.getElementById("obs-body");
  body.replaceChildren();
  for (const r of (obs.daily || []).slice(-30).reverse()) {
    const tr = el("tr", {}, body);
    el("td", {}, tr, fmtDay(day(r.date)));
    el("td", {}, tr, ft(r.stage));
    el("td", {}, tr, ft(r.toho));
  }
}

function renderRecord(sc) {
  const host = document.getElementById("record");
  host.replaceChildren();
  const rows = (sc && sc.by_h) || [];
  if (!rows.some((r) => r.n > 0)) {
    el("p", { class: "empty" }, host, "No forecasts have been scored yet. Scores appear here once each forecast's target day has been observed.");
    return;
  }
  const t = el("table", { class: "record-table" }, host);
  const hr = el("tr", {}, el("thead", {}, t));
  for (const h of ["Days ahead", "Scored", "Average miss", "Inside likely range", "NWM average miss"]) el("th", { scope: "col" }, hr, h);
  const body = el("tbody", {}, t);
  for (const r of rows) {
    const tr = el("tr", {}, body);
    el("th", { scope: "row" }, tr, String(r.h));
    el("td", {}, tr, String(r.n));
    el("td", {}, tr, r.mae === null ? "—" : `${r.mae.toFixed(2)} ft`);
    el("td", {}, tr, r.cov80 === null ? "—" : `${Math.round(r.cov80 * 100)}%`);
    el("td", {}, tr, r.nwm_mae === null ? "—" : `${r.nwm_mae.toFixed(2)} ft`);
  }
}

/* ---------------- boot ---------------- */

async function main() {
  let fc, obs, sc;
  try {
    [fc, obs, sc] = await Promise.all([getJSON("data/forecast.json"), getJSON("data/observed.json"), getJSON("data/scorecard.json").catch(() => null)]);
  } catch (err) {
    document.getElementById("headline").textContent = "The forecast data couldn't be loaded. Try again in a few minutes.";
    console.error(err);
    return;
  }
  const draw = () => { renderGauge(fc); renderHydro(fc, obs); };
  renderReading(fc, obs);
  renderWeek(fc);
  renderObsTable(obs);
  renderRecord(sc);
  draw();
  if (fc.generated_at) {
    const g = new Date(fc.generated_at).toLocaleString("en-US", { timeZone: TZ, month: "short", day: "numeric", hour: "numeric", minute: "2-digit", timeZoneName: "short" });
    document.getElementById("generated").textContent = `Updated ${g}. Forecast issued for ${fmtDay(day(fc.issue_day))}; model trained on data through ${fc.model && fc.model.train_end}.`;
  }
  let raf = 0;
  window.addEventListener("resize", () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(() => renderHydro(fc, obs)); });
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
}

main();
