(function () {
  "use strict";
  const LANGS = ["en", "de", "es"];
  const LOCALE = { en: "en-GB", de: "de-DE", es: "es-ES" };
  const DATA = {};
  let lang = "en", map, base = {}, step = 1, view = "found", bounds = {}, minesLayer, markers;

  try { lang = localStorage.getItem("otr-lang") || ""; } catch (e) {}
  if (!LANGS.includes(lang)) lang = LANGS.includes((navigator.language || "en").slice(0, 2)) ? (navigator.language || "en").slice(0, 2) : "en";

  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const nf = (n) => new Intl.NumberFormat(LOCALE[lang]).format(n);
  const pct = (x, d = 0) => (x * 100).toLocaleString(LOCALE[lang], { maximumFractionDigits: d });
  function t(key, vars) {
    let s = (window.I18N[lang] && window.I18N[lang][key]) || window.I18N.en[key] || key;
    if (vars) for (const k in vars) s = s.split("{" + k + "}").join(vars[k]);
    return s;
  }

  /* ---------- translations ---------- */
  function applyI18n() {
    document.documentElement.lang = lang;
    $$("[data-i18n]").forEach((el) => (el.textContent = t(el.dataset.i18n)));
    $$("[data-i18n-aria]").forEach((el) => el.setAttribute("aria-label", t(el.dataset.i18nAria)));
    $$(".lang button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.lang === lang)));
    renderStep(); renderDynamic();
  }
  function setLang(l) { lang = l; try { localStorage.setItem("otr-lang", l); } catch (e) {} applyI18n(); if (map) renderMarkers(); }

  /* ---------- steps ---------- */
  function renderStep() {
    $$("#stepTabs button").forEach((b) => b.setAttribute("aria-selected", String(+b.dataset.step === step)));
    $$("[data-step-img]").forEach((im) => im.classList.toggle("on", +im.dataset.stepImg === step));
    $("#stepNo").textContent = t("step_of", { i: step });
    $("#stepTitle").textContent = t("step" + step + "_t");
    $("#stepDesc").textContent = t("step" + step + "_d");
  }
  function goStep(n) { step = ((n - 1 + 4) % 4) + 1; renderStep(); }

  /* ---------- numbers and charts ---------- */
  function counts() {
    const c = { known: 0, found: 0, ai: 0, human: 0, cand: 0, mines: DATA.stats ? DATA.stats.mines : 0 };
    DATA.known.features.forEach((f) => (c.known += f.properties.n));
    DATA.found.features.forEach((f) => { c.found += f.properties.n; c[f.properties.how] += f.properties.n; });
    DATA.cand.features.forEach((f) => (c.cand += f.properties.n));
    return c;
  }
  function renderDynamic() {
    const s = DATA.stats; if (!s) return;
    const c = counts();
    $$("[data-count]").forEach((el) => { el.dataset.target = s[el.dataset.count]; el.textContent = nf(s[el.dataset.count]); });
    $$("[data-n]").forEach((el) => (el.textContent = nf(c[el.dataset.n] ?? 0)));
    $("#foundSplit").textContent = t("found_split", { ai: nf(c.ai), human: nf(c.human) });
    const pn = $(".places"); if (pn) pn.textContent = t("places_note", { k: s.known_items });
    $("#heroGain").innerHTML = t("hero_gain", { added: "<b>" + nf(s.found_items) + "</b>", pct: "<b>+" + pct(s.added_share) + "%</b>", known: nf(s.known_items), n: nf(s.reviewed), open: nf(s.candidates) });
    const last = s.growth[s.growth.length - 1];
    $("#growthBig").textContent = nf(s.known_items) + " → " + nf(last.total);
    $("#growthText").textContent = t("growth_text", { a: nf(s.known_items), b: nf(last.total), open: nf(s.candidates) });
    const g = s.gain.points, at = (p) => g.find((q) => q.p === p);
    $("#gainText").textContent = t("gain_text", { x: pct(at(0.7).x, 0), y: pct(at(0.7).y, 0), x2: pct(at(0.2).x, 0), y2: pct(at(0.2).y, 0) });
    $("#aiBig").textContent = pct(s.ai_high.share) + " %";
    $("#aiText").textContent = t("ai_text", { hi: pct(s.ai_high.share), hir: pct(s.ai_high_recent.share), lo: pct(s.ai_low.share, 1) });
    $("#safeBig").textContent = nf(s.spot_misses) + " / " + nf(s.spot_checks);
    $("#safeText").textContent = t("safe_text", { auto: nf(s.auto_clean + s.spot_checks), spot: nf(s.spot_checks), miss: nf(s.spot_misses) });
    $("#honestText").textContent = t("honest_text", { rate: pct(s.hit_rate, 1) });
    renderGrowth(); renderGain(); renderAiBars(); renderChart(); renderRegions(); legendState();
  }
  function roundNo(r) { return parseInt(r.replace(/\D/g, ""), 10); }

  function renderGrowth() {
    const s = DATA.stats, el = $("#growthChart"); if (!el) return;
    const pts = [{ label: "0", total: s.known_items }].concat(s.growth.map((g) => ({ label: String(roundNo(g.round)), total: g.total })));
    const narrow = el.clientWidth && el.clientWidth < 560, W = narrow ? 400 : 760, H = narrow ? 250 : 230, L = 40, R = 18, T = 20, B = 34, max = Math.ceil(Math.max(...pts.map((p) => p.total)) / 20) * 20;
    const x = (i) => L + (i * (W - L - R)) / (pts.length - 1), y = (v) => T + (1 - v / max) * (H - T - B);
    const line = pts.map((p, i) => `${x(i)},${y(p.total)}`).join(" ");
    const area = `${x(0)},${y(0)} ${line} ${x(pts.length - 1)},${y(0)}`;
    const grid = [0, 0.25, 0.5, 0.75, 1].map((f) => `<line class="gridline" x1="${L}" x2="${W - R}" y1="${y(max * f)}" y2="${y(max * f)}"/><text x="${L - 8}" y="${y(max * f) + 4}" text-anchor="end">${Math.round(max * f)}</text>`).join("");
    const base0 = `<line x1="${L}" x2="${W - R}" y1="${y(pts[0].total)}" y2="${y(pts[0].total)}" stroke="#5aa9ff" stroke-width="1.5" stroke-dasharray="5 4"/>`;
    const dots = pts.map((p, i) => `<circle cx="${x(i)}" cy="${y(p.total)}" r="${i === 0 || i === pts.length - 1 ? 5 : 3.5}" fill="${i === 0 ? "#5aa9ff" : "#3ddc97"}" stroke="#0e1116" stroke-width="2"><title>${p.total}</title></circle>`).join("");
    const labels = pts.map((p, i) => `<text x="${x(i)}" y="${H - 12}" text-anchor="middle">${i === 0 ? "" : p.label}</text>`).join("");
    const ends = `<text x="${x(0) + 8}" y="${y(pts[0].total) - 10}" fill="#5aa9ff" font-weight="700" style="fill:#5aa9ff;font-size:13px">${nf(pts[0].total)}</text><text x="${x(pts.length - 1) - 8}" y="${y(pts[pts.length - 1].total) - 12}" text-anchor="end" style="fill:#3ddc97;font-size:14px;font-weight:700">${nf(pts[pts.length - 1].total)}</text>`;
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="axis" aria-hidden="true"><defs><linearGradient id="gg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#3ddc97" stop-opacity=".35"/><stop offset="1" stop-color="#3ddc97" stop-opacity="0"/></linearGradient></defs>${grid}<polygon points="${area}" fill="url(#gg)"/>${base0}<polyline points="${line}" fill="none" stroke="#3ddc97" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>${dots}${labels}${ends}</svg>`;
  }

  function renderGain() {
    const s = DATA.stats, el = $("#gainChart"); if (!el) return;
    const W = 420, H = 280, L = 58, R = 14, T = 14, B = 44, x = (v) => L + v * (W - L - R), y = (v) => T + (1 - v) * (H - T - B);
    const pts = s.gain.points.slice().sort((a, b) => a.x - b.x);
    const curve = pts.map((p) => `${x(p.x)},${y(p.y)}`).join(" ");
    const grid = [0, 0.25, 0.5, 0.75, 1].map((f) => `<line class="gridline" x1="${L}" x2="${W - R}" y1="${y(f)}" y2="${y(f)}"/><text x="${L - 8}" y="${y(f) + 4}" text-anchor="end">${Math.round(f * 100)}%</text><text x="${x(f)}" y="${H - B + 18}" text-anchor="middle">${Math.round(f * 100)}%</text>`).join("");
    const mark = pts.find((p) => p.p === 0.7), mark2 = pts.find((p) => p.p === 0.2);
    const callout = (p, dx, dy, txt) => `<circle cx="${x(p.x)}" cy="${y(p.y)}" r="5" fill="#f2a541" stroke="#0e1116" stroke-width="2"/><text x="${x(p.x) + dx}" y="${y(p.y) + dy}" style="fill:#f2a541;font-size:12px;font-weight:700">${txt}</text>`;
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="axis" aria-hidden="true">${grid}
      <line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="#9aa6b2" stroke-dasharray="5 5"/>
      <text x="${x(0.62)}" y="${y(0.62) + 18}" style="fill:#9aa6b2;font-size:11px" transform="rotate(-31 ${x(0.62)} ${y(0.62) + 18})">${t("gain_rand")}</text>
      <polyline points="${curve}" fill="none" stroke="#3ddc97" stroke-width="3" stroke-linejoin="round"/>
      ${callout(mark, 10, 18, pct(mark.x, 0) + "% → " + pct(mark.y, 0) + "%")}${callout(mark2, 10, 18, pct(mark2.x, 0) + "% → " + pct(mark2.y, 0) + "%")}
      <text x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">${t("gain_x")}</text>
      <text x="11" y="${(T + H - B) / 2}" text-anchor="middle" transform="rotate(-90 11 ${(T + H - B) / 2})">${t("gain_y")}</text></svg>`;
  }

  function renderAiBars() {
    const el = $("#aiBars"); if (!el) return;
    const rows = DATA.stats.ai_rounds.filter((r) => r.flagged >= 5);
    el.innerHTML = rows.map((r) => {
      const share = r.flagged_real / r.flagged;
      return `<div class="aibar"><span>${t("round_n", { i: roundNo(r.round) })}</span><span class="track"><span class="fill" data-w="${share * 100}"></span></span><span class="v">${pct(share)}%<small>${r.flagged_real}/${r.flagged}</small></span></div>`;
    }).join("");
    requestAnimationFrame(() => $$(".fill", el).forEach((f) => (f.style.width = f.dataset.w + "%")));
  }

  function renderChart() {
    const box = $("#chart"); if (!box) return;
    const rs = DATA.stats.rounds, done = rs.filter((r) => !r.open);
    const maxShare = Math.max(...done.map((r) => r.confirmed / r.checked));
    box.innerHTML = rs.map((r, i) => {
      const name = `${t("round_n", { i: i + 1 })} · ${t("tag_" + r.tag)}`;
      if (r.open) return `<div class="bar open"><span>${name}</span><span class="track"></span><span class="v">${t("chart_open", { n: nf(r.listed) })}</span></div>`;
      const share = r.confirmed / r.checked, auto = r.auto ? `<small title="${t("chart_auto", { a: nf(r.auto) })}">+${nf(r.auto)} AI</small>` : "";
      return `<div class="bar"><span>${name}</span><span class="track"><span class="fill" data-w="${Math.min(100, (share / maxShare) * 88)}"></span></span><span class="v">${t("n_of", { c: r.confirmed, n: nf(r.checked) })}${auto}</span></div>`;
    }).join("");
    requestAnimationFrame(() => $$(".fill", box).forEach((f) => (f.style.width = f.dataset.w + "%")));
  }

  /* ---------- map: fewer, clearer marks ---------- */
  const CELL = 0.01;
  function gridSize(zoom) { return CELL * Math.pow(2, Math.max(0, 10 - zoom)); }   // zoomed out: bigger cells, one bubble each
  function aggregate(features, g) {
    const m = new Map();
    features.forEach((f) => {
      const [lon, lat] = f.geometry.coordinates, n = f.properties.n, k = Math.floor(lon / g) + ":" + Math.floor(lat / g);
      const a = m.get(k) || { n: 0, lon: 0, lat: 0, ai: 0, human: 0, props: f.properties };
      a.n += n; a.lon += lon * n; a.lat += lat * n;
      if (f.properties.how) a[f.properties.how] += n;
      m.set(k, a);
    });
    return [...m.values()].map((a) => ({ ...a, lon: a.lon / a.n, lat: a.lat / a.n }));
  }
  function bubble(cat, a, single) {
    const size = single ? (cat === "cand" ? 14 : 20) : Math.min(46, 20 + 6 * Math.sqrt(a.n));
    const txt = single && a.n === 1 ? "" : nf(a.n);
    return L.divIcon({ className: "", html: `<div class="bub ${cat}" style="width:${size}px;height:${size}px">${txt}</div>`, iconSize: [size, size] });
  }
  function popupHtml(cat, a) {
    if (cat === "known") return `<div class="pop-t">${t("pop_known", { n: a.n })}</div>`;
    if (cat === "cand") return `<div class="pop-t">${t("pop_cand", { n: a.n })}</div><div class="pop-s">${t("pop_score", { s: a.props.score })} · ${t("pop_size", { s: t("size_" + a.props.size) })}</div><div class="pop-s">${t("pop_cand_note")}</div>`;
    const how = a.ai && !a.human ? "pop_how_ai" : !a.ai && a.human ? "pop_how_human" : null;
    return `<div class="pop-t">${t("pop_found_n", { n: a.n })}</div>` + (how ? `<div class="pop-s">${t(how)}</div>` : `<div class="pop-s">${t("pop_how_ai")}: ${a.ai} · ${t("pop_how_human")}: ${a.human}</div>`);
  }
  function renderMarkers() {
    if (!map) return;
    markers.clearLayers();
    const z = map.getZoom(), g = gridSize(z), single = g <= CELL + 1e-9;
    const cats = view === "start" ? ["known"] : view === "found" ? ["known", "found"] : ["cand", "known", "found"];
    const src = { known: DATA.known.features, found: DATA.found.features, cand: DATA.cand.features };
    cats.forEach((cat) => {
      aggregate(src[cat], g).forEach((a) => {
        const mk = L.marker([a.lat, a.lon], { icon: bubble(cat, a, single), zIndexOffset: cat === "cand" ? 0 : cat === "known" ? 400 : 800 });
        if (single) mk.bindPopup(popupHtml(cat, a));
        else mk.on("click", () => map.setView([a.lat, a.lon], Math.min(12, z + 2)));
        if (!single) mk.bindTooltip(`${nf(a.n)} · ${t("pop_zoom")}`);
        mk.addTo(markers);
      });
    });
    if (minesLayer) { const on = $("#minesToggle").checked && z >= 9; on ? minesLayer.addTo(map) : map.removeLayer(minesLayer); }
  }
  function legendState() {
    const on = view === "start" ? ["known"] : view === "found" ? ["known", "found"] : ["known", "found", "cand"];
    $$(".legend li").forEach((li) => li.classList.toggle("off", !on.includes(li.dataset.cat)));
    $$("#viewSeg button").forEach((b) => b.classList.toggle("on", b.dataset.view === view));
  }
  function setView(v) { view = v; legendState(); renderMarkers(); }
  function initMap() {
    map = L.map("mapEl", { zoomControl: true, minZoom: 4 }); window.otrMap = map;
    base.sat = L.tileLayer("https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg", { maxZoom: 13, attribution: 'Sentinel-2 cloudless – <a href="https://s2maps.eu">s2maps.eu</a> by EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2020)' });
    base.map = L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", { maxZoom: 18, subdomains: "abcd", attribution: '&copy; OpenStreetMap contributors &copy; CARTO' });
    base.sat.addTo(map);
    markers = L.layerGroup().addTo(map);
    minesLayer = L.geoJSON(DATA.mines, { style: { color: "#f2a541", weight: 1.4, fillColor: "#f2a541", fillOpacity: 0.08 }, interactive: false });
    const regionPts = {};
    [DATA.known, DATA.found, DATA.cand].forEach((fc) => fc.features.forEach((f) => { (regionPts[f.properties.region] = regionPts[f.properties.region] || []).push([f.geometry.coordinates[1], f.geometry.coordinates[0]]); }));
    Object.keys(regionPts).forEach((r) => (bounds[r] = L.latLngBounds(regionPts[r]).pad(0.1)));
    const home = () => { map.invalidateSize(); map.fitBounds(bounds.atacama || bounds[Object.keys(bounds)[0]], { maxZoom: 9, animate: false }); renderMarkers(); };
    home(); setTimeout(home, 300); window.addEventListener("load", home);
    map.on("zoomend", renderMarkers);
    let rz; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => { renderGrowth(); renderGain(); }, 200); });
    $$("#viewSeg button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
    $("#minesToggle").addEventListener("change", renderMarkers);
    $$("[data-base]").forEach((b) => b.addEventListener("click", () => {
      const k = b.dataset.base; Object.values(base).forEach((l) => map.removeLayer(l)); base[k].addTo(map);
      $$("[data-base]").forEach((x) => x.classList.toggle("on", x === b));
    }));
  }
  function renderRegions() {
    const box = $("#regions"); if (!box || !DATA.stats) return;
    box.innerHTML = Object.entries(DATA.stats.regions).map(([k, r]) => `<div class="region"><h4>${t("region_" + k)}</h4><dl>
      <dt>${t("reg_known")}</dt><dd>${nf(r.known)}</dd><dt>${t("reg_found")}</dt><dd>${nf(r.found)}</dd>
      <dt>${t("reg_reviewed")}</dt><dd>${nf(r.reviewed)}</dd><dt>${t("reg_cand")}</dt><dd>${nf(r.candidates)}</dd></dl>
      <button class="btn ghost" type="button" data-region="${k}">${t("reg_show")}</button></div>`).join("");
    $$("[data-region]", box).forEach((b) => b.addEventListener("click", () => { const bb = bounds[b.dataset.region]; if (bb && map) { map.fitBounds(bb, { maxZoom: 10 }); $("#mapEl").scrollIntoView({ behavior: "smooth", block: "center" }); } }));
  }

  /* ---------- misc ---------- */
  function initForm() {
    const f = $("#contactForm"), msg = $("#formMsg");
    f.addEventListener("submit", (e) => {
      e.preventDefault();
      const ok = f.name.value.trim() && /^\S+@\S+\.\S+$/.test(f.email.value) && f.msg.value.trim();
      msg.className = ok ? "" : "bad"; msg.textContent = ok ? t("f_thanks") : t("f_bad"); if (ok) f.reset();
    });
  }
  function initReveal() {
    const els = $$(".band .wrap > *, .card, .region"); els.forEach((e) => e.classList.add("reveal"));
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.05 });
    els.forEach((e) => io.observe(e));
  }
  function animateCounters() {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    $$("[data-count]").forEach((el) => {
      const target = +el.dataset.target; if (!target) return; const t0 = performance.now();
      (function tick(now) { const k = Math.min(1, (now - t0) / 1200); el.textContent = nf(Math.round(target * (1 - Math.pow(1 - k, 3)))); if (k < 1) requestAnimationFrame(tick); })(t0);
    });
  }
  async function load() {
    if (window.OTR_DATA) { Object.assign(DATA, window.OTR_DATA); return; }          // data/data.js (also works from file://)
    const get = (f) => fetch("data/" + f).then((r) => { if (!r.ok) throw new Error(f); return r.json(); });
    const [stats, found, known, cand, mines] = await Promise.all(["stats.json", "found.json", "known.json", "candidates.json", "mines.json"].map(get));
    Object.assign(DATA, { stats, found, known, cand, mines });
  }

  document.addEventListener("DOMContentLoaded", async () => {
    $$(".lang button").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.lang)));
    $$("#stepTabs button").forEach((b) => b.addEventListener("click", () => goStep(+b.dataset.step)));
    $("#prevStep").addEventListener("click", () => goStep(step - 1)); $("#nextStep").addEventListener("click", () => goStep(step + 1));
    initForm();
    applyI18n();
    try { await load(); initMap(); applyI18n(); animateCounters(); }
    catch (e) { const m = $("#mapEl"); if (m) m.textContent = "Data could not be loaded (" + e.message + ")."; console.error(e); }
    initReveal();
  });
})();
