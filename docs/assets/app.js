(function () {
  "use strict";
  const LANGS = ["en", "de", "es"];
  const LOCALE = { en: "en-GB", de: "de-DE", es: "es-ES" };
  const DATA = {};
  let lang = "en", map, base = {}, layers = {}, step = 1, bounds = {};

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
  function setLang(l) { lang = l; try { localStorage.setItem("otr-lang", l); } catch (e) {} applyI18n(); if (map) rebuildMarkers(); }

  /* ---------- steps ---------- */
  function renderStep() {
    $$("#stepTabs button").forEach((b) => b.setAttribute("aria-selected", String(+b.dataset.step === step)));
    $$("[data-step-img]").forEach((im) => im.classList.toggle("on", +im.dataset.stepImg === step));
    $("#stepNo").textContent = t("step_of", { i: step });
    $("#stepTitle").textContent = t("step" + step + "_t");
    $("#stepDesc").textContent = t("step" + step + "_d");
  }
  function goStep(n) { step = ((n - 1 + 4) % 4) + 1; renderStep(); }

  /* ---------- counters / dynamic text from stats.json ---------- */
  function renderDynamic() {
    const s = DATA.stats; if (!s) return;
    $$("[data-count]").forEach((el) => { el.dataset.target = s[el.dataset.count]; el.textContent = nf(s[el.dataset.count]); });
    $$("[data-n]").forEach((el) => { const k = el.dataset.n; el.textContent = nf(counts()[k] ?? 0); });
    $("#aiBig").textContent = pct(s.ai_high.share) + " %";
    $("#aiText").textContent = t("ai_text", { hi: pct(s.ai_high.share), hir: pct(s.ai_high_recent.share), lo: pct(s.ai_low.share, 1) });
    $("#safeBig").textContent = nf(s.spot_misses) + " / " + nf(s.spot_checks);
    $("#safeText").textContent = t("safe_text", { auto: nf(s.auto_clean + s.spot_checks), spot: nf(s.spot_checks), miss: nf(s.spot_misses) });
    $("#honestText").textContent = t("honest_text", { rate: pct(s.hit_rate, 1) });
    renderChart(); renderRegions();
  }
  function counts() {
    const c = { known: 0, ai: 0, human: 0, cand: 0, mines: DATA.stats ? DATA.stats.mines : 0 };
    DATA.known.features.forEach((f) => (c.known += f.properties.n));
    DATA.found.features.forEach((f) => (c[f.properties.how] += f.properties.n));
    DATA.cand.features.forEach((f) => (c.cand += f.properties.n));
    return c;
  }
  function animateCounters() {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    $$("[data-count]").forEach((el) => {
      const target = +el.dataset.target; if (!target) return; const t0 = performance.now();
      (function tick(now) { const k = Math.min(1, (now - t0) / 1200); el.textContent = nf(Math.round(target * (1 - Math.pow(1 - k, 3)))); if (k < 1) requestAnimationFrame(tick); })(t0);
    });
  }

  /* ---------- chart ---------- */
  function renderChart() {
    const box = $("#chart"); if (!box) return;
    const order = ["v4", "v5", "v6", "chile", "v7"], by = Object.fromEntries(DATA.stats.rounds.map((r) => [r.round, r]));
    const maxShare = Math.max(...DATA.stats.rounds.map((r) => r.confirmed / r.reviewed));
    box.innerHTML = order.filter((k) => by[k]).map((k) => {
      const r = by[k], share = r.confirmed / r.reviewed;
      return `<div class="bar"><span>${t("round_" + k)}</span><span class="track"><span class="fill" data-w="${Math.min(100, share * 100 * 4)}"></span></span><span class="v">${t("n_of", { c: r.confirmed, n: nf(r.reviewed) })}</span></div>`;
    }).join("");
    requestAnimationFrame(() => $$(".fill", box).forEach((f) => (f.style.width = f.dataset.w + "%")));
  }

  /* ---------- map ---------- */
  function icon(cls, n, extra) { return L.divIcon({ className: "", html: `<div class="mk ${cls}"><span>${n > 1 ? n : ""}</span></div>`, iconSize: extra || [18, 18] }); }
  function popup(kind, p) {
    let h = "";
    if (kind === "known") h = `<div class="pop-t">${t("pop_known", { n: p.n })}</div><div class="pop-s">${t("pop_size", { s: t("size_" + p.size) })}</div>`;
    else if (kind === "cand") h = `<div class="pop-t">${t("pop_cand", { n: p.n })}</div><div class="pop-s">${t("pop_score", { s: p.score })} · ${t("pop_size", { s: t("size_" + p.size) })}</div><div class="pop-s">${t("pop_cand_note")}</div>`;
    else h = `<div class="pop-t">${t("pop_found", { n: p.n })}</div><div class="pop-s">${t("pop_kind_" + p.kind)} · ${t("pop_size", { s: t("size_" + p.size) })}</div><div class="pop-s">${t("pop_how_" + p.how)}</div>`;
    return h;
  }
  function buildLayers() {
    layers.mines = L.geoJSON(DATA.mines, { style: { color: "#8b98a5", weight: 1.2, fillColor: "#8b98a5", fillOpacity: 0.12 }, interactive: false });
    layers.cand = L.layerGroup(); layers.known = L.layerGroup(); layers.ai = L.layerGroup(); layers.human = L.layerGroup();
    rebuildMarkers();
  }
  function rebuildMarkers() {
    ["cand", "known", "ai", "human"].forEach((k) => layers[k].clearLayers());
    DATA.cand.features.forEach((f) => { const [x, y] = f.geometry.coordinates; L.circleMarker([y, x], { radius: 4 + Math.min(4, f.properties.n), color: "#d8dee6", weight: 1.5, fillColor: "#d8dee6", fillOpacity: 0.15 }).bindPopup(popup("cand", f.properties)).addTo(layers.cand); });
    DATA.known.features.forEach((f) => { const [x, y] = f.geometry.coordinates; L.marker([y, x], { icon: icon("mk-known", f.properties.n, [16, 16]) }).bindPopup(popup("known", f.properties)).addTo(layers.known); });
    DATA.found.features.forEach((f) => { const [x, y] = f.geometry.coordinates, p = f.properties;
      L.marker([y, x], { icon: icon(p.how === "ai" ? "mk-ai" : "mk-human", p.n, p.how === "ai" ? [18, 18] : [16, 16]), zIndexOffset: 500 }).bindPopup(popup("found", p)).addTo(layers[p.how]); });
  }
  function initMap() {
    map = L.map("mapEl", { zoomControl: true, worldCopyJump: false, minZoom: 4 }); window.otrMap = map;
    base.sat = L.tileLayer("https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg", { maxZoom: 13, attribution: 'Sentinel-2 cloudless – <a href="https://s2maps.eu">s2maps.eu</a> by EOX IT Services GmbH (Contains modified Copernicus Sentinel data 2020)' });
    base.map = L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", { maxZoom: 18, subdomains: "abcd", attribution: '&copy; OpenStreetMap contributors &copy; CARTO' });
    base.sat.addTo(map);
    buildLayers(); ["mines", "cand", "known", "ai", "human"].forEach((k) => layers[k].addTo(map));
    const all = [...DATA.known.features, ...DATA.found.features, ...DATA.cand.features].map((f) => [f.geometry.coordinates[1], f.geometry.coordinates[0]]);
    map.fitBounds(L.latLngBounds(all).pad(0.08));
    map.on("load", () => {});
    $$("[data-layer]").forEach((cb) => cb.addEventListener("change", () => (cb.checked ? layers[cb.dataset.layer].addTo(map) : map.removeLayer(layers[cb.dataset.layer]))));
    $$("[data-base]").forEach((b) => b.addEventListener("click", () => {
      const k = b.dataset.base; Object.values(base).forEach((l) => map.removeLayer(l)); base[k].addTo(map);
      $$("[data-base]").forEach((x) => x.classList.toggle("on", x === b));
    }));
    ["known", "found", "cand"].forEach(() => {});
    const regionPts = {};
    [DATA.known, DATA.found, DATA.cand].forEach((fc) => fc.features.forEach((f) => { (regionPts[f.properties.region] = regionPts[f.properties.region] || []).push([f.geometry.coordinates[1], f.geometry.coordinates[0]]); }));
    Object.keys(regionPts).forEach((r) => (bounds[r] = L.latLngBounds(regionPts[r]).pad(0.1)));
    const home = () => { map.invalidateSize(); if (bounds.atacama) map.fitBounds(bounds.atacama, { maxZoom: 9, animate: false }); };
    home(); setTimeout(home, 300); window.addEventListener("load", home);
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
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.08 });
    els.forEach((e) => io.observe(e));
  }
  async function load() {
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
    try {
      await load(); initMap(); applyI18n(); animateCounters();
    } catch (e) {
      const m = $("#mapEl"); if (m) m.textContent = "Data could not be loaded (" + e.message + "). Serve the page over http(s), not file://.";
    }
    initReveal();
  });
})();
