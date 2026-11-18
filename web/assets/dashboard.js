/* Smart Cycle dashboard: every number and point comes from the API (same origin).
   Public view = raw reports only; Partner city view = severity, segments, alerts, priorities
   (header X-Partner-Key, a DEMO-ONLY control). Severity is an uncalibrated heuristic.
   URL parameters (for demos / screenshots): view=partner, select=<event id>, dialog=model,
   theme=light|dark, replay=<ride id>, replay_ms=<duration>, selftest=1. */
"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  const TYPE_LABEL = { Pothole: "Pothole / bump", Irregularity: "Irregularity",
                       Rough: "Rough stretch", Smooth: "Smooth" };
  const REASONS = [["glass", "Broken glass"], ["parked_vehicle", "Parked vehicle"],
                   ["drain_cover", "Drain cover"], ["pothole_avoided", "Pothole avoided"],
                   ["other", "Other"]];
  const PROV_TEXT = { real: "Real data", synthetic: "Synthetic demo data",
                      mixed: "Mixed real + synthetic", unknown: "Unknown provenance",
                      none: "No rides yet" };
  const DEFAULT_KEY = "demo-partner";
  const REPLAY_NOTE = "These rides are training data, replayed for demonstration, not an evaluation.";
  const PRIVACY = "Privacy: never photograph people's faces or vehicle number plates. " +
                  "Photo metadata (EXIF, GPS) is removed on upload.";
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

  const S = {
    view: "public", key: null, health: null, model: null, events: [], byId: new Map(),
    markers: new Map(), sel: null, rides: [], segs: null, alerts: null, prios: null,
    segGeom: new Map(), fitted: false, replaying: false, prov: {},
  };

  // ------------------------------------------------------------------ helpers
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const typeLabel = (t) => TYPE_LABEL[t] || t;
  const sevBand = (s) => (s >= 4 ? "hi" : s >= 3 ? "mid" : "lo");
  const fmtDate = (iso) => {
    const d = new Date(iso);
    return isNaN(d) ? "unknown date" : d.toLocaleString("en-IE", {
      day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
  };
  // "recorded" = time of the ride (epoch timestamps or Phyphox meta/time.csv); otherwise the
  // only known date is the upload time, and the UI says so.
  const dateWord = (p) => (p.date_kind === "recorded" ? "Recorded" : "Uploaded");
  const fmtWhen = (p) => `${dateWord(p)} ${fmtDate(p.date)}`;
  const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
  const f3 = (x) => (typeof x === "number" ? x.toFixed(3) : "–");
  const segName = (s) => (!s.name || /^Unnamed/.test(s.name)
    ? `Unnamed ${s.highway || (s.is_cycleway ? "cycleway" : "way")}` : s.name);

  async function api(path, opts = {}) {
    const headers = Object.assign({}, opts.headers || {});
    if (opts.partner && S.key) headers["X-Partner-Key"] = S.key;
    let r;
    try {
      r = await fetch(path, { method: opts.method || "GET", body: opts.body, headers });
    } catch (e) {
      const err = new Error("The API is not reachable."); err.status = 0; throw err;
    }
    let body = null;
    try { body = await r.json(); } catch (e) { /* non-JSON body */ }
    if (!r.ok) {
      const msg = (body && (body.error || body.detail)) || `HTTP ${r.status}`;
      const err = new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
      err.status = r.status; throw err;
    }
    return body;
  }

  function banner(html, isError) {
    const b = $("banner");
    b.innerHTML = html; b.hidden = !html; b.classList.toggle("err", !!isError);
  }

  // ------------------------------------------------------------------ theme
  function setTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    try { localStorage.setItem("sc-theme", t); } catch (e) { /* storage blocked */ }
  }
  function currentTheme() {
    const t = document.documentElement.getAttribute("data-theme");
    if (t) return t;
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  {
    let t = params.get("theme");
    if (!t) { try { t = localStorage.getItem("sc-theme"); } catch (e) { t = null; } }
    if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
  }
  $("themeBtn").addEventListener("click", () => setTheme(currentTheme() === "dark" ? "light" : "dark"));

  // ------------------------------------------------------------------ map
  const map = L.map("map", { zoomControl: true }).setView([53.3498, -6.2603], 13);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);
  const layers = {};
  for (const k of ["net", "rough", "seg", "track", "ev", "ride"]) layers[k] = L.layerGroup().addTo(map);

  function legend() {
    const par = S.view === "partner";
    $("legend").innerHTML = par
      ? `<span><i class="bar" style="background:var(--lo)"></i>Good</span>
         <span><i class="bar" style="background:var(--mid)"></i>Needs attention</span>
         <span><i class="bar" style="background:var(--hi)"></i>Repair soon</span>
         <span><i class="bar" style="background:var(--rough);height:10px"></i>Rough stretch</span>
         <span>Dot size and colour = severity (uncalibrated heuristic)</span>`
      : `<span><i style="background:var(--dot)"></i>Reported point (raw report)</span>
         <span><i class="bar" style="background:var(--green)"></i>Cycleway (OSM)</span>
         <span><i class="bar" style="background:var(--net)"></i>Other path / road (OSM)</span>`;
  }

  function fitOnce() {
    if (S.fitted) return;
    const pts = S.events.map((f) => [f.properties.lat, f.properties.lon]);
    if (pts.length > 1) map.fitBounds(L.latLngBounds(pts).pad(0.25), { maxZoom: 18 });
    else if (pts.length === 1) map.setView(pts[0], 17);
    else if (S.netBounds) map.fitBounds(S.netBounds);
    else return;
    S.fitted = true;
  }

  // ------------------------------------------------------------------ data loading
  async function boot() {
    banner("");
    try {
      S.health = await api("/api/health");
    } catch (e) {
      banner(`The API is not reachable. Start it with <code>python -m smartcycle serve</code>, then
              <button class="btn sm" type="button" id="retryBtn">Retry</button>`, true);
      $("retryBtn").addEventListener("click", boot);
      $("modelBadge").textContent = "API down";
      return false;
    }
    const h = S.health;
    if (!h.model_loaded) {
      banner(`No trained model: uploads are disabled. ${esc(h.hint || "")}`);
    }
    $("netNote").textContent = h.osm_loaded
      ? (h.network.synthetic ? "Network: SYNTHETIC routes." : `Network: OSM (${h.network.n_segments} segments).`)
      : "No OSM network loaded: points are shown at raw GPS.";
    await Promise.all([loadModel(), loadNetwork(), loadRides()]);
    await refresh();
    return true;
  }

  async function loadModel() {
    const badge = $("modelBadge");
    try {
      S.model = await api("/api/model");
      const c = S.model.model_card;
      badge.innerHTML = `Model: ${esc((c.model || "?").toUpperCase())}
        <span class="pv pv-${esc(c.data_provenance)}">${esc(c.data_provenance)}</span>`;
    } catch (e) {
      S.model = null;
      badge.innerHTML = `No model <span class="pv pv-none">none</span>`;
    }
    replayWarn();
  }

  async function loadNetwork() {
    layers.net.clearLayers();
    if (!S.health.osm_loaded) return;
    try {
      const gj = await api("/api/network");
      L.geoJSON(gj, {
        interactive: false,
        style: (f) => ({ className: f.properties.is_cycleway ? "net cyc" : "net", weight: 2 }),
      }).addTo(layers.net);
      const b = gj.bbox;
      if (b) S.netBounds = L.latLngBounds([[b[1], b[0]], [b[3], b[2]]]);
    } catch (e) { /* network optional */ }
  }

  async function loadRides() {
    try {
      S.rides = (await api("/api/rides")).rides;
    } catch (e) { S.rides = []; }
    const sel = $("rideSel");
    const cur = sel.value;
    sel.innerHTML = S.rides.length
      ? S.rides.map((r) => `<option value="${esc(r.id)}">${esc(String(r.source || r.id)
          .replace(/^(cli|csv):/, ""))} · ${esc(r.provenance)} · ${esc(r.mount)} · ${Math.round(r.duration_s || 0)} s</option>`).join("")
      : `<option value="">(no ride stored)</option>`;
    if (cur && S.rides.some((r) => r.id === cur)) sel.value = cur;
    $("replayBtn").disabled = !S.rides.length;
    replayWarn();
  }

  function replayWarn() {
    const w = $("replayWarn");
    const prov = S.model && S.model.model_card.data_provenance;
    w.textContent = (S.model && S.model.notes && S.model.notes.replay_note) || REPLAY_NOTE;
    // the demo rides of the real DB are the training rides of the real model
    const ride = S.rides.find((r) => r.id === $("rideSel").value);
    w.hidden = !(ride && ride.provenance === "real" && prov === "real");
  }
  $("rideSel").addEventListener("change", replayWarn);

  async function refresh() {
    const par = S.view === "partner";
    if (par && !S.key) { showKeyBox(); return; }
    let ev;
    try {
      ev = await api(par ? "/api/events?view=partner" : "/api/events", { partner: par });
    } catch (e) {
      if (e.status === 401) { S.key = null; showKeyBox("Wrong partner key."); return; }
      banner(`Could not load events: ${esc(e.message)}`, true);
      return;
    }
    S.events = ev.features;
    S.byId = new Map(S.events.map((f) => [f.properties.id, f.properties]));
    S.prov = { model: ev.model_provenance, data: ev.data_provenance,
               values: ev.data_provenance_values || [ev.data_provenance] };
    S.publicNote = ev.note;
    if (par) {
      try {
        [S.segs, S.alerts, S.prios] = await Promise.all([
          api("/api/partner/segments", { partner: true }),
          api("/api/partner/alerts", { partner: true }),
          api("/api/partner/priorities", { partner: true })]);
      } catch (e) {
        if (e.status === 401) { S.key = null; showKeyBox("Wrong partner key."); return; }
        banner(`Could not load partner analytics: ${esc(e.message)}`, true);
      }
    }
    if (S.sel && !S.byId.has(S.sel)) S.sel = null;
    provenance();
    renderLayers();
    renderLists();
    renderSelected();
    fitOnce();
  }

  function provenance() {
    const m = S.prov.model, d = S.prov.data, vals = S.prov.values;
    const synthModel = m === "synthetic" || m === "mixed";
    const synthData = vals.includes("synthetic");
    const allSynth = m === "synthetic" && (d === "synthetic" || d === "none");
    const sb = $("synthBanner");
    sb.hidden = !(synthModel || synthData);
    sb.textContent = allSynth ? "SYNTHETIC DEMO DATA"
      : `SYNTHETIC DEMO DATA: ${[synthModel && `model ${m}`, synthData && (d === "synthetic" ? "rides are synthetic" : "some rides are synthetic")]
          .filter(Boolean).join(", ")}`;
    const pf = $("provFooter");
    const dText = d === "mixed" ? `Mixed (${vals.join(" + ")})` : (PROV_TEXT[d] || d);
    pf.innerHTML = `Data: <span class="pv pv-${esc(synthData ? "synthetic" : d === "mixed" ? "unknown" : d)}">${esc(dText)}</span>
      Model: <span class="pv pv-${esc(m || "none")}">${esc(m || "none")}</span>`;
  }

  // ------------------------------------------------------------------ rendering
  function renderLayers() {
    const par = S.view === "partner";
    layers.ev.clearLayers(); layers.seg.clearLayers(); layers.rough.clearLayers();
    S.markers.clear(); S.segGeom.clear();
    if (par && S.segs) {
      for (const f of S.segs.features) {
        S.segGeom.set(f.properties.segment_id, f);
        if (f.properties.rough_m > 0) {
          L.geoJSON(f, { interactive: false, style: { className: "roughl", weight: 12 } }).addTo(layers.rough);
        }
        L.geoJSON(f, { interactive: false,
          style: { className: `segb ${f.properties.band}`, weight: 7 } }).addTo(layers.seg);
      }
    }
    // biggest first so small dots stay clickable on top
    const feats = [...S.events].sort((a, b) => (b.properties.severity || 0) - (a.properties.severity || 0));
    for (const f of feats) {
      const p = f.properties;
      const cls = ["ev"];
      if (par) { cls.push(sevBand(p.severity || 1)); if (p.matched === false) cls.push("unmatched"); }
      if (S.sel === p.id) cls.push("sel");
      const m = L.circleMarker([p.lat, p.lon], {
        radius: par ? 4 + 1.6 * (p.severity || 1) : 6, className: cls.join(" "),
        bubblingMouseEvents: false, weight: 1.5 });
      m.on("click", () => select(p.id, false));
      m.addTo(layers.ev);
      const el = m.getElement();
      if (el) {
        el.setAttribute("tabindex", "0");
        el.setAttribute("role", "button");
        el.setAttribute("aria-label", `${typeLabel(p.type)}, ${p.id}` +
          (par ? `, severity ${p.severity} of 5` : "") + `, ${plural(p.reports, "report")}`);
        el.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(p.id, false); }
        });
      }
      S.markers.set(p.id, m);
    }
    legend();
  }

  function item(p, sub) {
    const where = p.way_name && p.way_name !== "Unnamed road" ? ` on ${esc(p.way_name)}` : "";
    return `<li><button type="button" data-id="${esc(p.id)}">${esc(typeLabel(p.type))}${where} (${esc(p.id)})
      <span>${sub}</span></button></li>`;
  }

  function renderLists() {
    const box = $("lists");
    const par = S.view === "partner";
    let h = "";
    if (!S.events.length) {
      h += `<h3>${par ? "Alerts" : "Latest reports"}</h3><ul><li class="empty">No reports yet:
            upload a ride CSV below${S.health && !S.health.model_loaded ? " (needs a trained model)" : ""}.</li></ul>`;
      box.innerHTML = h; return;
    }
    if (par) {
      const al = (S.alerts && S.alerts.alerts) || [];
      h += `<h3>Alerts</h3><ul>${al.length ? al.map((a) => item(a,
        `Severity ${a.severity}/5 · ${esc(fmtWhen(a))} · ${plural(a.reports, "report")}`)).join("")
        : `<li class="empty">No new alerts.</li>`}</ul>`;
      if (S.alerts) {
        h += `<p class="note">Severity ≥ ${S.alerts.min_severity} in the last ${S.alerts.window_days} days.
              ${S.alerts.note ? esc(S.alerts.note) : ""}</p>`;
      }
      const pr = (S.prios && S.prios.priorities) || [];
      h += `<h3 style="margin-top:12px">Priority repairs</h3><ul>${pr.length ? pr.map((s, i) =>
        `<li><button type="button" data-seg="${esc(s.segment_id)}">${i + 1}. ${esc(segName(s))}
          <span>${esc(s.band_label)} · score ${s.score} · ${plural(s.n_events, "event")}${s.rough_m ? ` · ${s.rough_m} m rough` : ""} · ${s.covered_m} m ridden</span></button></li>`).join("")
        : `<li class="empty">No scored segment.</li>`}</ul>`;
      if (S.segs) h += `<p class="note">${esc(S.segs.score_note)}</p>`;
    } else {
      const latest = [...S.events].map((f) => f.properties)
        .sort((a, b) => String(b.date).localeCompare(String(a.date))).slice(0, 6);
      h += `<h3>Latest reports</h3><ul>${latest.map((p) =>
        item(p, `${esc(fmtWhen(p))} · ${plural(p.reports, "report")}`)).join("")}</ul>
        <p class="note">${esc(S.publicNote || "Public view shows raw reports only. Analysis is for partner cities.")}</p>`;
    }
    box.innerHTML = h;
  }

  $("lists").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.id) select(b.dataset.id, true);
    else if (b.dataset.seg) selectSegment(b.dataset.seg);
  });

  function selectSegment(segId) {
    const g = S.segGeom.get(segId);
    if (g) map.flyToBounds(L.geoJSON(g).getBounds().pad(0.6), { maxZoom: 19, animate: !reducedMotion });
    const evs = S.events.map((f) => f.properties).filter((p) => p.segment_id === segId)
      .sort((a, b) => (b.severity || 0) - (a.severity || 0));
    if (evs.length) select(evs[0].id, !g);
  }

  function select(id, fly) {
    const p = S.byId.get(id);
    if (!p) return;
    if (S.sel && S.markers.has(S.sel)) {
      const el = S.markers.get(S.sel).getElement(); if (el) el.classList.remove("sel");
    }
    S.sel = id;
    const m = S.markers.get(id);
    if (m) {
      const el = m.getElement(); if (el) el.classList.add("sel");
      m.bringToFront();
    }
    if (fly) map.flyTo([p.lat, p.lon], Math.max(map.getZoom(), 17), { animate: !reducedMotion });
    renderSelected();
    $("selected").scrollIntoView({ block: "nearest", behavior: reducedMotion ? "auto" : "smooth" });
  }

  function renderSelected() {
    const box = $("selected");
    const par = S.view === "partner";
    const p = S.sel && S.byId.get(S.sel);
    if (!p) {
      box.innerHTML = `<h3 id="selH">Selected point</h3><p class="note">Select a point on the map
        (mouse, or Tab then Enter) or in a list.</p>`;
      return;
    }
    const way = p.way_name || "unknown way";
    let h = `<h3 id="selH">Selected point</h3>
      <p style="margin:0"><b>${esc(typeLabel(p.type))}</b> (${esc(p.id)})</p>
      <dl class="kv">
        <dt>Location</dt><dd>${p.lat.toFixed(5)}, ${p.lon.toFixed(5)}<br>${esc(way)}</dd>
        <dt>${dateWord(p)}</dt><dd>${esc(fmtDate(p.date))}</dd>
        <dt>Reports</dt><dd>${p.reports}</dd>`;
    if (par) {
      h += `<dt>Severity</dt><dd>${p.severity}/5 <span class="note">(uncalibrated heuristic)</span></dd>
            <dt>Confidence</dt><dd>${Math.round((p.confidence || 0) * 100)} %</dd>`;
      if (p.matched === false) h += `<dt>Map match</dt><dd>not matched: shown at raw GPS</dd>`;
      if (p.detail && p.detail.reason) {
        h += `<dt>Rider detail</dt><dd>${esc(REASONS.find((r) => r[0] === p.detail.reason)?.[1] || p.detail.reason)}${p.detail.text ? `: ${esc(p.detail.text)}` : ""}</dd>`;
      }
    } else {
      h += `<dt>Severity</dt><dd class="note">partner view only</dd>`;
    }
    h += `</dl>
      <form id="detailForm">
        <label for="reason">Add a detail</label>
        <div class="row">
          <select id="reason">${REASONS.map(([v, t]) => `<option value="${v}">${t}</option>`).join("")}</select>
        </div>
        <textarea id="detailText" maxlength="200" aria-describedby="detailCount" placeholder="Optional note (200 characters max)"></textarea>
        <div class="row"><button class="btn sm" type="submit">Send detail</button>
          <span id="detailCount" class="note">0/200</span></div>
        <p id="detailMsg" class="note" aria-live="polite"></p>
      </form>
      <div class="photo">
        <p style="margin:0">${esc(PRIVACY)}</p>
        <div class="row">
          <label for="photoFile" class="sr">Photo (JPEG or PNG)</label>
          <input id="photoFile" type="file" accept="image/jpeg,image/png">
          <button id="photoBtn" class="btn sm" type="button">Upload photo</button>
        </div>
        <p id="photoMsg" class="note" aria-live="polite"></p>
        <div id="photoThumb">${p.photo_url ? `<img src="${esc(p.photo_url)}" alt="Photo attached to ${esc(p.id)}">` : ""}</div>
      </div>`;
    box.innerHTML = h;
    if (par && p.detail && p.detail.reason) $("reason").value = p.detail.reason;
    $("detailText").addEventListener("input", (e) => {
      $("detailCount").textContent = `${e.target.value.length}/200`;
    });
    $("detailForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      const msg = $("detailMsg");
      const text = $("detailText").value.trim();
      try {
        await api(`/api/events/${encodeURIComponent(p.id)}/detail`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ reason: $("reason").value, text: text || null }) });
        msg.textContent = "Saved. Thank you."; msg.className = "note ok";
        if (par) p.detail = { reason: $("reason").value, text: text || null };
      } catch (err) { msg.textContent = `Not saved: ${err.message}`; msg.className = "note bad"; }
    });
    $("photoBtn").addEventListener("click", () => uploadPhoto(p, $("photoFile").files[0]));
  }

  async function uploadPhoto(p, file) {
    const msg = $("photoMsg");
    if (!file) { msg.textContent = "Choose a JPEG or PNG first."; msg.className = "note bad"; return null; }
    if (!/^image\/(jpeg|png)$/.test(file.type)) {
      msg.textContent = "Only JPEG or PNG photos are accepted."; msg.className = "note bad"; return null;
    }
    if (file.size > 5 * 1024 * 1024) {
      msg.textContent = "Photo larger than 5 MB."; msg.className = "note bad"; return null;
    }
    const fd = new FormData();
    fd.append("file", file, file.name || "photo");
    msg.textContent = "Uploading…"; msg.className = "note";
    try {
      const r = await api(`/api/events/${encodeURIComponent(p.id)}/photo`, { method: "POST", body: fd });
      msg.textContent = r.privacy; msg.className = "note ok";
      p.photo_url = r.photo_url;
      $("photoThumb").innerHTML = `<img src="${esc(r.photo_url)}" alt="Photo attached to ${esc(p.id)}">`;
      return r;
    } catch (err) { msg.textContent = `Upload failed: ${err.message}`; msg.className = "note bad"; return null; }
  }

  // ------------------------------------------------------------------ view toggle + key
  function setView(v) {
    S.view = v;
    $("vpub").setAttribute("aria-pressed", String(v === "public"));
    $("vpar").setAttribute("aria-pressed", String(v === "partner"));
    $("keyBox").hidden = true;
    if (v === "public") { S.segs = S.alerts = S.prios = null; }
    return refresh();
  }
  function showKeyBox(err) {
    $("keyBox").hidden = false;
    $("keyErr").hidden = !err; $("keyErr").textContent = err || "";
    $("lists").innerHTML = `<p class="note">Enter the partner key to see severity, segments,
      alerts and priority repairs.</p>`;
    layers.seg.clearLayers(); layers.rough.clearLayers();
  }
  $("vpub").addEventListener("click", () => setView("public"));
  $("vpar").addEventListener("click", () => setView("partner"));
  $("keyForm").addEventListener("submit", (e) => {
    e.preventDefault();
    S.key = $("keyInput").value.trim() || null;
    $("keyBox").hidden = true;
    refresh();
  });

  // ------------------------------------------------------------------ ride log, upload, replay
  function log(html, cls) {
    const l = $("rideLog");
    if (l.dataset.used !== "1") { l.innerHTML = ""; l.dataset.used = "1"; }
    const d = document.createElement("div");
    if (cls) d.className = cls;
    d.innerHTML = html;
    l.appendChild(d);
    l.scrollTop = l.scrollHeight;
  }
  function clearLog() { const l = $("rideLog"); l.innerHTML = ""; l.dataset.used = "1"; }

  function eventLine(e, par, verb) {
    const t = typeof e.t_peak_ms === "number" ? `t = ${(e.t_peak_ms / 1000).toFixed(1)} s · ` : "";
    return `${t}<b>${esc(typeLabel(e.type))}</b> ${verb}` +
      (par && typeof e.confidence === "number" ? ` · confidence ${Math.round(e.confidence * 100)} %` : "") +
      (par && e.severity ? ` · severity ${e.severity}/5` : "") +
      ` · ${e.lat.toFixed(5)}, ${e.lon.toFixed(5)}` +
      (e.merged_into_existing ? " · merged into an existing report" : "");
  }

  $("uploadForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = $("csvFile").files[0];
    if (!file) return;
    if (file.size > 20 * 1024 * 1024) { log("CSV larger than 20 MB.", "bad"); return; }
    const par = S.view === "partner" && !!S.key;
    const fd = new FormData();
    fd.append("file", file, file.name);
    fd.append("mount", $("csvMount").value);
    fd.append("provenance", $("csvProv").value);
    clearLog();
    $("rideLog").scrollIntoView({ block: "nearest", behavior: reducedMotion ? "auto" : "smooth" });
    log(`Uploading ${esc(file.name)}…`);
    try {
      const r = await api("/api/rides/upload-csv", { method: "POST", body: fd, partner: par });
      const s = r.summary || {};
      log(`Ride <b>${esc(r.ride_id)}</b> (${esc(r.provenance)}): ${Math.round(s.duration_s || 0)} s,
           ${Math.round(s.distance_m || 0)} m, ${plural(r.events.length, "detection")}` +
          (r.matched ? `, ${s.gps.pct_matched} % of fixes map-matched.` : ". Not map-matched: shown at raw GPS."));
      for (const ev of r.events) log(eventLine(ev, par, "detected"));
      if (!r.events.length) log("No event detected (smooth ride).", "note");
      await loadRides();
      $("rideSel").value = r.ride_id;
      S.fitted = false;
      await refresh();
    } catch (err) {
      log(`Upload failed: ${esc(err.message)}`, "bad");
    }
  });

  function interp(coords, t, tt) {
    if (tt <= t[0]) return coords[0];
    for (let i = 1; i < t.length; i++) {
      if (tt <= t[i]) {
        const f = (tt - t[i - 1]) / Math.max(1, t[i] - t[i - 1]);
        const a = coords[i - 1], b = coords[i];
        return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
      }
    }
    return coords[coords.length - 1];
  }

  function ring(lat, lon) {
    if (reducedMotion) return;
    const r = L.circleMarker([lat, lon], { radius: 10, className: "ring", interactive: false }).addTo(layers.ride);
    setTimeout(() => layers.ride.removeLayer(r), 2600);
  }

  async function replay(rideId, durMs) {
    if (S.replaying || !rideId) return false;
    const par = S.view === "partner" && !!S.key;
    let ride;
    try {
      ride = await api(`/api/rides/${encodeURIComponent(rideId)}`, { partner: par });
    } catch (err) { clearLog(); log(`Could not load the ride: ${esc(err.message)}`, "bad"); return false; }
    clearLog();
    replayWarn();
    $("rideLog").scrollIntoView({ block: "nearest", behavior: reducedMotion ? "auto" : "smooth" });
    layers.track.clearLayers(); layers.ride.clearLayers();
    const tr = ride.track;
    if (!tr || !tr.geometry || tr.geometry.coordinates.length < 2) {
      log("This ride has no GPS track to replay.", "bad"); return false;
    }
    const coords = tr.geometry.coordinates.map(([lo, la]) => [la, lo]);
    const t = tr.properties.t_ms;
    const T = t[t.length - 1] || 1;
    const matched = (tr.properties.matched || []).some(Boolean);
    L.polyline(coords, { className: "track", interactive: false }).addTo(layers.track);
    map.fitBounds(L.latLngBounds(coords).pad(0.3), { maxZoom: 18, animate: !reducedMotion });
    log(`Replaying ride <b>${esc(ride.id)}</b> (${esc(ride.provenance)}, ${esc(ride.mount)},
         ${Math.round(ride.duration_s || T / 1000)} s compressed to ~${Math.round(durMs / 1000)} s)` +
        (matched ? "" : ". Track not map-matched: raw GPS."));
    const evs = [...ride.events].sort((a, b) => a.t_peak_ms - b.t_peak_ms);
    const rider = L.circleMarker(coords[0], { radius: 9, className: "rider", interactive: false }).addTo(layers.ride);
    S.replaying = true;
    $("replayBtn").disabled = true;
    let k = 0;
    await new Promise((done) => {
      const t0 = performance.now();
      const step = (now) => {
        const f = reducedMotion ? 1 : Math.min(1, (now - t0) / durMs);
        const tt = f * T;
        rider.setLatLng(interp(coords, t, tt));
        while (k < evs.length && evs[k].t_peak_ms <= tt) {
          const e = evs[k++];
          ring(e.lat, e.lon);
          log(eventLine(e, par, "detected"));
          const m = S.markers.get(e.id);
          if (m) m.bringToFront();
        }
        if (f < 1) requestAnimationFrame(step); else done();
      };
      requestAnimationFrame(step);
    });
    log(`Ride finished: ${plural(evs.length, "detection")}. Select a point to add a detail or a photo.`, "note");
    S.replaying = false;
    $("replayBtn").disabled = false;
    setTimeout(() => layers.ride.removeLayer(rider), 1500);
    return true;
  }
  const replayMs = () => Math.max(300, Number(params.get("replay_ms")) || 10000);
  $("replayBtn").addEventListener("click", () => replay($("rideSel").value, replayMs()));

  // ------------------------------------------------------------------ model dialog
  function modelHtml() {
    if (!S.model) {
      return `<p class="bad">No model card found. Train a model with
        <code>python -m smartcycle train --data-dir &lt;dir&gt;</code>.</p>`;
    }
    const c = S.model.model_card, m = S.model.metrics, n = S.model.notes;
    const pv = c.data_provenance;
    let h = `<p>Data provenance <span class="pv pv-${esc(pv)}" style="padding:.1em .6em;border-radius:999px">${esc(pv)}</span>
      · model <b>${esc((c.model || "?").toUpperCase())}</b>
      · ${c.n_rides ?? "?"} rides · ${c.n_windows ?? "?"} windows
      ${c.mount_filter ? `· mount ${esc(c.mount_filter)}` : ""}
      · speed features ${c.include_speed ? "included" : "excluded"}
      ${c.trained_at ? `· trained ${esc(fmtDate(c.trained_at))}` : ""}</p>`;
    if (pv !== "real") {
      h += `<p class="bad"><b>${pv === "mixed" ? "Trained partly on synthetic data" : "Trained on SYNTHETIC data"}:</b>
            these numbers say nothing about real roads.</p>`;
    }
    if (c.n_windows_per_class) {
      h += `<h3>Windows per class</h3><table><tr>${Object.keys(c.n_windows_per_class)
        .map((k) => `<th>${esc(typeLabel(k))}</th>`).join("")}</tr><tr>${Object.values(c.n_windows_per_class)
        .map((v) => `<td class="n">${v}</td>`).join("")}</tr></table>`;
    }
    if (m && m.models) {
      const proto = m.cv_protocol || {};
      const evAny = Object.values(m.models).find((x) => x.events)?.events;
      h += `<h3>Cross-validation (${esc(proto.mode || m.cv_mode || "?")}, ${proto.n_splits || m.n_folds || "?"} folds, pooled out-of-fold)</h3>
        ${proto.header ? `<p class="note">${esc(proto.header)}</p>` : ""}
        <table><tr><th>Model</th><th>Macro-F1</th><th>Pothole / bump events</th><th>False alarms (Pothole / bump)</th>
        <th>Irregularity events</th><th>False alarms (Irregularity)</th></tr>`;
      const fa = (e) => (e ? `${e.false_alarms} (${typeof e.false_alarms_per_km === "number" ? e.false_alarms_per_km.toFixed(1) : "–"} / km)` : "–");
      const names = { rf: "Random forest", rule: "Rule baseline", hgb: "Gradient boosting" };
      for (const k of ["rf", "rule", "hgb"]) {
        const x = m.models[k];
        if (!x) continue;
        const ev = x.events || {};
        h += `<tr${k === c.model ? ' style="font-weight:700"' : ""}><td>${names[k]}${k === c.model ? " (final)" : ""}</td>
          <td class="n">${f3(x.macro_f1)}</td>
          <td class="n">${esc(ev.Pothole?.detected_of || "–")}</td><td class="n">${fa(ev.Pothole)}</td>
          <td class="n">${esc(ev.Irregularity?.detected_of || "–")}</td><td class="n">${fa(ev.Irregularity)}</td></tr>`;
      }
      h += `</table>`;
      if (evAny) h += `<p class="note">Event matching ±${evAny.tolerance_s} s; evaluated distance ${evAny.distance_km} km
        (false alarms per km over such a short distance are very noisy).</p>`;
      const d3 = m.three_class_diagnostic;
      if (d3) {
        h += `<p>3-class diagnostic (${esc((d3.classes || []).join(" / "))}): macro-F1 <b>${f3(d3.macro_f1)}</b>.
          <span class="note">${esc(d3.note || "")}</span></p>`;
      }
      if (m.provenance_note) h += `<p class="note">${esc(m.provenance_note)}</p>`;
    }
    if (n && n.caveats && n.caveats.length) {
      h += `<h3>Caveats</h3><ul class="cav">${n.caveats.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;
    }
    if (c.limitations && c.limitations.length) {
      h += `<details><summary>Model card limitations</summary><ul class="cav">${c.limitations
        .map((x) => `<li>${esc(x)}</li>`).join("")}</ul></details>`;
    }
    const imgs = S.model.report_images || [];
    if (imgs.length) {
      h += `<div class="imgs">${imgs.map((u) => `<figure style="margin:0"><img src="${esc(u)}" alt="${esc(u.split("/").pop().replace(".png", "").replace("_", " "))}" loading="lazy">
        <figcaption class="note">${esc(u.split("/").pop())}</figcaption></figure>`).join("")}</div>`;
    }
    h += `<p class="note">Severity (1–5) shown in the partner view is an uncalibrated heuristic, not a model output.</p>`;
    return h;
  }
  function openModel() {
    $("modelBody").innerHTML = modelHtml();
    $("modelDlg").showModal();
  }
  $("modelBadge").addEventListener("click", openModel);
  $("modelClose").addEventListener("click", () => $("modelDlg").close());

  // ------------------------------------------------------------------ self-test (headless)
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  async function selftest() {
    const out = [];
    const ok = (name, cond, info) => out.push(`${cond ? "PASS" : "FAIL"} ${name}${info ? `: ${info}` : ""}`);
    try {
      ok("events loaded", S.events.length > 0, `${S.events.length} events`);
      ok("one marker per event", S.markers.size === S.events.length);
      ok("network layer", !S.health.osm_loaded || layers.net.getLayers().length > 0);
      const lists = $("lists").textContent;
      ok("public: latest reports", /Latest reports/.test(lists));
      ok("public: no severity value in lists", !/Severity \d|\d\/5/i.test(lists));
      ok("public: Pothole label is 'Pothole / bump'", !S.events.some((f) => f.properties.type === "Pothole")
        || /Pothole \/ bump/.test(lists + document.querySelector(".leaflet-overlay-pane").innerHTML));
      // keyboard selection
      const first = S.events[0].properties.id;
      const el = S.markers.get(first).getElement();
      el.focus();
      el.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));
      ok("keyboard select (Enter on marker)", S.sel === first && $("selected").textContent.includes(first));
      ok("public: severity hidden in selected panel", !/\d\/5/.test($("selected").textContent));
      // mouse selection
      const second = S.events[S.events.length - 1].properties.id;
      S.markers.get(second).fire("click");
      ok("mouse select (click on marker)", S.sel === second);
      // detail
      $("reason").value = "other"; $("detailText").value = "selftest";
      $("detailForm").dispatchEvent(new Event("submit", { cancelable: true }));
      await sleep(400);
      ok("detail form saved", /Saved/.test($("detailMsg").textContent), $("detailMsg").textContent);
      // photo (PNG drawn on a canvas)
      const cv = document.createElement("canvas"); cv.width = 64; cv.height = 48;
      cv.getContext("2d").fillRect(0, 0, 32, 24);
      const blob = await new Promise((r) => cv.toBlob(r, "image/png"));
      const r = await uploadPhoto(S.byId.get(second), new File([blob], "t.png", { type: "image/png" }));
      ok("photo upload + thumbnail", !!r && !!$("photoThumb").querySelector("img"), r && r.photo_url);
      // partner view
      await setView("partner");
      ok("partner: key prompt shown", !$("keyBox").hidden);
      $("keyInput").value = DEFAULT_KEY;
      $("keyForm").dispatchEvent(new Event("submit", { cancelable: true }));
      await sleep(800);
      const pl = $("lists").textContent;
      ok("partner: alerts + priority repairs", /Alerts/.test(pl) && /Priority repairs/.test(pl));
      ok("partner: segments coloured", layers.seg.getLayers().length > 0, `${layers.seg.getLayers().length}`);
      ok("partner: dot size = severity", S.events.every((f) => f.properties.severity >= 1));
      const pb = $("lists").querySelector("button[data-seg],button[data-id]");
      if (pb) { pb.click(); await sleep(300); }
      ok("partner: list click selects a point", !!S.sel && /Severity/.test($("selected").textContent));
      // model dialog
      openModel();
      const mt = $("modelBody").textContent;
      ok("model dialog open", $("modelDlg").open && /Macro-F1/.test(mt));
      ok("model dialog: provenance + caveats", /provenance/i.test(mt) && (!S.model?.notes || /Caveats/.test(mt)));
      $("modelDlg").close();
      // replay
      if (S.rides.length) {
        const done = await replay(S.rides[0].id, 600);
        ok("replay finished", done && /Ride finished/.test($("rideLog").textContent));
      }
      // theme
      const before = currentTheme();
      $("themeBtn").click();
      ok("theme toggle", currentTheme() !== before);
      $("themeBtn").click();
      await setView("public");
      ok("back to public: no severity value", !/Severity \d|\d\/5/i.test($("lists").textContent));
    } catch (e) {
      out.push(`FAIL exception: ${e && e.stack ? e.stack : e}`);
    }
    const n = out.filter((l) => l.startsWith("PASS")).length;
    out.push(`SELFTEST DONE ${n}/${out.length}`);
    const st = $("selftest"); st.hidden = false; st.textContent = out.join("\n");
    st.style.whiteSpace = "pre"; console.log(out.join("\n"));
  }

  // ------------------------------------------------------------------ start
  (async () => {
    const up = await boot();
    if (!up) return;
    if (params.get("view") === "partner") {
      S.key = DEFAULT_KEY;
      await setView("partner");
    }
    if (params.get("select")) select(params.get("select"), true);
    if (params.get("dialog") === "model") openModel();
    if (params.get("replay")) replay(params.get("replay"), replayMs());
    if (params.get("selftest") === "1") selftest();
  })();
})();
