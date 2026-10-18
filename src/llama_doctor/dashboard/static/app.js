(() => {
  "use strict";

  const rootStyle = getComputedStyle(document.documentElement);
  const color = (name) => rootStyle.getPropertyValue(name).trim();
  const byId = (id) => document.getElementById(id);
  const colors = {
    grid: color("--grid"), tick: color("--text-3"), legend: color("--text-2"), decode: color("--s-decode"),
    prefill: color("--s-prefill"), kv: color("--s-kv"), cache: color("--s-cache"),
    draft: color("--s-draft"), acceptance: color("--s-accept"), context: color("--s-ctx")
  };
  let lastState = null;
  let retryDelay = 2000;

  function number(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }

  function text(value, fallback = "-") {
    return typeof value === "string" || typeof value === "number" ? String(value) : fallback;
  }

  // Evidence arrives at full precision; round only here: 1 decimal, or 3 below 1 so ratios stay readable.
  function displayNumber(value) {
    if (typeof value !== "number" || Number.isInteger(value)) {
      return text(value, "n/a");
    }
    return String(Number(value.toFixed(Math.abs(value) >= 1 ? 1 : 3)));
  }

  function rate(value) {
    const valueNumber = number(value);
    return valueNumber === null ? "n/a" : valueNumber.toFixed(1);
  }

  function percent(value) {
    const valueNumber = number(value);
    return valueNumber === null ? "n/a" : `${(valueNumber * 100).toFixed(1)} %`;
  }

  function draft(value) {
    const valueNumber = number(value);
    return valueNumber === null ? "n/a" : valueNumber.toFixed(2);
  }

  function timeLabel(value) {
    const valueNumber = number(value);
    if (valueNumber === null) {
      return "-";
    }
    const date = new Date(valueNumber * 1000);
    const pad = (part) => String(part).padStart(2, "0");
    return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  }

  function severityLabel(severity) {
    return {
      critical: "CRIT", warn: "WARN", info: "INFO", ok: "OK",
      insufficient_data: "INSUFFICIENT DATA"
    }[severity] || "INSUFFICIENT DATA";
  }

  function severityValue(value) {
    return ["critical", "warn", "info", "ok", "insufficient_data"].includes(value)
      ? value
      : "insufficient_data";
  }

  function impactValue(value) {
    return ["high", "medium", "low"].includes(value) ? value : "low";
  }

  function append(parent, tag, value, className) {
    const element = document.createElement(tag);
    element.textContent = value;
    if (className) {
      element.className = className;
    }
    parent.append(element);
    return element;
  }

  function isolatedPointRadius(context) {
    const data = context.dataset.data;
    const at = (index) => index >= 0 && index < data.length ? data[index] : null;
    const index = context.dataIndex;
    return at(index) !== null && at(index - 1) === null && at(index + 1) === null ? 3 : 0;
  }

  function lineChart(canvasId, caption, datasets, percentScale) {
    return new Chart(byId(canvasId), {
      type: "line",
      data: { labels: [], datasets },
      options: {
        animation: false, responsive: true, maintainAspectRatio: false, spanGaps: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          // Line swatches carry each dataset's borderDash, so dashed series read as dashed in the legend too.
          legend: {
            display: canvasId === "chart-kv-cache",
            labels: { color: colors.legend, usePointStyle: true, pointStyle: "line", pointStyleWidth: 28 }
          },
          tooltip: {
            callbacks: {
              label(context) {
                const value = context.parsed.y;
                const unit = context.dataset.unit;
                return `${context.dataset.label}: ${value === null ? "n/a" : value.toFixed(unit === "tokens" ? 2 : 1)} ${unit}`;
              }
            }
          }
        },
        scales: {
          x: { grid: { color: colors.grid }, ticks: { color: colors.tick } },
          y: {
            beginAtZero: true, max: percentScale ? 100 : undefined,
            grid: { color: colors.grid }, ticks: { color: colors.tick },
            // The rotated title must fit the plot height, so the unit goes on its own line.
            title: { display: true, text: caption.split(/ (?=\()/), color: colors.legend }
          }
        },
        elements: { line: { borderWidth: 2, tension: 0 }, point: { radius: isolatedPointRadius, hoverRadius: 4 } }
      }
    });
  }

  const charts = {
    decode: lineChart("chart-decode", "Decode (tokens/s)", [{ label: "Decode", data: [], borderColor: colors.decode, unit: "tok/s" }], false),
    prefill: lineChart("chart-prefill", "Prefill (tokens/s)", [{ label: "Prefill", data: [], borderColor: colors.prefill, unit: "tok/s" }], false),
    kvCache: lineChart("chart-kv-cache", "Usage (%)", [
      { label: "KV usage", data: [], borderColor: colors.kv, backgroundColor: colors.kv, unit: "%" },
      { label: "Prefix-cache hit", data: [], borderColor: colors.cache, backgroundColor: colors.cache, unit: "%" },
      // Dashed: a peak since server start, not a live gauge like the other two lines.
      { label: "Context used", data: [], borderColor: colors.context, backgroundColor: colors.context, borderDash: [6, 4], unit: "%" }
    ], true),
    draft: lineChart("chart-draft-len", "Draft length (tokens)", [{ label: "Draft length", data: [], borderColor: colors.draft, unit: "tokens" }], false),
    acceptance: lineChart("chart-acceptance", "Acceptance (%)", [{ label: "Acceptance", data: [], borderColor: colors.acceptance, unit: "%" }], true)
  };

  function historyOf(state) {
    return Array.isArray(state.history) ? state.history : [];
  }

  function updateChart(chart, history, fields) {
    chart.data.labels = history.map((point) => timeLabel(point.t));
    fields.forEach((field, index) => {
      chart.data.datasets[index].data = history.map((point) => {
        const value = number(point[field]);
        return fields[index].includes("ratio") || fields[index] === "draft_acceptance"
          ? value === null ? null : value * 100
          : value;
      });
    });
    chart.update("none");
  }

  function updateCharts(history) {
    updateChart(charts.decode, history, ["decode_tps"]);
    updateChart(charts.prefill, history, ["prefill_tps"]);
    updateChart(charts.kvCache, history, ["kv_usage_ratio", "cache_hit_ratio", "context_used_ratio"]);
    updateChart(charts.draft, history, ["avg_draft_len"]);
    updateChart(charts.acceptance, history, ["draft_acceptance"]);
  }

  function latest(history, field) {
    for (let index = history.length - 1; index >= 0; index -= 1) {
      const value = number(history[index][field]);
      if (value !== null) {
        return value;
      }
    }
    return null;
  }

  function count(value) {
    const valueNumber = number(value);
    return valueNumber === null ? "n/a" : String(Math.round(valueNumber));
  }

  // Labels are built once so the info buttons inside them keep focus across renders.
  const TILE_LABELS = {
    "tile-decode": "Decode tok/s",
    "tile-prefill": "Prefill tok/s",
    "tile-cache": "Prefix-cache hit",
    "tile-kv": "KV usage",
    "tile-context": "Context",
    "tile-queue": "Queue"
  };

  Object.entries(TILE_LABELS).forEach(([id, label]) => {
    const tile = byId(id);
    append(append(tile, "div", "", "tile-label"), "span", label);
    append(tile, "div", "", "tile-value");
  });

  function tileValue(id) {
    const body = byId(id).querySelector(".tile-value");
    body.replaceChildren();
    return body;
  }

  function renderTile(id, live, lifetime, formatter) {
    const body = tileValue(id);
    const hero = live === null ? lifetime : live;
    append(body, "div", formatter(hero), "hero");
    append(body, "div", live === null ? "lifetime avg" : `lifetime ${formatter(lifetime)}`, "secondary");
  }

  // KV usage is an instantaneous gauge with no lifetime average, so it gets no lifetime line.
  function renderKvTile(history, unexported, kv) {
    const body = tileValue("tile-kv");
    if (unexported) {
      append(body, "div", "not exported", "hero");
      if (kv && kv.estimated === true && number(kv.usage_ratio) !== null) {
        append(body, "div", `~${percent(number(kv.usage_ratio))} estimated - see Context`, "secondary");
      }
      return;
    }
    append(body, "div", percent(latest(history, "kv_usage_ratio")), "hero");
    append(body, "div", "current", "secondary");
  }

  function renderQueueTile(history) {
    const body = tileValue("tile-queue");
    append(body, "div", count(latest(history, "requests_processing")), "hero");
    append(body, "div", `${count(latest(history, "requests_deferred"))} waiting`, "secondary");
  }

  // Peak tokens over n_ctx; ratio and "78.5k / 80.1k" label come preformatted from the sampler.
  function renderContextTile(state) {
    const context = state.context && typeof state.context === "object" ? state.context : {};
    const body = tileValue("tile-context");
    const used = number(context.live_used);
    const capacity = number(context.live_capacity);
    if (used !== null && capacity !== null && capacity > 0) {
      append(body, "div", percent(used / capacity), "hero");
      append(body, "div", `peak ${percent(number(context.used_ratio))}`, "secondary");
    } else {
      append(body, "div", percent(number(context.used_ratio)), "hero");
    }
    append(body, "div", text(context.label, "n/a"), "secondary");
  }

  function updateTiles(state, history) {
    const lifetime = state.lifetime && typeof state.lifetime === "object" ? state.lifetime : {};
    const notes = state.notes && typeof state.notes === "object" ? state.notes : {};
    renderTile("tile-decode", latest(history, "decode_tps"), number(lifetime.decode_tps), rate);
    renderTile("tile-prefill", latest(history, "prefill_tps"), number(lifetime.prefill_tps), rate);
    renderTile("tile-cache", latest(history, "cache_hit_ratio"), number(lifetime.cache_hit_ratio), percent);
    renderKvTile(history, notes.kv_unexported === true, state.kv && typeof state.kv === "object" ? state.kv : null);
    renderContextTile(state);
    renderQueueTile(history);
  }

  // Which catalog entries explain each tile, chart header and model-strip field.
  // The tooltip text itself always comes from /api/catalog.
  const METRIC_INFO = {
    "tile-decode": ["decode_tps"],
    "tile-prefill": ["prefill_tps"],
    "tile-cache": ["prefix_cache_hit"],
    "tile-kv": ["kv_usage"],
    "tile-context": ["context_used"],
    "tile-queue": ["queue_processing", "queue_deferred"],
    "chart-decode": ["decode_tps"],
    "chart-prefill": ["prefill_tps"],
    "chart-kv-cache": ["kv_usage", "prefix_cache_hit", "context_used"],
    "chart-draft-len": ["draft_length"],
    "chart-acceptance": ["acceptance"],
    "hdr-quant": ["quant"],
    "hdr-ctx": ["context_tokens"],
    "hdr-slots": ["slots"]
  };
  // Tiles show a single info button; when one covers several entries each section gets this prefix.
  const INFO_PREFIX = {
    queue_processing: "Processing",
    queue_deferred: "Deferred"
  };
  let infoCount = 0;
  let catalogLoaded = false;
  let catalogPending = false;

  function infoHost(element) {
    if (element.classList.contains("tile")) {
      return element.querySelector(".tile-label");
    }
    if (element.tagName === "CANVAS") {
      return element.closest(".panel").querySelector("h2");
    }
    return element.previousElementSibling;
  }

  // items: [{ metricId, entry }]; more than one item renders prefixed sections in one tooltip.
  function infoButton(host, items) {
    const key = items.map((item) => item.metricId).join(" ");
    if (host.querySelector(`[data-metric="${key}"]`)) {
      return;
    }
    const combined = items.length > 1;
    infoCount += 1;
    const tipId = `info-tip-${infoCount}`;
    const wrap = document.createElement("span");
    wrap.className = "info-wrap";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "info";
    button.dataset.metric = key;
    button.textContent = "i";
    const labels = items.map(({ metricId, entry }) => text(entry.label, metricId));
    button.setAttribute("aria-label", `About ${labels.join(" and ")}`);
    button.setAttribute("aria-describedby", tipId);
    const tip = document.createElement("span");
    tip.id = tipId;
    tip.className = "chart-note info-tip";
    tip.setAttribute("role", "tooltip");
    items.forEach(({ metricId, entry }, index) => {
      const description = append(tip, "span", text(entry.description, ""));
      if (combined) {
        description.prepend(`${INFO_PREFIX[metricId] || labels[index]}: `);
      }
      append(tip, "span", text(entry.interpretation, ""), "info-interp");
    });
    // Escape hides the tooltip without moving focus or the pointer.
    button.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        wrap.classList.add("dismissed");
      }
    });
    ["blur", "mouseleave"].forEach((name) => wrap.addEventListener(name, () => wrap.classList.remove("dismissed"), true));
    wrap.append(button, tip);
    host.append(wrap);
  }

  function attachInfo(catalog) {
    if (!catalog || typeof catalog !== "object") {
      throw new Error("invalid catalog");
    }
    Object.entries(METRIC_INFO).forEach(([elementId, metricIds]) => {
      const element = byId(elementId);
      const host = infoHost(element);
      const items = metricIds
        .map((metricId) => ({ metricId, entry: catalog[metricId] }))
        .filter(({ entry }) => entry && typeof entry === "object");
      if (element.classList.contains("tile")) {
        if (items.length) {
          infoButton(host, items);
        }
      } else {
        items.forEach((item) => infoButton(host, [item]));
      }
    });
  }

  function loadCatalog() {
    if (catalogLoaded || catalogPending) {
      return;
    }
    catalogPending = true;
    fetch("/api/catalog")
      .then((response) => {
        if (!response.ok) {
          throw new Error("catalog request failed");
        }
        return response.json();
      })
      .then((catalog) => {
        attachInfo(catalog);
        catalogLoaded = true;
      })
      .catch(() => {})
      .finally(() => {
        catalogPending = false;
      });
  }

  // The sampler decides which notes apply; the note text lives in index.html.
  const NOTE_IDS = {
    decode_idle: "note-decode",
    prefill_idle: "note-prefill",
    kv_unexported: "note-kv",
    cache_hit_unexported: "note-cache-hit",
    slots_unavailable: "note-slots",
  };

  const SLOTS_NOTE_FALLBACK = "Live slot data unavailable; context use is estimated from metrics";
  const SLOTS_NOTE_ESTIMATED = "KV usage is estimated from /slots";

  // With live slots the estimate comes from /slots; without them the old wording applies.
  function updateSlotsNote(state, notes) {
    const note = byId("note-slots");
    const kv = state.kv && typeof state.kv === "object" ? state.kv : null;
    const fromSlots = notes.slots_unavailable !== true && notes.kv_unexported === true
      && kv !== null && kv.estimated === true;
    note.textContent = fromSlots ? SLOTS_NOTE_ESTIMATED : SLOTS_NOTE_FALLBACK;
    note.classList.toggle("hidden", !(fromSlots || notes.slots_unavailable === true));
  }

  function updateNotes(state) {
    const notes = state.notes && typeof state.notes === "object" ? state.notes : {};
    Object.entries(NOTE_IDS).forEach(([flag, id]) => {
      byId(id).classList.toggle("hidden", notes[flag] !== true);
    });
    updateSlotsNote(state, notes);
  }

  function updateRateWindow(state) {
    const windowSeconds = number(state.rate_window_s);
    const label = windowSeconds === null ? "" : `avg over ${Math.round(windowSeconds)} s`;
    document.querySelectorAll(".rate-window").forEach((element) => {
      element.textContent = label;
    });
  }

  function updateHeader(state) {
    const server = state.server && typeof state.server === "object" ? state.server : {};
    byId("hdr-target").textContent = text(state.target);
    byId("hdr-model").textContent = text(server.model_name);
    byId("hdr-quant").textContent = text(server.quant);
    byId("hdr-ctx").textContent = text(server.n_ctx);
    byId("hdr-slots").textContent = text(server.total_slots);
    byId("hdr-build").textContent = text(server.build_info);
    const report = state.report && typeof state.report === "object" ? state.report : null;
    const severity = severityValue(report && report.overall);
    const badge = byId("overall-badge");
    badge.textContent = severityLabel(severity);
    badge.className = `badge sev-${severity}`;
  }

  function updateUpdatedAt() {
    const target = byId("updated-at");
    const timestamp = lastState && number(lastState.updated_at);
    if (timestamp === null) {
      target.textContent = "no data yet";
      return;
    }
    const seconds = Math.max(0, Math.floor(Date.now() / 1000 - timestamp));
    target.textContent = `updated ${seconds} s ago`;
  }

  function updateTable(container, history, fields) {
    container.replaceChildren();
    const table = document.createElement("table");
    const head = document.createElement("thead");
    const headRow = document.createElement("tr");
    ["time", ...fields.map((item) => item.label)].forEach((label) => append(headRow, "th", label));
    head.append(headRow);
    table.append(head);
    const body = document.createElement("tbody");
    history.slice(-20).forEach((point) => {
      const row = document.createElement("tr");
      append(row, "td", timeLabel(point.t));
      fields.forEach((item) => append(row, "td", item.format(number(point[item.key]))));
      body.append(row);
    });
    table.append(body);
    container.append(table);
  }

  const tableFields = {
    decode: [{ label: "tok/s", key: "decode_tps", format: rate }],
    prefill: [{ label: "tok/s", key: "prefill_tps", format: rate }],
    "kv-cache": [
      { label: "KV usage", key: "kv_usage_ratio", format: percent },
      { label: "Prefix-cache hit", key: "cache_hit_ratio", format: percent },
      { label: "Context used", key: "context_used_ratio", format: percent }
    ],
    speculative: [
      { label: "draft tokens", key: "avg_draft_len", format: draft },
      { label: "acceptance", key: "draft_acceptance", format: percent }
    ]
  };

  function updateTableIfOpen(key, history) {
    const container = document.querySelector(`[data-table="${key}"]`);
    if (container.closest("details").open) {
      updateTable(container, history, tableFields[key]);
    }
  }

  function updateTables(history) {
    Object.keys(tableFields).forEach((key) => updateTableIfOpen(key, history));
  }

  function updateFindings(state) {
    const report = state.report && typeof state.report === "object" ? state.report : {};
    const findings = Array.isArray(report.findings) ? report.findings : [];
    const visible = findings.filter((item) => item && !["ok", "insufficient_data"].includes(severityValue(item.severity)));
    const skipped = findings.filter((item) => item && severityValue(item.severity) === "insufficient_data");
    const passed = findings.filter((item) => item && severityValue(item.severity) === "ok");
    const list = byId("findings-list");
    list.replaceChildren();
    visible.forEach((finding) => {
      const row = document.createElement("article");
      row.className = "finding";
      const heading = document.createElement("div");
      heading.className = "finding-heading";
      const severity = severityValue(finding.severity);
      append(heading, "span", severityLabel(severity), `badge sev-${severity}`);
      append(heading, "code", text(finding.code));
      append(heading, "strong", text(finding.title));
      row.append(heading);
      append(row, "p", text(finding.detail));
      if (finding.evidence && typeof finding.evidence === "object" && !Array.isArray(finding.evidence)) {
        const evidence = document.createElement("ul");
        evidence.className = "evidence";
        Object.entries(finding.evidence).forEach(([key, value]) => append(evidence, "li", `${key} ${displayNumber(value)}`));
        row.append(evidence);
      }
      list.append(row);
    });
    const recs = byId("recs-list");
    recs.replaceChildren();
    (Array.isArray(report.recommendations) ? report.recommendations : []).forEach((recommendation) => {
      const item = document.createElement("li");
      item.className = "recommendation";
      const impact = impactValue(recommendation.impact);
      append(item, "span", impact.toUpperCase(), `impact impact-${impact}`);
      append(item, "span", text(recommendation.action));
      if (recommendation.flag !== null) {
        append(item, "code", text(recommendation.flag));
      }
      append(item, "p", text(recommendation.rationale));
      recs.append(item);
    });
    const skippedList = byId("skipped-list");
    skippedList.replaceChildren();
    skipped.forEach((finding) => append(skippedList, "li", `${text(finding.code)}: insufficient data`));
    passed.forEach((finding) => append(skippedList, "li", `${text(finding.code)}: passed`));
    byId("skipped-list").parentElement.querySelector("summary").textContent = `${findings.length - skipped.length} run, ${skipped.length} skipped`;
  }

  function updateBanner(state) {
    const banner = byId("conn-banner");
    const errors = state.errors && typeof state.errors === "object" ? Object.entries(state.errors) : [];
    if (errors.length > 0 && historyOf(state).length === 0) {
      banner.textContent = `server unreachable: ${errors.map(([endpoint, reason]) => `${endpoint} ${text(reason)}`).join("; ")}`;
      banner.classList.remove("hidden");
    } else {
      banner.classList.add("hidden");
    }
  }

  function render(state) {
    lastState = state;
    const history = historyOf(state);
    updateHeader(state);
    updateRateWindow(state);
    updateTiles(state, history);
    updateCharts(history);
    updateNotes(state);
    updateTables(history);
    updateFindings(state);
    updateBanner(state);
    updateUpdatedAt();
  }

  function disconnected() {
    const banner = byId("conn-banner");
    banner.textContent = "dashboard disconnected, retrying";
    banner.classList.remove("hidden");
  }

  function poll() {
    fetch("/api/state", { cache: "no-store" })
      .then((response) => {
        if (!response.ok) {
          throw new Error("state request failed");
        }
        return response.json();
      })
      .then((state) => {
        if (!state || typeof state !== "object") {
          throw new Error("invalid state");
        }
        const interval = number(state.interval_s);
        retryDelay = interval === null ? 2000 : interval * 1000;
        try {
          render(state);
        } catch (error) {
          console.error("dashboard render failed", error);
        }
        // Retried on every successful poll until it loads once.
        loadCatalog();
        setTimeout(poll, retryDelay);
      })
      .catch(() => {
        disconnected();
        retryDelay = Math.min(retryDelay * 2, 10000);
        setTimeout(poll, retryDelay);
      });
  }

  Object.entries(tableFields).forEach(([key]) => {
    const container = document.querySelector(`[data-table="${key}"]`);
    container.closest("details").addEventListener("toggle", (event) => {
      if (event.currentTarget.open && lastState) {
        updateTableIfOpen(key, historyOf(lastState));
      }
    });
  });

  setInterval(updateUpdatedAt, 1000);
  poll();
})();
