(() => {
  "use strict";

  const rootStyle = getComputedStyle(document.documentElement);
  const color = (name) => rootStyle.getPropertyValue(name).trim();
  const byId = (id) => document.getElementById(id);
  const colors = {
    grid: color("--grid"), tick: color("--text-3"), legend: color("--text-2"), decode: color("--s-decode"),
    prefill: color("--s-prefill"), kv: color("--s-kv"), cache: color("--s-cache"),
    draft: color("--s-draft"), acceptance: color("--s-accept")
  };
  let lastState = null;
  let retryDelay = 2000;

  function number(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : null;
  }

  function text(value, fallback = "-") {
    return typeof value === "string" || typeof value === "number" ? String(value) : fallback;
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
          legend: { display: canvasId === "chart-kv-cache", labels: { color: colors.legend } },
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
      { label: "Prefix-cache hit", data: [], borderColor: colors.cache, backgroundColor: colors.cache, unit: "%" }
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
    updateChart(charts.kvCache, history, ["kv_usage_ratio", "cache_hit_ratio"]);
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

  function renderTile(id, label, live, lifetime, formatter, unexported = false) {
    const tile = byId(id);
    tile.replaceChildren();
    append(tile, "div", label, "tile-label");
    if (unexported) {
      append(tile, "div", "not exported", "hero");
      return;
    }
    const hero = live === null ? lifetime : live;
    append(tile, "div", formatter(hero), "hero");
    append(tile, "div", live === null ? "lifetime avg" : `lifetime ${formatter(lifetime)}`, "secondary");
  }

  function updateTiles(state, history) {
    const lifetime = state.lifetime && typeof state.lifetime === "object" ? state.lifetime : {};
    const notes = state.notes && typeof state.notes === "object" ? state.notes : {};
    renderTile("tile-decode", "Decode tok/s", latest(history, "decode_tps"), number(lifetime.decode_tps), rate);
    renderTile("tile-prefill", "Prefill tok/s", latest(history, "prefill_tps"), number(lifetime.prefill_tps), rate);
    renderTile("tile-cache", "Prefix-cache hit", latest(history, "cache_hit_ratio"), number(lifetime.cache_hit_ratio), percent);
    renderTile("tile-kv", "KV usage", latest(history, "kv_usage_ratio"), number(lifetime.kv_usage_ratio), percent,
      notes.kv_unexported === true);
  }

  // The sampler decides which notes apply; the note text lives in index.html.
  const NOTE_IDS = {
    decode_idle: "note-decode",
    prefill_idle: "note-prefill",
    kv_unexported: "note-kv",
    cache_hit_unexported: "note-cache-hit",
  };

  function updateNotes(state) {
    const notes = state.notes && typeof state.notes === "object" ? state.notes : {};
    Object.entries(NOTE_IDS).forEach(([flag, id]) => {
      byId(id).classList.toggle("hidden", notes[flag] !== true);
    });
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
      { label: "Prefix-cache hit", key: "cache_hit_ratio", format: percent }
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
        Object.entries(finding.evidence).forEach(([key, value]) => append(evidence, "li", `${key} ${text(value, "n/a")}`));
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
    byId("skipped-list").parentElement.querySelector("summary").textContent = `${skipped.length} checks skipped (insufficient data)`;
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
