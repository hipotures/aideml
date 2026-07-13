let refreshMs = 2000;
let treeTarget = null;
let treeFollow = false;
let lastTreeTarget = null;
let lastSnapshot = null;
let lastTokenUsageUpdatedAt = null;
const tabOrder = ["tree", "run", "logs", "codex"];

function setLogsConnectionError(hasError) {
  const logsTab = document.querySelector('[data-tab="logs"]');
  if (logsTab) {
    logsTab.classList.toggle("connection-error", hasError);
  }
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.panel === name);
  });
  updateVisibleTabs(name);
}

function updateVisibleTabs(activeName) {
  const activeIndex = tabOrder.indexOf(activeName);
  const firstVisibleIndex = activeIndex >= 2 ? 1 : 0;
  const tabs = document.querySelector(".tabs");
  tabs.classList.toggle("has-tabs-left", firstVisibleIndex > 0);
  tabs.classList.toggle("has-tabs-right", firstVisibleIndex + 3 < tabOrder.length);
  document.querySelectorAll(".tab").forEach((tab) => {
    const index = tabOrder.indexOf(tab.dataset.tab);
    tab.classList.toggle(
      "tab-hidden",
      index < firstVisibleIndex || index >= firstVisibleIndex + 3,
    );
  });
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => switchTab(tab.dataset.tab));
});

document.querySelectorAll("[data-tree-target]").forEach((button) => {
  button.addEventListener("click", () => {
    treeTarget = treeTarget === button.dataset.treeTarget
      ? null
      : button.dataset.treeTarget;
    if (treeTarget !== "active") treeFollow = false;
    updateTreeToolbar();
    scrollTreeToTarget({ force: true });
  });
});

document.getElementById("tree-follow").addEventListener("click", () => {
  if (treeTarget !== "active") return;
  treeFollow = !treeFollow;
  updateTreeToolbar();
  scrollTreeToTarget({ force: treeFollow });
});

function updateTreeToolbar() {
  document.querySelectorAll("[data-tree-target]").forEach((button) => {
    button.classList.toggle("active", treeTarget === button.dataset.treeTarget);
  });
  const follow = document.getElementById("tree-follow");
  follow.disabled = treeTarget !== "active";
  follow.classList.toggle("active", treeTarget === "active" && treeFollow);
}

function text(value) {
  return value == null ? "" : String(value);
}

function useCompactTreePrefix() {
  return window.matchMedia("(max-width: 767px)").matches;
}

function treePrefix(line) {
  if (useCompactTreePrefix()) return text(line.prefix);
  return text(line.desktop_prefix || line.prefix);
}

function renderTree(snapshot) {
  document.getElementById("tree-title").textContent =
    snapshot.tree_title || "Solution tree";
  const tree = document.getElementById("tree");
  tree.replaceChildren();
  const lines = snapshot.tree_lines || [];
  if (!lines.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "waiting for tree";
    tree.appendChild(empty);
    return;
  }
  for (const line of lines) {
    const row = document.createElement("div");
    row.className = `tree-line ${line.kind || "ok"}`;
    const prefix = document.createElement("span");
    prefix.className = "prefix";
    prefix.textContent = treePrefix(line);
    const dot = document.createElement("span");
    dot.className = "dot";
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = text(line.label);
    row.append(prefix, dot, label);
    tree.appendChild(row);
  }
  scrollTreeToTarget({ force: treeFollow });
}

function scrollTreeToTarget({ force = false } = {}) {
  if (!treeTarget) {
    lastTreeTarget = null;
    return;
  }
  const selector = treeTarget === "best" ? ".tree-line.best" : ".tree-line.active";
  const target = document.querySelector(selector);
  if (!target) return;
  const targetText = target.textContent;
  if (!force && lastTreeTarget === `${treeTarget}:${targetText}`) return;
  target.scrollIntoView({ block: "center", inline: "nearest" });
  lastTreeTarget = `${treeTarget}:${targetText}`;
}

function renderRunData(snapshot) {
  const list = document.getElementById("run-data");
  list.replaceChildren();
  const legacyItems = snapshot.run_data || [];
  const sections = (snapshot.run_sections && snapshot.run_sections.length)
    ? snapshot.run_sections
    : legacyRunSections(legacyItems);
  if (!sections.some((section) => (section.items || []).length)) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "waiting for run data";
    list.appendChild(empty);
    return;
  }
  for (const section of sections) {
    const sectionEl = document.createElement("section");
    sectionEl.className = "run-section";
    const title = document.createElement("div");
    title.className = "section-title";
    title.textContent = text(section.title);
    sectionEl.appendChild(title);
    for (const item of section.items || []) {
      const row = document.createElement("div");
      row.className = "datum";
      const label = document.createElement("div");
      label.className = "datum-label";
      label.textContent = text(item.label);
      const value = document.createElement("div");
      value.className = "datum-value";
      value.textContent = text(item.value);
      row.append(label, value);
      sectionEl.appendChild(row);
    }
    list.appendChild(sectionEl);
  }
}

function legacyRunSections(items) {
  const buckets = {
    Run: [],
    Models: [],
    Agent: [],
    Paths: [],
    "Last Error": [],
    Notice: [],
  };
  for (const item of items || []) {
    const label = text(item.label);
    const lower = label.toLowerCase();
    if (lower.startsWith("model ")) {
      buckets.Models.push({
        label: label.replace(/^model\s+/i, ""),
        value: item.value,
      });
    } else if (["mode", "gpu", "aux"].includes(lower)) {
      buckets.Agent.push({ label: lower, value: item.value });
    } else if (["log dir", "workspace", "artifact"].includes(lower)) {
      buckets.Paths.push({ label: lower.replace(" dir", ""), value: item.value });
    } else if (lower === "last error") {
      buckets["Last Error"].push({ label: "error", value: item.value });
    } else if (lower === "notice") {
      buckets.Notice.push({ label: "message", value: item.value });
    } else {
      buckets.Run.push({ label: lower, value: item.value });
    }
  }
  return Object.entries(buckets).map(([title, sectionItems]) => ({
    title,
    items: sectionItems,
  }));
}

function renderLogs(snapshot) {
  const logs = document.getElementById("logs");
  logs.textContent = (snapshot.log_lines || []).join("\n") || "waiting for process log";
}

function formatTokens(value) {
  return Number(value || 0).toLocaleString("en-US");
}

function shortId(value) {
  return text(value).slice(0, 8);
}

function useCompactTokenTable() {
  return window.matchMedia("(max-width: 767px)").matches;
}

function tokenAgent(agent) {
  if (!useCompactTokenTable()) return text(agent);
  return agent === "code" ? "C" : agent === "feedback" ? "F" : text(agent);
}

function tokenAction(action) {
  if (!useCompactTokenTable()) return text(action);
  return { start: "▶", resume: "↻", fork: "⑂" }[action] || text(action);
}

function renderTokenUsage(snapshot) {
  const title = document.getElementById("token-title");
  title.textContent = `Codex token usage · ${text(snapshot.run_id)}`;
  const table = document.getElementById("token-table");
  const body = table.querySelector("tbody");
  const empty = document.getElementById("token-empty");
  const rows = snapshot.token_usage || [];
  body.replaceChildren();
  table.hidden = rows.length === 0;
  empty.hidden = rows.length > 0;
  let previousStep = null;
  let stepRank = -1;
  for (const row of rows) {
    if (row.step !== previousStep) stepRank += 1;
    const tr = document.createElement("tr");
    tr.dataset.step = text(row.step);
    tr.classList.toggle("step-alt", stepRank % 2 === 1);
    const values = [
      row.step,
      tokenAgent(row.agent),
      shortId(row.thread_id),
      shortId(row.turn_id),
      tokenAction(row.action),
      formatTokens(row.input_tokens),
      formatTokens(row.cached_input_tokens),
      formatTokens(row.input_tokens - row.cached_input_tokens),
      formatTokens(row.output_tokens),
      formatTokens(row.turn_total_tokens),
      formatTokens(row.thread_total_tokens),
    ];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      cell.textContent = text(value);
      if (index === 2) cell.title = text(row.thread_id);
      if (index === 3) cell.title = text(row.turn_id);
      tr.appendChild(cell);
    });
    body.appendChild(tr);
    previousStep = row.step;
  }
}

async function refresh() {
  try {
    const response = await fetch("/api/snapshot", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const snapshot = await response.json();
    lastSnapshot = snapshot;
    setLogsConnectionError(false);
    refreshMs = Math.max(
      500,
      Math.min(
        30000,
        Number(snapshot.refresh_seconds || refreshMs / 1000) * 1000,
      ),
    );
    renderTree(snapshot);
    renderRunData(snapshot);
    renderLogs(snapshot);
    if (lastTokenUsageUpdatedAt !== snapshot.token_usage_updated_at) {
      renderTokenUsage(snapshot);
      lastTokenUsageUpdatedAt = snapshot.token_usage_updated_at;
    }
  } catch (error) {
    setLogsConnectionError(true);
    document.getElementById("logs").textContent = `dashboard refresh failed: ${error}`;
  } finally {
    setTimeout(refresh, refreshMs);
  }
}

window.addEventListener("resize", () => {
  if (lastSnapshot) {
    renderTree(lastSnapshot);
    renderTokenUsage(lastSnapshot);
  }
});
updateVisibleTabs("tree");
refresh();
