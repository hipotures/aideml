let refreshMs = 2000;

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.panel === name);
  });
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => switchTab(tab.dataset.tab));
});

function text(value) {
  return value == null ? "" : String(value);
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
    prefix.textContent = text(line.prefix);
    const dot = document.createElement("span");
    dot.className = "dot";
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = text(line.label);
    row.append(prefix, dot, label);
    tree.appendChild(row);
  }
}

function renderRunData(snapshot) {
  const list = document.getElementById("run-data");
  list.replaceChildren();
  const items = snapshot.run_data || [];
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.textContent = "waiting for run data";
    list.appendChild(empty);
    return;
  }
  for (const item of items) {
    const row = document.createElement("div");
    row.className = "datum";
    const label = document.createElement("div");
    label.className = "datum-label";
    label.textContent = text(item.label);
    const value = document.createElement("div");
    value.className = "datum-value";
    value.textContent = text(item.value);
    row.append(label, value);
    list.appendChild(row);
  }
}

function renderLogs(snapshot) {
  const logs = document.getElementById("logs");
  logs.textContent = (snapshot.log_lines || []).join("\n") || "waiting for process log";
}

async function refresh() {
  try {
    const response = await fetch("/api/snapshot", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const snapshot = await response.json();
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
  } catch (error) {
    document.getElementById("logs").textContent = `dashboard refresh failed: ${error}`;
  } finally {
    setTimeout(refresh, refreshMs);
  }
}

refresh();
