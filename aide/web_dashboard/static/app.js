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
