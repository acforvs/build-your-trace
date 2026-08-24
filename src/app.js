const app = document.querySelector("#app");
const methodDialog = document.querySelector("#method-dialog");
const methodContent = document.querySelector("#method-content");
const progressKey = "build-your-trace:completed:v1";
const comparisonProgressKey = "build-your-trace:comparisons-completed:v1";
const comparisonJourneyKey = "build-your-trace:comparison-journeys:v1";
const collectiveProgressKey = "build-your-trace:collectives-completed:v1";
const collectivePlacementKey = "build-your-trace:collective-placements:v1";
const chapterStateKey = "build-your-trace:chapter-state:v1";
const briefCollapsedKey = "build-your-trace:brief-collapsed:v1";
const scrollPositionKey = "build-your-trace:scroll-positions:v1";
const themeKey = "build-your-trace:theme:v1";
const placementKey = "build-your-trace:placements:v1";
const taskTimelineLabelWidth = 128;
const state = { tasks: [], comparisons: [], collectives: [], assumptions: {}, resources: {}, social: {}, activeBlock: null, keyboardOrigin: null, placements: new Map(), completed: new Set(), comparisonsCompleted: new Set(), collectivesCompleted: new Set(), solutionRevealed: false, preRevealPlacements: null, activeTask: null, timelineZoom: 1, timelineNaturalWidth: 0, timelineAtFit: false, timelineZoomController: null, comparisonZoomControllers: [], focusedComparisonSide: null, suppressPlacementSave: false, activeDebriefIndex: 0, cancelInteraction: null };
let activeRouteKey = null;
let scrollTrackingReady = false;
let pendingChallengeScroll = false;

if ("scrollRestoration" in history) history.scrollRestoration = "manual";

function routeKey() {
  return location.hash || "#/";
}

function readScrollPositions() {
  try { return JSON.parse(sessionStorage.getItem(scrollPositionKey) || "{}"); } catch { return {}; }
}

function saveScrollPosition(key = activeRouteKey) {
  if (!scrollTrackingReady || !key) return;
  const positions = readScrollPositions();
  positions[key] = window.scrollY;
  try { sessionStorage.setItem(scrollPositionKey, JSON.stringify(positions)); } catch {}
}

function positionRoute(key, restore) {
  const top = restore ? Number(readScrollPositions()[key] || 0) : 0;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    const previousScrollBehavior = document.documentElement.style.scrollBehavior;
    document.documentElement.style.scrollBehavior = "auto";
    if (pendingChallengeScroll && key === "#/") {
      pendingChallengeScroll = false;
      document.querySelector("#challenges")?.scrollIntoView();
    } else {
      window.scrollTo(0, top);
    }
    document.documentElement.style.scrollBehavior = previousScrollBehavior;
    scrollTrackingReady = true;
  }));
}

function setTheme(theme, persist = true) {
  const isDark = theme === "dark";
  document.documentElement.dataset.theme = isDark ? "dark" : "light";
  const toggle = document.querySelector("[data-theme-toggle]");
  if (toggle) {
    toggle.setAttribute("aria-pressed", String(isDark));
    toggle.setAttribute("aria-label", `Switch to ${isDark ? "day" : "night"} mode`);
    toggle.querySelector("[data-theme-icon]").textContent = isDark ? "☀" : "☾";
    toggle.querySelector("[data-theme-label]").textContent = isDark ? "Day" : "Night";
  }
  if (persist) {
    try { localStorage.setItem(themeKey, isDark ? "dark" : "light"); } catch {}
  }
}

setTheme(document.documentElement.dataset.theme || "light", false);

function githubConfig() {
  const github = state.social.github;
  return github?.enabled && github.url ? github : null;
}

function configureGithubLinks(root) {
  const github = githubConfig();
  root.querySelectorAll("[data-github-link]").forEach((link) => {
    link.hidden = !github;
    if (!github) return;
    link.href = github.url;
    const label = link.querySelector("[data-github-label]");
    if (label) label.textContent = github.label || "Star on GitHub";
  });
}

function completionGithubLink() {
  const github = githubConfig();
  if (!github) return "";
  return `<a class="completion-social" href="${escapeHtml(github.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(github.completionPrompt || "Found this useful? Star the project on GitHub.")} <span aria-hidden="true">↗</span></a>`;
}

const typeMeta = {
  attention: { label: "Attention", className: "attention" },
  mlp: { label: "MLP / experts", className: "mlp" },
  collective: { label: "Collective", className: "collective" },
  routing: { label: "Routing", className: "routing" },
  optimizer: { label: "Optimizer", className: "optimizer" },
  loss: { label: "Loss", className: "loss" },
  bubble: { label: "Bubble", className: "bubble" },
};

const traceTooltip = document.createElement("div");
traceTooltip.className = "trace-tooltip";
traceTooltip.hidden = true;
document.body.append(traceTooltip);

const escapeHtml = (value) => String(value)
  .replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;").replaceAll("'", "&#039;");

function bindBlockTooltip(element, block) {
  const setContent = () => {
    traceTooltip.innerHTML = `<strong>${escapeHtml(block.label)}</strong>${block.description ? `<span>${escapeHtml(block.description)}</span>` : ""}`;
  };
  const moveTooltip = (event) => {
    const width = 330;
    const left = Math.max(10, Math.min(window.innerWidth - width - 10, event.clientX + 14));
    const top = Math.max(10, Math.min(window.innerHeight - 120, event.clientY + 14));
    traceTooltip.style.left = `${left}px`;
    traceTooltip.style.top = `${top}px`;
  };
  const moveTooltipToElement = () => {
    const rect = element.getBoundingClientRect();
    const width = 330;
    traceTooltip.style.left = `${Math.max(10, Math.min(window.innerWidth - width - 10, rect.left))}px`;
    traceTooltip.style.top = `${Math.max(10, Math.min(window.innerHeight - 120, rect.bottom + 8))}px`;
  };
  element.addEventListener("mouseenter", (event) => {
    setContent();
    traceTooltip.hidden = false;
    moveTooltip(event);
  });
  element.addEventListener("mousemove", moveTooltip);
  element.addEventListener("mouseleave", () => {
    if (document.activeElement !== element) traceTooltip.hidden = true;
  });
  element.addEventListener("focus", () => {
    setContent();
    traceTooltip.hidden = false;
    moveTooltipToElement();
  });
  element.addEventListener("blur", () => { traceTooltip.hidden = true; });
}

async function loadData() {
  const response = await fetch("data/tasks.json");
  if (!response.ok) throw new Error("Could not load challenge data.");
  const data = await response.json();
  const taskRecords = await Promise.all((data.taskFiles || []).map(async (path) => {
    const taskResponse = await fetch(path);
    if (!taskResponse.ok) throw new Error(`Could not load task: ${path}`);
    return taskResponse.json();
  }));
  data.tasks.push(...taskRecords);
  const overrides = await Promise.all((data.taskOverrideFiles || []).map(async (path) => {
    const overrideResponse = await fetch(path);
    if (!overrideResponse.ok) throw new Error(`Could not load task override: ${path}`);
    return overrideResponse.json();
  }));
  overrides.forEach((override) => {
    const task = data.tasks.find((item) => item.id === override.id);
    if (!task) throw new Error(`Task override references unknown task: ${override.id}`);
    Object.assign(task, override);
  });
  const variantRecords = await Promise.all((data.taskVariantFiles || []).map(async (path) => {
    const variantResponse = await fetch(path);
    if (!variantResponse.ok) throw new Error(`Could not load task variant: ${path}`);
    return variantResponse.json();
  }));
  variantRecords.forEach((record) => {
    const task = data.tasks.find((item) => item.id === record.taskId);
    if (!task) throw new Error(`Task variant references unknown task: ${record.taskId}`);
    task.variants = [...(task.variants || []), record.variant];
  });
  if (data.teachingFile) {
    const teachingResponse = await fetch(data.teachingFile);
    if (!teachingResponse.ok) throw new Error(`Could not load teaching content: ${data.teachingFile}`);
    const teaching = await teachingResponse.json();
    Object.entries(teaching).forEach(([taskId, content]) => {
      const task = data.tasks.find((item) => item.id === taskId);
      if (!task) throw new Error(`Teaching content references unknown task: ${taskId}`);
      Object.assign(task, content);
    });
  }
  if (data.comparisonFile) {
    const comparisonResponse = await fetch(data.comparisonFile);
    if (!comparisonResponse.ok) throw new Error(`Could not load comparison labs: ${data.comparisonFile}`);
    state.comparisons = ((await comparisonResponse.json()).comparisons || []).sort((left, right) => left.order - right.order);
  }
  if (data.collectiveFile) {
    const collectiveResponse = await fetch(data.collectiveFile);
    if (!collectiveResponse.ok) throw new Error(`Could not load collective lessons: ${data.collectiveFile}`);
    state.collectives = (await collectiveResponse.json()).lessons || [];
  }
  state.tasks = data.tasks;
  state.assumptions = data.assumptions;
  state.resources = data.resources || {};
  state.social = data.social || {};
  try { state.completed = new Set(JSON.parse(localStorage.getItem(progressKey) || "[]")); } catch { state.completed = new Set(); }
  try { state.comparisonsCompleted = new Set(JSON.parse(localStorage.getItem(comparisonProgressKey) || "[]")); } catch { state.comparisonsCompleted = new Set(); }
  try { state.collectivesCompleted = new Set(JSON.parse(localStorage.getItem(collectiveProgressKey) || "[]")); } catch { state.collectivesCompleted = new Set(); }
}

function renderMethod() {
  const { hardware, workload } = state.assumptions;
  methodContent.innerHTML = `
    <div class="method-intro">
      <strong>Two kinds of timing are used.</strong>
      <p>Measured tasks use captured GPU intervals or documented aggregate envelopes. Analytical variants keep measured compute fixed and rescale the named collectives with the assumptions shown below.</p>
    </div>
    <div class="method-formulas">
      <div><span>Analytical compute or memory</span><code>t = max(FLOPs / ηF, bytes / ηB)</code></div>
      <div><span>Analytical collective</span><code>t ≈ bytes per rank / effective fabric BW + latency</code></div>
    </div>
    <div class="method-section-heading"><span>Reference hardware</span><small>H100 SXM · effective rates are lower than peak</small></div>
    <div class="assumption-grid">
      ${Object.entries(hardware).map(([key, item]) => `<article><span>${escapeHtml(key)}</span><strong>${escapeHtml(item.value)}</strong></article>`).join("")}
    </div>
    <div class="method-columns">
      <div><h3>Reference shapes and conventions</h3><dl>${Object.entries(workload).map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl></div>
      <div><h3>Read widths within one trace</h3><p>Measured tasks use captured GPU intervals or grouped kernel envelopes. Analytical variants rescale only the communication named in the task.</p><p>Treat widths as local to each trace; software, topology, and contention change them.</p></div>
    </div>`;
}

function difficultyStars(value) {
  const descriptions = {
    1: "One dependency chain",
    2: "One parallel primitive",
    3: "Multiple interacting dependencies",
    4: "Multi-stream scheduling or backward ownership",
    5: "Composed parallelisms and critical-path reasoning",
  };
  return `<span class="stars" aria-label="Difficulty ${value} out of 5: ${descriptions[value]}" title="${descriptions[value]}">${Array.from({ length: 5 }, (_, index) => `<i class="${index < value ? "filled" : ""}">★</i>`).join("")}</span>`;
}

function cardFamily(tags = []) {
  if (tags.includes("MoE") || tags.includes("Expert Parallel")) return "moe";
  if (tags.includes("FSDP")) return "fsdp";
  if (tags.includes("Tensor Parallel") || tags.includes("Sequence Parallel")) return "tp";
  if (tags.includes("Context Parallel")) return "cp";
  if (tags.includes("Data Parallel")) return "dp";
  return "fundamentals";
}

function highlightCode(code) {
  const tokenPattern = /(#[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\b(?:def|for|in|if|else|with|return|await|range)\b|\b(?:all_gather|all_gather_tensor|reduce_scatter|reduce_scatter_async|all_reduce|all_to_all|all_to_all_single_autograd|permute_tensor|peer_fetch_async|sub_gemm|route_tokens|permute|unpermute|grouped_swiglu|flash_attention|online_merge|rms_norm|backward|prefetch|wait|router|qkv|local_qkv|silu|concat|scatter_weighted)\b)/g;
  return String(code).split(tokenPattern).map((token) => {
    if (!token) return "";
    let className = "";
    if (token.startsWith("#")) className = "code-comment";
    else if (token.startsWith('"') || token.startsWith("'")) className = "code-string";
    else if (/^(def|for|in|if|else|with|return|await|range)$/.test(token)) className = "code-keyword";
    else if (/^(all_gather|all_gather_tensor|reduce_scatter|reduce_scatter_async|all_reduce|all_to_all|all_to_all_single_autograd|permute_tensor|peer_fetch_async|sub_gemm|route_tokens|permute|unpermute|grouped_swiglu|flash_attention|online_merge|rms_norm|backward|prefetch|wait|router|qkv|local_qkv|silu|concat|scatter_weighted)$/.test(token)) className = "code-operation";
    const escaped = escapeHtml(token);
    return className ? `<span class="${className}">${escaped}</span>` : escaped;
  }).join("");
}

function renderImplementationSketch(sketch) {
  if (!sketch?.code) return "";
  const code = Array.isArray(sketch.code) ? sketch.code.join("\n") : sketch.code;
  return `<figure class="implementation-card"><figcaption><span>Implementation sketch</span><strong>${escapeHtml(sketch.label)}</strong></figcaption><pre><code>${highlightCode(code)}</code></pre>${sketch.note ? `<p>${escapeHtml(sketch.note)}</p>` : ""}</figure>`;
}

function bindChapterPanels() {
  const panels = [...app.querySelectorAll("[data-chapter-panel]")];
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(chapterStateKey) || "{}"); } catch {}
  panels.forEach((panel) => {
    if (typeof saved[panel.dataset.chapterPanel] === "boolean") panel.open = saved[panel.dataset.chapterPanel];
    panel.addEventListener("toggle", () => {
      const nextState = Object.fromEntries(panels.map((item) => [item.dataset.chapterPanel, item.open]));
      try { localStorage.setItem(chapterStateKey, JSON.stringify(nextState)); } catch {}
    });
  });
}

function renderHome() {
  const fragment = document.querySelector("#home-template").content.cloneNode(true);
  const catalogTasks = state.tasks
    .filter((task) => !task.catalogHidden)
    .sort((left, right) => left.order - right.order);
  const verifiedCount = catalogTasks.filter((task) => task.verification?.status === "measured").length;
  fragment.querySelector("[data-task-count]").textContent = `${verifiedCount} trace builds · ${state.comparisons.length} comparison labs`;
  const filterOrder = ["MoE", "Data Parallel", "Tensor Parallel", "Sequence Parallel", "FSDP", "Expert Parallel", "Context Parallel"];
  const filters = ["All", ...filterOrder.filter((filter) => catalogTasks.some((task) => task.catalogTags?.includes(filter)))];
  const filterRow = fragment.querySelector(".filter-row");
  const grid = fragment.querySelector(".task-grid");
  const taskSectionMeta = new Map([
    ["I · Execution fundamentals", { label: "Execution fundamentals", id: "execution-fundamentals" }],
    ["II · Data parallelism", { label: "Data parallelism", id: "data-parallelism" }],
    ["III · Tensor parallelism", { label: "Tensor parallelism", id: "tensor-parallelism" }],
    ["IV · Sharding and overlap", { label: "Sharding and overlap", id: "sharding-and-overlap" }],
    ["V · Context parallelism", { label: "Context parallelism", id: "context-parallelism" }],
  ]);

  const drawCards = (filter = "All") => {
    const visibleTasks = catalogTasks.filter((task) => filter === "All" || task.catalogTags?.includes(filter));
    const sections = filter === "All" ? [...new Set(visibleTasks.map((task) => task.section || "Challenges"))] : [filter];
    grid.innerHTML = sections.map((section) => {
      const sectionTasks = visibleTasks.filter((task) => filter !== "All" || (task.section || "Challenges") === section);
      if (!sectionTasks.length) return "";
      const sectionMeta = taskSectionMeta.get(section);
      const sectionLabel = sectionMeta ? `1.${[...taskSectionMeta.keys()].indexOf(section) + 1} · ${sectionMeta.label}` : section;
      return `<section class="task-section"${sectionMeta ? ` id="${sectionMeta.id}"` : ""}><div class="task-section-heading"><span>${escapeHtml(sectionLabel)}</span><small>${sectionTasks.length} challenge${sectionTasks.length === 1 ? "" : "s"}</small></div><div class="task-section-grid">${sectionTasks.map((task, index) => {
      const measured = task.verification?.status === "measured";
      const tag = measured ? "a" : "article";
      const placementCount = task.blocks.filter((block) => !block.fixed).length;
      return `
      <${tag} class="task-card${state.completed.has(task.id) ? " completed" : ""}${measured ? "" : " pending"}" data-task-id="${escapeHtml(task.id)}" data-card-family="${cardFamily(task.catalogTags)}"${measured ? ` href="#/task/${task.id}"` : ""} style="--delay:${index * 45}ms">
        <div class="card-top"><span class="task-number"><b>${String(task.order).padStart(2, "0")}</b><small>Challenge</small></span>${state.completed.has(task.id) ? '<span class="completed-badge">Completed <b>✓</b></span>' : ""}</div>
        <div class="card-copy"><h3>${escapeHtml(task.title)}</h3>
        <p>${measured ? escapeHtml(task.summary) : "Trace calibration is not complete."}</p></div>
        <div class="card-tags">${(task.catalogTags || [task.category]).map((catalogTag) => `<span>${escapeHtml(catalogTag)}</span>`).join("")}</div>
        <div class="card-footer"><span class="card-difficulty"><small>Difficulty</small>${difficultyStars(task.difficulty)}</span><span class="card-cta">${measured ? `<small>${placementCount} block${placementCount === 1 ? "" : "s"} to place</small><b>Open <i>→</i></b>` : "Calibration pending"}</span></div>
      </${tag}>`;
    }).join("")}</div></section>`;
    }).join("");
  };

  filters.forEach((filter, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `filter-button${index === 0 ? " active" : ""}`;
    const count = filter === "All" ? catalogTasks.length : catalogTasks.filter((task) => task.catalogTags?.includes(filter)).length;
    button.innerHTML = `${escapeHtml(filter)}<small>${count}</small>`;
    button.addEventListener("click", () => {
      filterRow.querySelectorAll("button").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      drawCards(filter);
    });
    filterRow.append(button);
  });
  drawCards();
  const collectiveGrid = fragment.querySelector("[data-collective-grid]");
  collectiveGrid.innerHTML = state.collectives
    .sort((left, right) => left.order - right.order)
    .map((lesson) => `<a class="collective-card${state.collectivesCompleted.has(lesson.id) ? " completed" : ""}" data-card-family="collective" href="#/collective/${escapeHtml(lesson.id)}">
      <div class="comparison-card-top"><span>Primitive ${String(lesson.order).padStart(2, "0")}</span>${state.collectivesCompleted.has(lesson.id) ? "<b>Completed ✓</b>" : ""}</div>
      <h3>${escapeHtml(lesson.title)}</h3>
      <p>${escapeHtml(lesson.summary)}</p>
      <div class="card-footer"><span>${lesson.ranks} ranks</span><b class="card-entry">Arrange tensors <i>→</i></b></div>
    </a>`).join("");
  const comparisonGrid = fragment.querySelector("[data-comparison-grid]");
  const comparisonSections = [...new Set(state.comparisons.map((comparison) => comparison.section || "Explain dependencies"))];
  comparisonGrid.innerHTML = comparisonSections.map((section) => `<section class="comparison-group"><div class="task-section-heading"><span>${escapeHtml(section)}</span><small>${state.comparisons.filter((comparison) => (comparison.section || "Explain dependencies") === section).length} labs</small></div><div class="comparison-grid">${state.comparisons
    .filter((comparison) => (comparison.section || "Explain dependencies") === section)
    .map((comparison) => `<a class="comparison-card${state.comparisonsCompleted.has(comparison.id) ? " completed" : ""}" data-card-family="${cardFamily(comparison.tags)}" href="#/compare/${escapeHtml(comparison.id)}">
      <div class="comparison-card-top"><span>Lab ${String(comparison.order).padStart(2, "0")}</span>${state.comparisonsCompleted.has(comparison.id) ? "<b>Completed ✓</b>" : ""}</div>
      <h3>${escapeHtml(comparison.title)}</h3>
      <p>${escapeHtml(comparison.summary)}</p>
      <div class="card-tags">${comparison.tags.map((tag) => `<span>${escapeHtml(tag)}</span>`).join("")}</div>
      <div class="card-footer"><span>${comparison.questions.length} reasoning steps</span><b class="card-entry">Open lab <i>→</i></b></div>
    </a>`).join("")}</div></section>`).join("");
  const contents = [
    { label: "Collective primer", count: state.collectives.length, panel: "collectives", target: "collective-primer" },
    ...[...taskSectionMeta.entries()].map(([section, meta]) => ({
      label: meta.label,
      count: catalogTasks.filter((task) => task.section === section).length,
      panel: "build",
      target: meta.id,
      taskSection: true,
    })),
    { label: "Comparison labs", count: state.comparisons.length, panel: "compare", target: "explain-trace" },
  ];
  fragment.querySelector("[data-contents-links]").innerHTML = contents.map((item) => `<button type="button" data-contents-panel="${item.panel}" data-contents-target="${item.target}"${item.taskSection ? " data-task-section" : ""}><span>${escapeHtml(item.label)}</span><small>${item.count}</small></button>`).join("");
  fragment.querySelector("[data-build-count]").textContent = `${catalogTasks.length} challenges`;
  fragment.querySelector("[data-comparison-count]").textContent = `${state.comparisons.length} labs`;
  fragment.querySelector("[data-collective-count]").textContent = `${state.collectives.length} primitives`;
  configureGithubLinks(fragment);
  app.replaceChildren(fragment);
  bindChapterPanels();
  app.querySelectorAll("[data-contents-target]").forEach((button) => button.addEventListener("click", () => {
    const panel = app.querySelector(`[data-chapter-panel="${button.dataset.contentsPanel}"]`);
    if (panel) panel.open = true;
    if (button.hasAttribute("data-task-section")) {
      filterRow.querySelectorAll("button").forEach((item, index) => item.classList.toggle("active", index === 0));
      drawCards();
    }
    requestAnimationFrame(() => document.getElementById(button.dataset.contentsTarget)?.scrollIntoView({ behavior: "smooth", block: "start" }));
  }));
  document.title = "Build Your Trace — Interactive LLM profiling puzzles";
}

function resolveTaskVariant(task, variantId) {
  const variant = task.variants?.find((item) => item.id === variantId) || task.variants?.find((item) => item.default) || null;
  if (!variant) return { ...task, sourceTask: task, activeVariant: null };
  const blockOverrides = new Map(Object.entries(variant.blockOverrides || {}));
  return {
    ...task,
    ...(variant.taskOverrides || {}),
    verification: { ...task.verification, ...(variant.verification || {}) },
    blocks: task.blocks.map((block) => ({ ...block, ...(blockOverrides.get(block.id) || {}) })).filter((block) => !block.hidden),
    sourceTask: task,
    activeVariant: variant,
  };
}

function placementStorageId(task) {
  return `${task.id}:${task.activeVariant?.id || "base"}`;
}

function readPlacementStore() {
  try { return JSON.parse(localStorage.getItem(placementKey) || "{}"); } catch { return {}; }
}

function savePlacements(task = state.activeTask) {
  if (!task || state.suppressPlacementSave || state.solutionRevealed) return;
  const store = readPlacementStore();
  const id = placementStorageId(task);
  if (state.placements.size) store[id] = Object.fromEntries(state.placements);
  else delete store[id];
  try { localStorage.setItem(placementKey, JSON.stringify(store)); } catch {}
}

function clearStoredPlacements(task) {
  const store = readPlacementStore();
  delete store[placementStorageId(task)];
  try { localStorage.setItem(placementKey, JSON.stringify(store)); } catch {}
}

function restorePlacements(task) {
  const saved = readPlacementStore()[placementStorageId(task)] || {};
  state.suppressPlacementSave = true;
  for (const [blockId, slotId] of Object.entries(saved)) {
    const block = task.blocks.find((item) => item.id === blockId && !item.fixed);
    const slot = app.querySelector(`[data-slot="${CSS.escape(slotId)}"]`);
    if (block && slot && !slot.dataset.occupied) placeBlock(blockId, slotId, task);
  }
  state.suppressPlacementSave = false;
}

function renderGame(sourceTask, variantId) {
  const task = resolveTaskVariant(sourceTask, variantId);
  state.activeTask = task;
  state.activeBlock = null;
  state.keyboardOrigin = null;
  state.placements = new Map();
  state.solutionRevealed = false;
  state.preRevealPlacements = null;
  state.cancelInteraction = () => {
    if (!state.activeBlock) return false;
    const origin = state.keyboardOrigin || app.querySelector("[data-block].selected");
    state.activeBlock = null;
    app.querySelectorAll("[data-block]").forEach((button) => button.classList.remove("selected"));
    app.querySelectorAll("[data-slot]").forEach((slot) => slot.classList.remove("selectable"));
    state.keyboardOrigin = null;
    updateKeyboardPlacement();
    requestAnimationFrame(() => origin?.focus());
    return true;
  };
  const fragment = document.querySelector("#game-template").content.cloneNode(true);
  const gamePage = fragment.querySelector(".game-page");
  gamePage.dataset.taskFamily = cardFamily(task.catalogTags || [task.category]);
  gamePage.classList.toggle("compact-trace", task.tracks.length <= 2);
  fragment.querySelector("[data-task-kicker]").textContent = `Challenge ${String(task.order).padStart(2, "0")} · ${task.category}`;
  fragment.querySelector("[data-task-title]").textContent = task.title;
  fragment.querySelector("[data-task-objective]").textContent = task.objective || task.description?.[0] || task.summary;
  fragment.querySelector("[data-task-insight]").textContent = task.insight;
  fragment.querySelector("[data-task-hint]").textContent = task.hint;
  const teachingPoints = task.teachingPoints || [];
  fragment.querySelector("[data-task-teaching-points]").innerHTML = teachingPoints.length
    ? `<section class="teaching-points"><ul>${teachingPoints.map((point) => `<li>${escapeHtml(point)}</li>`).join("")}</ul></section>`
    : "";
  fragment.querySelector("[data-task-implementation]").innerHTML = renderImplementationSketch(task.implementationSketch);
  fragment.querySelector("[data-block-count]").textContent = task.blocks.filter((block) => !block.fixed).length;
  fragment.querySelector("[data-timing-source]").textContent = task.verification?.status === "modeled" ? "analytical communication model" : "measured timing";
  const timingLabel = task.verification?.status === "modeled" ? "Analytical model" : "Measured GPU timing";
  fragment.querySelector(".task-facts").innerHTML = `<div><dt>Difficulty</dt><dd>${difficultyStars(task.difficulty)}</dd></div><div><dt>Topology</dt><dd>${escapeHtml(task.topology)}</dd></div><div><dt>Scope</dt><dd>${escapeHtml(task.scope)}</dd></div><div><dt>Timing</dt><dd><span class="task-timing-badge">${timingLabel}</span></dd></div>`;
  const concepts = task.concepts || task.glossary?.slice(0, 3) || [];
  if (concepts.length) {
    const glossary = document.createElement("details");
    glossary.className = "glossary-panel";
    glossary.innerHTML = `<summary>Concepts</summary><dl>${concepts.map((item) => `<div><dt>${escapeHtml(item.term)}</dt><dd>${escapeHtml(item.definition)}</dd></div>`).join("")}</dl>`;
    fragment.querySelector(".brief-scroll").append(glossary);
  }
  const resources = (task.resourceIds || []).map((id) => state.resources[id]).filter(Boolean);
  if (resources.length) {
    const resourcePanel = document.createElement("div");
    resourcePanel.className = "resource-panel";
    resourcePanel.innerHTML = `<details><summary>References</summary><div>${resources.map((resource) => `<a href="${escapeHtml(resource.url)}" target="_blank" rel="noreferrer">${escapeHtml(resource.label)} ↗</a>`).join("")}</div></details>`;
    fragment.querySelector(".brief-scroll").append(resourcePanel);
  }

  const usedTypes = [...new Set(task.blocks.map((block) => block.type))];
  fragment.querySelector("[data-legend]").innerHTML = `${usedTypes.map((type) => `<span><i class="${typeMeta[type].className}"></i>${typeMeta[type].label}</span>`).join("")}<span><i class="micro-event-key"></i>Tiny event — focus or tap</span><span><i class="empty-target-key"></i>Empty target</span>`;
  app.replaceChildren(fragment);
  document.title = `${task.title} — Build Your Trace`;
  buildTimeline(task);
  buildVariantPicker(task);
  buildPalette(task);
  placeFixedBlocks(task);
  restorePlacements(task);
  bindGameActions(task);
  bindBriefToggle();
  bindTimelineZoom(task);
  updateFocusButton(document.body.classList.contains("trace-focus"));
  if (state.completed.has(task.id) && traceIsSolved(task)) requestAnimationFrame(() => showCompletionDebrief(task));
}

function setBriefCollapsed(collapsed) {
  const gamePage = app.querySelector(".game-page");
  const toggle = app.querySelector("[data-brief-toggle]");
  if (!gamePage || !toggle) return;
  gamePage.classList.toggle("brief-collapsed", collapsed);
  toggle.setAttribute("aria-expanded", String(!collapsed));
  toggle.setAttribute("aria-label", collapsed ? "Expand task brief" : "Collapse task brief");
  toggle.textContent = collapsed ? "›" : "‹";
  try { localStorage.setItem(briefCollapsedKey, String(collapsed)); } catch {}
}

function bindBriefToggle() {
  const toggle = app.querySelector("[data-brief-toggle]");
  if (!toggle) return;
  let collapsed = false;
  try { collapsed = localStorage.getItem(briefCollapsedKey) === "true"; } catch {}
  setBriefCollapsed(collapsed);
  toggle.addEventListener("click", () => {
    setBriefCollapsed(!app.querySelector(".game-page").classList.contains("brief-collapsed"));
  });
}

function buildVariantPicker(task) {
  const picker = app.querySelector("[data-variant-picker]");
  const variants = task.sourceTask.variants || [];
  picker.hidden = variants.length < 2;
  variants.forEach((variant) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = variant.id === task.activeVariant?.id ? "active" : "";
    button.textContent = variant.label;
    button.addEventListener("click", () => renderGame(task.sourceTask, variant.id));
    picker.append(button);
  });
}

function buildTimeline(task) {
  if (task.timelineMode === "absolute") {
    buildAbsoluteTimeline(task);
    return;
  }
  const timeline = app.querySelector("[data-timeline]");
  const labelWidth = taskTimelineLabelWidth;
  const minimumPhaseMs = Math.min(...task.phases.map((phase) => phase.durationMs));
  const pxPerMs = Math.max(24, 880 / task.totalMs, 54 / minimumPhaseMs);
  const timelineWidth = Math.round(task.totalMs * pxPerMs);
  state.timelineNaturalWidth = timelineWidth;
  timeline.style.setProperty("--timeline-width", `${timelineWidth}px`);
  timeline.style.setProperty("--label-width", `${labelWidth}px`);

  const ruler = document.createElement("div");
  ruler.className = "ruler-row";
  ruler.innerHTML = `<div class="track-label">track</div><div class="ruler" style="width:var(--timeline-width)">${task.phases.map((phase, index) => `<div style="width:${phase.durationMs / task.totalMs * 100}%"><span>${index + 1}</span></div>`).join("")}</div>`;
  timeline.append(ruler);

  task.tracks.forEach((track) => {
    const row = document.createElement("div");
    row.className = "track-row";
    row.innerHTML = `<div class="track-label"><b>${escapeHtml(track.label)}</b><small>${escapeHtml(track.subtitle || "")}</small></div><div class="track-cells" style="width:var(--timeline-width)"></div>`;
    const cells = row.querySelector(".track-cells");
    task.phases.forEach((phase, phaseIndex) => {
      const slot = document.createElement("button");
      slot.type = "button";
      slot.className = "trace-slot";
      slot.style.width = `${phase.durationMs / task.totalMs * 100}%`;
      slot.dataset.slot = `${track.id}:${phase.id}`;
      slot.dataset.emptyLabel = `Empty target on ${track.label}, phase ${phaseIndex + 1}`;
      slot.setAttribute("aria-label", slot.dataset.emptyLabel);
      slot.innerHTML = `<span>drop</span>`;
      slot.addEventListener("dragover", (event) => { event.preventDefault(); slot.classList.add("drag-over"); });
      slot.addEventListener("dragleave", () => slot.classList.remove("drag-over"));
      slot.addEventListener("drop", (event) => {
        event.preventDefault();
        slot.classList.remove("drag-over");
        placeBlock(event.dataTransfer.getData("text/plain"), slot.dataset.slot, task);
      });
      slot.addEventListener("keydown", (event) => {
        if (state.activeBlock && (event.key === "Enter" || event.key === " ")) {
          event.preventDefault();
          placeBlock(state.activeBlock, slot.dataset.slot, task, { keyboard: true });
        }
      });
      slot.addEventListener("click", () => {
        if (state.activeBlock) placeBlock(state.activeBlock, slot.dataset.slot, task);
      });
      cells.append(slot);
    });
    timeline.append(row);
  });
}

function configureSlot(slot, task) {
  slot.addEventListener("dragover", (event) => { event.preventDefault(); slot.classList.add("drag-over"); });
  slot.addEventListener("dragleave", () => slot.classList.remove("drag-over"));
  slot.addEventListener("drop", (event) => {
    event.preventDefault();
    slot.classList.remove("drag-over");
    placeBlock(event.dataTransfer.getData("text/plain"), slot.dataset.slot, task);
  });
  slot.addEventListener("keydown", (event) => {
    if (state.activeBlock && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      placeBlock(state.activeBlock, slot.dataset.slot, task, { keyboard: true });
    }
  });
  slot.addEventListener("click", () => {
    if (state.activeBlock) placeBlock(state.activeBlock, slot.dataset.slot, task);
  });
}

function buildAbsoluteTimeline(task) {
  const timeline = app.querySelector("[data-timeline]");
  timeline.classList.add("absolute-timeline");
  const labelWidth = taskTimelineLabelWidth;
  const pxPerMs = task.pxPerMs || Math.max(24, 1100 / task.totalMs);
  const timelineWidth = Math.round(task.totalMs * pxPerMs);
  const gridSegments = task.gridSegments || 12;
  state.timelineNaturalWidth = timelineWidth;
  timeline.style.setProperty("--timeline-width", `${timelineWidth}px`);
  timeline.style.setProperty("--label-width", `${labelWidth}px`);

  const ruler = document.createElement("div");
  ruler.className = "ruler-row";
  ruler.innerHTML = `<div class="track-label">track</div><div class="ruler absolute-ruler" style="width:var(--timeline-width)">${Array.from({ length: gridSegments }, () => `<div style="width:${100 / gridSegments}%"><span>·</span></div>`).join("")}</div>`;
  timeline.append(ruler);

  task.tracks.forEach((track) => {
    const row = document.createElement("div");
    row.className = "track-row";
    row.innerHTML = `<div class="track-label"><b>${escapeHtml(track.label)}</b><small>${escapeHtml(track.subtitle || "")}</small></div><div class="track-cells absolute-track-cells" style="width:var(--timeline-width)"></div>`;
    const cells = row.querySelector(".track-cells");
    task.blocks.filter((block) => block.track === track.id).forEach((block) => {
      const slot = document.createElement("button");
      slot.type = "button";
      slot.className = "trace-slot absolute-slot";
      slot.style.left = `${block.startMs / task.totalMs * 100}%`;
      slot.style.width = `${block.durationMs / task.totalMs * 100}%`;
      slot.dataset.slot = block.id;
      slot.dataset.absolute = "true";
      const callout = getCalloutPresentation(task, block);
      const microTarget = block.durationMs * pxPerMs < 44;
      if (microTarget) {
        const hitOffsets = [-36, -18, 18, 36];
        slot.dataset.microTarget = "true";
        slot.dataset.hitLane = String(callout?.lane || 0);
        const hitOffset = hitOffsets[callout?.lane || 0];
        slot.classList.add("micro-target");
        slot.style.setProperty("--hit-offset", `${hitOffset}px`);
      }
      slot.dataset.emptyLabel = `Empty target on ${track.label}`;
      slot.setAttribute("aria-label", slot.dataset.emptyLabel);
      slot.innerHTML = "<span>drop</span>";
      configureSlot(slot, task);
      cells.append(slot);
    });
    timeline.append(row);
  });
}

function createWidthZoomController({
  naturalWidth,
  scrollElements,
  getFitWidth,
  applyWidth,
  fitButton,
  zoomOutButton,
  zoomInButton,
  levelElement,
  maxScale = 16,
  onChange = () => {},
}) {
  let currentWidth = naturalWidth;
  let atFit = false;
  const pointerPositions = new Map();
  const observer = new ResizeObserver(() => { if (atFit) fit(); });

  function setWidth(width, { fit: fitting = false, centerRatio = null, anchorPoints = null } = {}) {
    const fitWidth = getFitWidth();
    const nextWidth = Math.max(fitWidth, Math.min(naturalWidth * maxScale, Math.round(width)));
    const ratios = scrollElements.map((scroll) => centerRatio ?? ((scroll.scrollLeft + scroll.clientWidth / 2) / Math.max(1, scroll.scrollWidth)));
    currentWidth = nextWidth;
    atFit = fitting || Math.abs(nextWidth - fitWidth) < 2;
    const scale = nextWidth / naturalWidth;
    applyWidth(nextWidth, scale);
    levelElement.textContent = `${Math.round(scale * 100)}%`;
    zoomOutButton.disabled = nextWidth <= fitWidth + 2;
    zoomInButton.disabled = nextWidth >= naturalWidth * maxScale - 2;
    onChange({ scale, atFit });
    scrollElements.forEach((scroll, index) => {
      const anchorPoint = anchorPoints?.[index];
      if (anchorPoint) {
        // Reading scrollWidth after applyWidth forces current layout geometry,
        // preventing rapid pinch events from anchoring against a stale width.
        const fixedWidth = Math.max(0, scroll.scrollWidth - currentWidth);
        scroll.scrollLeft = Math.max(0, fixedWidth + anchorPoint.ratio * currentWidth - anchorPoint.pixel);
        return;
      }
      scroll.scrollLeft = Math.max(0, ratios[index] * scroll.scrollWidth - scroll.clientWidth / 2);
    });
  }

  function zoomBy(factor, anchorPixels = null) {
    const anchorPoints = scrollElements.map((scroll, index) => {
      const pixel = anchorPixels?.[index] ?? scroll.clientWidth / 2;
      const fixedWidth = Math.max(0, scroll.scrollWidth - currentWidth);
      const ratio = Math.max(0, Math.min(1, (scroll.scrollLeft + pixel - fixedWidth) / Math.max(1, currentWidth)));
      return { pixel, ratio };
    });
    setWidth(currentWidth * factor, { anchorPoints });
  }

  function fit() {
    setWidth(getFitWidth(), { fit: true, centerRatio: .5 });
  }

  fitButton.addEventListener("click", fit);
  zoomOutButton.addEventListener("click", () => setWidth(currentWidth / 1.35));
  zoomInButton.addEventListener("click", () => setWidth(currentWidth * 1.35));
  const wheelHandlers = scrollElements.map((scroll, index) => {
    let previousPinchDistance = null;
    const trackPointer = (event) => {
      const rect = scroll.getBoundingClientRect();
      pointerPositions.set(scroll, Math.max(0, Math.min(scroll.clientWidth, event.clientX - rect.left)));
    };
    const touchStart = (event) => {
      if (event.touches.length !== 2) return;
      const [first, second] = event.touches;
      previousPinchDistance = Math.hypot(second.clientX - first.clientX, second.clientY - first.clientY);
    };
    const touchMove = (event) => {
      if (event.touches.length !== 2) return;
      event.preventDefault();
      const [first, second] = event.touches;
      const distance = Math.hypot(second.clientX - first.clientX, second.clientY - first.clientY);
      if (!previousPinchDistance || !distance) {
        previousPinchDistance = distance;
        return;
      }
      const rect = scroll.getBoundingClientRect();
      const midpoint = (first.clientX + second.clientX) / 2 - rect.left;
      const pointer = Math.max(0, Math.min(scroll.clientWidth, midpoint));
      const anchorPixels = scrollElements.map((item, itemIndex) => itemIndex === index ? pointer : item.clientWidth / 2);
      zoomBy(Math.max(.86, Math.min(1.16, distance / previousPinchDistance)), anchorPixels);
      previousPinchDistance = distance;
    };
    const touchEnd = (event) => { if (event.touches.length < 2) previousPinchDistance = null; };
    const handler = (event) => {
      if (!event.ctrlKey && !event.altKey) return;
      event.preventDefault();
      const rect = scroll.getBoundingClientRect();
      const eventPointer = Math.max(0, Math.min(scroll.clientWidth, event.clientX - rect.left));
      const pointer = event.ctrlKey ? (pointerPositions.get(scroll) ?? eventPointer) : eventPointer;
      const anchorPixels = scrollElements.map((item, itemIndex) => itemIndex === index ? pointer : item.clientWidth / 2);
      zoomBy(Math.max(.88, Math.min(1.14, Math.exp(-event.deltaY * .004))), anchorPixels);
    };
    scroll.addEventListener("pointermove", trackPointer, { passive: true, capture: true });
    scroll.addEventListener("touchstart", touchStart, { passive: true, capture: true });
    scroll.addEventListener("touchmove", touchMove, { passive: false, capture: true });
    scroll.addEventListener("touchend", touchEnd, { passive: true, capture: true });
    scroll.addEventListener("touchcancel", touchEnd, { passive: true, capture: true });
    scroll.addEventListener("wheel", handler, { passive: false, capture: true });
    observer.observe(scroll);
    return { scroll, handler, trackPointer, touchStart, touchMove, touchEnd };
  });
  requestAnimationFrame(fit);
  return { fit, setWidth, zoomBy, destroy: () => {
    observer.disconnect();
    wheelHandlers.forEach(({ scroll, handler, trackPointer, touchStart, touchMove, touchEnd }) => {
      scroll.removeEventListener("pointermove", trackPointer, { capture: true });
      scroll.removeEventListener("touchstart", touchStart, { capture: true });
      scroll.removeEventListener("touchmove", touchMove, { capture: true });
      scroll.removeEventListener("touchend", touchEnd, { capture: true });
      scroll.removeEventListener("touchcancel", touchEnd, { capture: true });
      scroll.removeEventListener("wheel", handler, { capture: true });
    });
  } };
}

function focusTimelineBlocks(task, blockIds) {
  if (task.timelineMode !== "absolute" || !blockIds.length) return;
  const blocks = blockIds.map((id) => task.blocks.find((block) => block.id === id)).filter(Boolean);
  if (!blocks.length) return;
  const start = Math.min(...blocks.map((block) => block.startMs));
  const end = Math.max(...blocks.map((block) => block.startMs + block.durationMs));
  const padding = Math.max(.05, (end - start) * .12);
  const paddedStart = Math.max(0, start - padding);
  const paddedEnd = Math.min(task.totalMs, end + padding);
  const visibleRatio = Math.max(.04, (paddedEnd - paddedStart) / task.totalMs);
  const fitWidth = Math.max(120, app.querySelector(".timeline-scroll").clientWidth - taskTimelineLabelWidth - 2);
  const targetWidth = Math.min(state.timelineNaturalWidth * 1.5, fitWidth * .86 / visibleRatio);
  state.timelineZoomController?.setWidth(targetWidth, { centerRatio: (paddedStart + paddedEnd) / 2 / task.totalMs });
}

function bindTimelineZoom() {
  state.timelineZoomController?.destroy();
  const task = state.activeTask;
  const scroll = app.querySelector(".timeline-scroll");
  const timeline = app.querySelector("[data-timeline]");
  state.timelineZoomController = createWidthZoomController({
    naturalWidth: state.timelineNaturalWidth,
    scrollElements: [scroll],
    getFitWidth: () => Math.max(120, scroll.clientWidth - taskTimelineLabelWidth - 2),
    applyWidth: (width, scale) => {
      timeline.style.setProperty("--timeline-width", `${width}px`);
      timeline.classList.toggle("compact-zoom", scale < .62);
      timeline.querySelectorAll(".callout-slot").forEach((slot) => {
        const block = task?.blocks.find((item) => item.id === slot.dataset.slot);
        const renderedWidth = block && task.totalMs ? block.durationMs / task.totalMs * width : 0;
        slot.classList.toggle("callout-readable", renderedWidth >= 62);
      });
    },
    fitButton: app.querySelector("[data-zoom-fit]"),
    zoomOutButton: app.querySelector("[data-zoom-out]"),
    zoomInButton: app.querySelector("[data-zoom-in]"),
    levelElement: app.querySelector("[data-zoom-level]"),
    onChange: ({ scale, atFit }) => {
      state.timelineZoom = scale;
      state.timelineAtFit = atFit;
    },
  });
}

function traceIsSolved(task) {
  const playableBlocks = task.blocks.filter((block) => !block.fixed);
  return playableBlocks.length === state.placements.size
    && playableBlocks.every((block) => state.placements.get(block.id) === targetSlotId(task, block));
}

function debriefDependencies(task) {
  const blockIds = new Set(task.blocks.map((block) => block.id));
  return (task.debrief?.dependencies || []).filter((dependency) => blockIds.has(dependency.fromBlockId) && blockIds.has(dependency.toBlockId));
}

function debriefBlockElement(blockId) {
  const slot = app.querySelector(`[data-occupied="${CSS.escape(blockId)}"]`);
  return slot?.querySelector(".placed-block") || slot;
}

function hideCompletionDebrief() {
  const panel = app.querySelector("[data-completion-debrief]");
  if (panel) panel.hidden = true;
  app.querySelector(".timeline-shell")?.classList.remove("has-debrief");
  const timeline = app.querySelector("[data-timeline]");
  timeline?.classList.remove("debrief-active");
  timeline?.querySelectorAll(".debrief-source, .debrief-target").forEach((element) => element.classList.remove("debrief-source", "debrief-target"));
}

function selectDebriefDependency(task, index) {
  const dependencies = debriefDependencies(task);
  const dependency = dependencies[index];
  if (!dependency) return;
  state.activeDebriefIndex = index;
  app.querySelectorAll("[data-debrief-step]").forEach((button, buttonIndex) => {
    const active = buttonIndex === index;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  const fromBlock = task.blocks.find((block) => block.id === dependency.fromBlockId);
  const toBlock = task.blocks.find((block) => block.id === dependency.toBlockId);
  app.querySelector("[data-debrief-count]").textContent = `${index + 1} / ${dependencies.length}`;
  app.querySelector("[data-debrief-from]").textContent = fromBlock?.label || dependency.fromBlockId;
  app.querySelector("[data-debrief-to]").textContent = toBlock?.label || dependency.toBlockId;
  app.querySelector("[data-debrief-relation]").textContent = dependency.label;
  app.querySelector("[data-debrief-explanation]").textContent = dependency.explanation;
  app.querySelector("[data-debrief-previous]").disabled = index === 0;
  app.querySelector("[data-debrief-next]").disabled = index === dependencies.length - 1;
  const timeline = app.querySelector("[data-timeline]");
  timeline?.querySelectorAll(".debrief-source, .debrief-target").forEach((element) => element.classList.remove("debrief-source", "debrief-target"));
  timeline?.classList.add("debrief-active");
  debriefBlockElement(dependency.fromBlockId)?.classList.add("debrief-source");
  debriefBlockElement(dependency.toBlockId)?.classList.add("debrief-target");
  focusTimelineBlocks(task, [dependency.fromBlockId, dependency.toBlockId]);
}

function showCompletionDebrief(task, { preview = false } = {}) {
  const dependencies = debriefDependencies(task);
  const panel = app.querySelector("[data-completion-debrief]");
  if (!panel || !dependencies.length) return;
  panel.hidden = false;
  panel.querySelector("[data-debrief-kicker]").textContent = preview ? "Solution debrief" : "Completion debrief";
  app.querySelector(".timeline-shell")?.classList.add("has-debrief");
  const steps = panel.querySelector("[data-debrief-steps]");
  steps.innerHTML = dependencies.map((dependency, index) => `<button type="button" role="tab" data-debrief-step="${index}"><span>${String(index + 1).padStart(2, "0")}</span>${escapeHtml(dependency.label)}</button>`).join("");
  steps.querySelectorAll("button").forEach((button) => button.addEventListener("click", () => selectDebriefDependency(task, Number(button.dataset.debriefStep))));
  panel.querySelector("[data-debrief-previous]").onclick = () => selectDebriefDependency(task, state.activeDebriefIndex - 1);
  panel.querySelector("[data-debrief-next]").onclick = () => selectDebriefDependency(task, state.activeDebriefIndex + 1);
  if (!preview) applySolvedActions(task);
  selectDebriefDependency(task, 0);
}

function applySolvedActions(task) {
  const checkButton = app.querySelector("[data-check]");
  checkButton.innerHTML = "Solved ✓";
  checkButton.disabled = true;
  const nextTask = state.tasks
    .filter((item) => !item.catalogHidden && item.verification?.status === "measured" && item.order > task.order)
    .sort((left, right) => left.order - right.order)[0];
  const nextButton = app.querySelector("[data-next]");
  nextButton.hidden = false;
  nextButton.href = nextTask ? `#/task/${nextTask.id}` : "#/";
  nextButton.innerHTML = nextTask ? "Next challenge <span>→</span>" : "Challenge library <span>→</span>";
}

function buildPalette(task) {
  const palette = app.querySelector("[data-palette]");
  const movableBlocks = task.blocks.filter((block) => !block.fixed);
  const groups = task.paletteGroups?.length ? task.paletteGroups : [{ id: "all", label: "All events", blockIds: movableBlocks.map((block) => block.id) }];
  const groupByBlock = new Map(groups.flatMap((group) => group.blockIds.map((blockId) => [blockId, group.id])));
  const groupNav = app.querySelector("[data-palette-groups]");
  groupNav.setAttribute("role", "tablist");
  groupNav.hidden = groups.length === 1;

  const activateGroup = (groupId) => {
    state.cancelInteraction?.();
    groupNav.querySelectorAll("button").forEach((button) => {
      const active = button.dataset.paletteGroup === groupId;
      button.classList.toggle("active", active);
      button.setAttribute("aria-selected", String(active));
    });
    palette.querySelectorAll("[data-block]").forEach((button) => {
      button.hidden = button.dataset.paletteGroup !== groupId;
    });
  };

  groups.forEach((group, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "palette-group-button";
    button.setAttribute("role", "tab");
    button.dataset.paletteGroup = group.id;
    button.innerHTML = `<span>${escapeHtml(group.label)}</span><small data-group-count>${group.blockIds.length} left</small>`;
    button.addEventListener("click", () => {
      activateGroup(group.id);
      focusTimelineBlocks(task, group.blockIds);
    });
    groupNav.append(button);
    if (index === 0) button.setAttribute("aria-selected", "true");
  });

  movableBlocks.sort(() => 0.5 - Math.random()).forEach((block) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `event-block ${typeMeta[block.type].className}`;
    button.dataset.block = block.id;
    button.dataset.paletteGroup = groupByBlock.get(block.id) || groups[0].id;
    button.draggable = true;
    button.innerHTML = `<span>${escapeHtml(block.label)}</span>`;
    button.addEventListener("dragstart", (event) => {
      event.dataTransfer.setData("text/plain", block.id);
      event.dataTransfer.effectAllowed = "move";
      button.classList.add("dragging");
    });
    button.addEventListener("dragend", () => button.classList.remove("dragging"));
    button.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        selectBlock(block.id, { keyboard: true, origin: button });
      }
    });
    button.addEventListener("click", () => selectBlock(block.id, { origin: button }));
    bindBlockTooltip(button, block);
    palette.append(button);
  });
  activateGroup(groups[0].id);
  updatePaletteGroups(task);
}

function updatePaletteGroups(task) {
  (task.paletteGroups || []).forEach((group) => {
    const button = app.querySelector(`.palette-group-button[data-palette-group="${CSS.escape(group.id)}"]`);
    if (!button) return;
    const remaining = group.blockIds.filter((blockId) => !state.placements.has(blockId)).length;
    button.querySelector("[data-group-count]").textContent = remaining ? `${remaining} left` : "Done";
    button.classList.toggle("complete", remaining === 0);
  });
}

function placeFixedBlocks(task) {
  task.blocks.filter((block) => block.fixed).forEach((block) => {
    const slotId = targetSlotId(task, block);
    const slot = app.querySelector(`[data-slot="${CSS.escape(slotId)}"]`);
    if (!slot) return;
    slot.dataset.occupied = block.id;
    slot.dataset.fixed = "true";
    slot.setAttribute("aria-label", `Given event: ${block.label}`);
    slot.title = block.label;
    slot.classList.add("occupied", "fixed-slot");
    const callout = getCalloutPresentation(task, block);
    if (block.marker) slot.classList.add("marker-slot");
    if (callout) slot.classList.add("callout-slot");
    const widthPercent = task.timelineMode === "absolute" ? 100 : Math.min(100, block.durationMs / task.phases.find((item) => item.id === block.phase).durationMs * 100);
    const microBlock = task.timelineMode === "absolute" && block.durationMs * (task.pxPerMs || 24) < 28;
    slot.innerHTML = `<div class="placed-block fixed-block${widthPercent < 45 ? " compact-block" : ""}${block.marker ? " marker-block" : ""}${callout ? ` callout-block callout-lane-${callout.lane}${callout.alignEnd ? " callout-align-end" : ""}` : ""}${microBlock ? " micro-block" : ""} ${typeMeta[block.type].className}" style="--block-width:${widthPercent}%;--offset:${block.offset || 0}%"><span>${escapeHtml(block.shortLabel || block.label)}</span><small>given</small></div>`;
    bindBlockTooltip(task.timelineMode === "absolute" ? slot : slot.querySelector(".fixed-block"), block);
  });
}

function orderedEmptySlots(task) {
  const trackOrder = new Map(task.tracks.map((track, index) => [track.id, index]));
  const phaseOrder = new Map((task.phases || []).map((phase, index) => [phase.id, index]));
  return [...app.querySelectorAll(".trace-slot:not([data-occupied])")].sort((left, right) => {
    if (task.timelineMode === "absolute") {
      const leftBlock = task.blocks.find((block) => block.id === left.dataset.slot);
      const rightBlock = task.blocks.find((block) => block.id === right.dataset.slot);
      return (leftBlock?.startMs || 0) - (rightBlock?.startMs || 0)
        || (trackOrder.get(leftBlock?.track) || 0) - (trackOrder.get(rightBlock?.track) || 0);
    }
    const [leftTrack, leftPhase] = left.dataset.slot.split(":");
    const [rightTrack, rightPhase] = right.dataset.slot.split(":");
    return (phaseOrder.get(leftPhase) || 0) - (phaseOrder.get(rightPhase) || 0)
      || (trackOrder.get(leftTrack) || 0) - (trackOrder.get(rightTrack) || 0);
  });
}

function updateKeyboardPlacement() {
  const indicator = app.querySelector("[data-keyboard-placement]");
  if (!indicator) return;
  app.querySelectorAll(".keyboard-target").forEach((slot) => slot.classList.remove("keyboard-target"));
  const block = state.activeTask?.blocks.find((item) => item.id === state.activeBlock);
  indicator.hidden = !block;
  indicator.innerHTML = block
    ? `<strong>Placing</strong><span>${escapeHtml(block.shortLabel || block.label)}</span><small>Tab targets · Enter to place</small>`
    : "";
}

function focusKeyboardTarget(slot, slots = orderedEmptySlots(state.activeTask)) {
  if (!slot) return;
  app.querySelectorAll(".keyboard-target").forEach((item) => item.classList.remove("keyboard-target"));
  slot.classList.add("keyboard-target");
  slot.focus();
  const indicator = app.querySelector("[data-keyboard-placement]");
  const hint = indicator?.querySelector("small");
  const index = slots.indexOf(slot);
  if (hint && index !== -1) hint.textContent = `Target ${index + 1}/${slots.length} · Tab moves · Enter places`;
}

function focusPaletteButton(button) {
  if (!button) return;
  if (button.hidden) app.querySelector(`.palette-group-button[data-palette-group="${CSS.escape(button.dataset.paletteGroup)}"]`)?.click();
  requestAnimationFrame(() => button.focus());
}

function focusNextPaletteBlock() {
  const enabled = [...app.querySelectorAll("[data-block]:not(:disabled)")];
  if (!enabled.length) {
    requestAnimationFrame(() => app.querySelector("[data-check]")?.focus());
    return;
  }
  let next = enabled.find((button) => !button.hidden);
  if (!next) {
    next = enabled[0];
  }
  focusPaletteButton(next);
}

function selectBlock(blockId, { keyboard = false, origin = null } = {}) {
  if (state.placements.has(blockId)) return;
  state.activeBlock = state.activeBlock === blockId ? null : blockId;
  state.keyboardOrigin = state.activeBlock ? origin : null;
  app.querySelectorAll("[data-block]").forEach((element) => element.classList.toggle("selected", element.dataset.block === state.activeBlock));
  app.querySelectorAll(".trace-slot").forEach((slot) => slot.classList.toggle("selectable", Boolean(state.activeBlock) && !slot.dataset.occupied));
  updateKeyboardPlacement();
  if (keyboard && state.activeBlock) {
    const targets = orderedEmptySlots(state.activeTask);
    requestAnimationFrame(() => focusKeyboardTarget(targets[0], targets));
  }
}

function targetSlotId(task, block) {
  return task.timelineMode === "absolute" ? block.id : `${block.track}:${block.phase}`;
}

function getCalloutPresentation(task, block) {
  if (task.timelineMode !== "absolute" || block.marker) return null;
  const pxPerMs = task.pxPerMs || 24;
  const needsCallout = (item) => item.callout || item.durationMs * pxPerMs < 56;
  if (!needsCallout(block)) return null;
  const candidates = task.blocks
    .filter((item) => item.track === block.track && !item.marker && needsCallout(item))
    .sort((left, right) => left.startMs - right.startMs);
  const laneEnds = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  const timelineWidth = task.totalMs * pxPerMs;
  for (const candidate of candidates) {
    const start = candidate.startMs * pxPerMs;
    const label = candidate.shortLabel || candidate.label;
    const labelWidth = Math.min(112, Math.max(42, label.length * 4.7 + 10));
    let lane = laneEnds.findIndex((end) => start >= end + 5);
    if (lane === -1) lane = laneEnds.indexOf(Math.min(...laneEnds));
    laneEnds[lane] = start + labelWidth;
    if (candidate.id === block.id) return { lane, alignEnd: start + labelWidth > timelineWidth };
  }
  return null;
}

function placeBlock(blockId, slotId, task, { keyboard = false } = {}) {
  if (!blockId) return;
  const block = task.blocks.find((item) => item.id === blockId);
  const slot = app.querySelector(`[data-slot="${CSS.escape(slotId)}"]`);
  if (!block || !slot || slot.dataset.occupied) return;

  const existingSlot = state.placements.get(blockId);
  if (existingSlot) clearSlot(existingSlot);

  slot.dataset.occupied = blockId;
  slot.classList.add("occupied");
  slot.setAttribute("aria-label", block.label);
  slot.title = block.label;
  const widthPercent = task.timelineMode === "absolute" ? 100 : Math.min(100, block.durationMs / task.phases.find((phase) => slotId.endsWith(`:${phase.id}`)).durationMs * 100);
  const microBlock = task.timelineMode === "absolute" && block.durationMs * (task.pxPerMs || 24) < 28;
  const callout = getCalloutPresentation(task, block);
  if (callout) slot.classList.add("callout-slot");
  slot.innerHTML = `<div class="placed-block${callout ? ` callout-block callout-lane-${callout.lane}${callout.alignEnd ? " callout-align-end" : ""}` : ""}${microBlock ? " micro-block" : ""} ${typeMeta[block.type].className}" style="--block-width:${widthPercent}%;--offset:${block.offset || 0}%"><span>${escapeHtml(block.shortLabel || block.label)}</span><button type="button" aria-label="Remove ${escapeHtml(block.label)}">×</button></div>`;
  bindBlockTooltip(task.timelineMode === "absolute" ? slot : slot.querySelector(".placed-block"), block);
  slot.querySelector("button").addEventListener("click", (event) => { event.stopPropagation(); removeBlock(blockId); });
  state.placements.set(blockId, slotId);
  const paletteBlock = app.querySelector(`[data-block="${CSS.escape(blockId)}"]`);
  paletteBlock.disabled = true;
  paletteBlock.classList.remove("selected");
  state.activeBlock = null;
  state.keyboardOrigin = null;
  updateKeyboardPlacement();
  clearValidationFeedback();
  updateProgress(task);
  savePlacements(task);
  if (keyboard) focusNextPaletteBlock();
}

function clearSlot(slotId) {
  const slot = app.querySelector(`[data-slot="${CSS.escape(slotId)}"]`);
  if (!slot || slot.dataset.fixed) return;
  delete slot.dataset.occupied;
  slot.setAttribute("aria-label", slot.dataset.emptyLabel || "Empty target");
  slot.removeAttribute("title");
  slot.className = `trace-slot${slot.dataset.absolute ? " absolute-slot" : ""}${slot.dataset.microTarget ? " micro-target" : ""}`;
  slot.innerHTML = "<span>drop</span>";
}

function removeBlock(blockId) {
  const slotId = state.placements.get(blockId);
  if (!slotId) return;
  clearSlot(slotId);
  state.placements.delete(blockId);
  const paletteBlock = app.querySelector(`[data-block="${CSS.escape(blockId)}"]`);
  paletteBlock.disabled = false;
  clearValidationFeedback();
  updateProgress(state.activeTask);
  savePlacements(state.activeTask);
  return paletteBlock;
}

function updateProgress(task) {
  const playableCount = task.blocks.filter((block) => !block.fixed).length;
  const complete = state.placements.size === playableCount;
  app.querySelector("[data-placed-count]").textContent = state.placements.size;
  app.querySelector(".progress-pill")?.style.setProperty("--progress", `${playableCount ? state.placements.size / playableCount * 100 : 0}%`);
  const checkButton = app.querySelector("[data-check]");
  checkButton.disabled = state.placements.size === 0 || state.solutionRevealed;
  if (!state.solutionRevealed) checkButton.innerHTML = `${complete ? "Check trace" : "Check progress"} <span>→</span>`;
  app.querySelectorAll(".trace-slot").forEach((slot) => slot.classList.remove("selectable"));
  updatePaletteGroups(task);
}

function clearValidationFeedback() {
  const resultBanner = app.querySelector("[data-result-banner]");
  if (!resultBanner || state.solutionRevealed) return;
  resultBanner.hidden = true;
  resultBanner.className = "result-banner";
  app.querySelector("[data-status]").textContent = "";
  app.querySelector("[data-status]").className = "status-message";
  app.querySelectorAll(".trace-slot").forEach((slot) => slot.classList.remove("correct", "incorrect"));
}

function resetGame(task, { clearSaved = true } = {}) {
  hideCompletionDebrief();
  state.placements.clear();
  state.activeBlock = null;
  state.keyboardOrigin = null;
  updateKeyboardPlacement();
  app.querySelectorAll(".trace-slot:not([data-fixed])").forEach((slot) => clearSlot(slot.dataset.slot));
  app.querySelectorAll("[data-block]").forEach((block) => { block.disabled = false; block.classList.remove("selected"); });
  const resultBanner = app.querySelector("[data-result-banner]");
  resultBanner.hidden = true;
  resultBanner.className = "result-banner";
  app.querySelector("[data-status]").textContent = "";
  app.querySelector("[data-status]").className = "status-message";
  if (clearSaved) clearStoredPlacements(task);
  updateProgress(task);
}

function showFinalTrace(task) {
  hideCompletionDebrief();
  state.preRevealPlacements = new Map(state.placements);
  state.suppressPlacementSave = true;
  resetGame(task, { clearSaved: false });
  state.solutionRevealed = true;
  task.blocks.filter((block) => !block.fixed).forEach((block) => {
    placeBlock(block.id, targetSlotId(task, block), task);
  });
  state.suppressPlacementSave = false;
  const status = app.querySelector("[data-status]");
  const resultBanner = app.querySelector("[data-result-banner]");
  resultBanner.hidden = false;
  resultBanner.className = "result-banner debug";
  resultBanner.querySelector("[data-result-label]").textContent = "Preview";
  status.className = "status-message debug";
  status.textContent = "Solution preview — this does not count as completing the challenge.";
  app.querySelector("[data-show-final]").disabled = true;
  app.querySelector("[data-show-final]").textContent = "Solution revealed";
  app.querySelector("[data-reset]").textContent = "Return to my trace";
  updateProgress(task);
  showCompletionDebrief(task, { preview: true });
}

function returnFromSolution(task) {
  const placements = new Map(state.preRevealPlacements || []);
  state.suppressPlacementSave = true;
  state.solutionRevealed = false;
  resetGame(task, { clearSaved: false });
  for (const [blockId, slotId] of placements) placeBlock(blockId, slotId, task);
  state.suppressPlacementSave = false;
  state.preRevealPlacements = null;
  app.querySelector("[data-show-final]").disabled = false;
  app.querySelector("[data-show-final]").textContent = "Reveal solution";
  app.querySelector("[data-reset]").textContent = "Reset trace";
  updateProgress(task);
  if (state.completed.has(task.id) && traceIsSolved(task)) showCompletionDebrief(task);
}

function checkTrace(task) {
  const playableBlocks = task.blocks.filter((block) => !block.fixed);
  const placedBlocks = playableBlocks.filter((block) => state.placements.has(block.id));
  const mistakes = placedBlocks.filter((block) => state.placements.get(block.id) !== targetSlotId(task, block));
  const unplaced = playableBlocks.length - placedBlocks.length;
  app.querySelectorAll(".trace-slot").forEach((slot) => slot.classList.remove("correct", "incorrect"));
  placedBlocks.forEach((block) => {
    const slot = app.querySelector(`[data-slot="${CSS.escape(state.placements.get(block.id) || "")}"]`);
    if (slot) slot.classList.add(mistakes.includes(block) ? "incorrect" : "correct");
  });
  const status = app.querySelector("[data-status]");
  const resultBanner = app.querySelector("[data-result-banner]");
  resultBanner.hidden = false;
  if (!mistakes.length && unplaced === 0) {
    resultBanner.className = "result-banner success";
    resultBanner.querySelector("[data-result-label]").textContent = "Correct trace";
    status.className = "status-message success";
    status.innerHTML = `<span>Trace complete — ${escapeHtml(task.explanation)}</span>${completionGithubLink()}`;
    state.completed.add(task.id);
    localStorage.setItem(progressKey, JSON.stringify([...state.completed]));
    showCompletionDebrief(task);
  } else {
    const correctCount = placedBlocks.length - mistakes.length;
    resultBanner.className = `result-banner ${mistakes.length ? "error" : "progress"}`;
    resultBanner.querySelector("[data-result-label]").textContent = mistakes.length ? "Review placement" : "Progress check";
    status.className = `status-message ${mistakes.length ? "error" : "progress"}`;
    const summary = `${correctCount} placed correctly${mistakes.length ? ` · ${mistakes.length} need${mistakes.length === 1 ? "s" : ""} review` : ""}${unplaced ? ` · ${unplaced} remaining` : ""}.`;
    const firstMistake = mistakes[0];
    status.innerHTML = `${escapeHtml(summary)}${firstMistake ? `<span class="feedback-detail"><strong>First issue: ${escapeHtml(firstMistake.label)}.</strong> ${escapeHtml(firstMistake.description || "Revisit the operation's input dependency and first consumer.")}</span>` : ""}`;
  }
}

function updateFocusButton(enable) {
  const focusButton = app.querySelector("[data-focus-mode]");
  if (!focusButton) return;
  focusButton.setAttribute("aria-pressed", String(enable));
  focusButton.querySelector("[data-focus-label]").textContent = enable ? "Exit focus" : "Focus";
}

async function setTraceFocus(enable) {
  document.body.classList.toggle("trace-focus", enable);
  updateFocusButton(enable);
  try {
    if (enable && !document.fullscreenElement) await document.documentElement.requestFullscreen();
    if (!enable && document.fullscreenElement) await document.exitFullscreen();
  } catch {
    document.body.classList.toggle("trace-focus", enable);
  }
}

function bindGameActions(task) {
  app.querySelector("[data-show-final]").addEventListener("click", () => showFinalTrace(task));
  app.querySelector("[data-reset]").addEventListener("click", () => {
    if (state.solutionRevealed) {
      returnFromSolution(task);
      return;
    }
    state.preRevealPlacements = null;
    app.querySelector("[data-check]").innerHTML = "Check progress <span>→</span>";
    app.querySelector("[data-next]").hidden = true;
    resetGame(task);
  });
  app.querySelector("[data-check]").addEventListener("click", () => checkTrace(task));
  app.querySelector("[data-start-task]").addEventListener("click", () => {
    app.querySelector(".workspace").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  const focusButton = app.querySelector("[data-focus-mode]");
  focusButton.addEventListener("click", () => setTraceFocus(!document.body.classList.contains("trace-focus")));
}

function renderReadOnlyTrace(view, side, scaleMs) {
  const sourceTask = state.tasks.find((item) => item.id === view.taskId);
  const task = sourceTask && view.variantId ? resolveTaskVariant(sourceTask, view.variantId) : sourceTask;
  if (!task || task.timelineMode !== "absolute") throw new Error(`Comparison source must be an absolute trace: ${view.taskId}`);
  const tracks = task.tracks.filter((track) => view.tracks.includes(track.id));
  const blocks = task.blocks.filter((block) => (
    view.tracks.includes(block.track) &&
    block.startMs < view.endMs &&
    block.startMs + block.durationMs > view.startMs
  ));
  const article = document.createElement("article");
  article.className = "comparison-trace-panel";
  article.dataset.traceSide = side;
  article.innerHTML = `<header><div><span>${escapeHtml(view.label)}</span><h2>${escapeHtml(task.title)}</h2></div><small>${(view.endMs - view.startMs).toFixed(2)} ms window</small></header><p>${escapeHtml(view.caption)}</p><div class="comparison-panel-toolbar"><div class="timeline-zoom" aria-label="${escapeHtml(view.label)} trace zoom"><button type="button" data-panel-fit>Fit</button><button type="button" data-panel-zoom-out aria-label="Zoom out">−</button><span data-panel-zoom-level>100%</span><button type="button" data-panel-zoom-in aria-label="Zoom in">+</button></div><button class="focus-button" type="button" data-panel-focus aria-pressed="false"><span class="focus-icon" aria-hidden="true"></span><span data-focus-label>Focus</span></button></div><div class="comparison-trace-scroll"><div class="comparison-trace"><div class="comparison-ruler"><span>0</span><span>shared scale: ${scaleMs.toFixed(2)} ms</span></div><div data-comparison-rows></div></div></div>`;
  const rows = article.querySelector("[data-comparison-rows]");
  tracks.forEach((track) => {
    const row = document.createElement("div");
    row.className = "comparison-trace-row";
    row.innerHTML = `<div class="comparison-track-label"><b>${escapeHtml(track.label)}</b><small>${escapeHtml(track.subtitle || "")}</small></div><div class="comparison-track-lane"></div>`;
    const lane = row.querySelector(".comparison-track-lane");
    const trackBlocks = blocks.filter((block) => block.track === track.id).sort((left, right) => left.startMs - right.startMs);
    const laneEnds = [];
    const blockLanes = new Map();
    trackBlocks.forEach((block) => {
      const visibleStart = Math.max(block.startMs, view.startMs);
      const visibleEnd = Math.min(block.startMs + block.durationMs, view.endMs);
      let laneIndex = laneEnds.findIndex((end) => end <= visibleStart + .0001);
      if (laneIndex < 0) laneIndex = laneEnds.length;
      laneEnds[laneIndex] = visibleEnd;
      blockLanes.set(block.id, laneIndex);
    });
    const rowHeight = Math.max(66, 18 + laneEnds.length * 36);
    row.style.minHeight = `${rowHeight}px`;
    lane.style.minHeight = `${rowHeight}px`;
    trackBlocks.forEach((block) => {
      const visibleStart = Math.max(block.startMs, view.startMs);
      const visibleEnd = Math.min(block.startMs + block.durationMs, view.endMs);
      const left = (visibleStart - view.startMs) / scaleMs * 100;
      const width = (visibleEnd - visibleStart) / scaleMs * 100;
      const button = document.createElement("button");
      button.type = "button";
      button.className = `comparison-trace-block ${typeMeta[block.type].className}${width < 4 ? " comparison-micro-block" : ""}`;
      button.style.left = `${left}%`;
      button.style.width = `${width}%`;
      button.style.top = `${10 + blockLanes.get(block.id) * 36}px`;
      button.dataset.comparisonBlock = block.id;
      button.dataset.comparisonSide = side;
      button.setAttribute("aria-label", block.label);
      const viewLabel = view.blockLabels?.[block.id];
      button.innerHTML = `<span>${escapeHtml(viewLabel || block.shortLabel || block.label.replace(/^Fixed context:\s*/i, ""))}</span>`;
      bindBlockTooltip(button, block);
      lane.append(button);
    });
    rows.append(row);
  });
  return article;
}

function readComparisonJourneys() {
  try { return JSON.parse(localStorage.getItem(comparisonJourneyKey) || "{}"); } catch { return {}; }
}

function saveComparisonJourney(comparisonId, questionIndex) {
  const journeys = readComparisonJourneys();
  if (questionIndex > 0) journeys[comparisonId] = { questionIndex };
  else delete journeys[comparisonId];
  try { localStorage.setItem(comparisonJourneyKey, JSON.stringify(journeys)); } catch {}
}

function setComparisonFocus(panel, enable) {
  const side = panel?.dataset.traceSide || null;
  document.body.classList.toggle("comparison-focus", enable);
  state.focusedComparisonSide = enable ? side : null;
  app.querySelectorAll(".comparison-trace-panel").forEach((tracePanel) => tracePanel.classList.toggle("is-focused", enable && tracePanel === panel));
  app.querySelectorAll("[data-panel-focus]").forEach((button) => {
    const active = enable && button.closest(".comparison-trace-panel") === panel;
    button.setAttribute("aria-pressed", String(active));
    button.querySelector("[data-focus-label]").textContent = active ? "Exit focus" : "Focus";
  });
  if (enable && !document.fullscreenElement) {
    document.documentElement.requestFullscreen().catch(() => {});
  } else if (!enable && document.fullscreenElement) {
    document.exitFullscreen().catch(() => {});
  }
  const controller = state.comparisonZoomControllers.find((item) => item.side === side)?.controller;
  requestAnimationFrame(() => requestAnimationFrame(() => controller?.fit()));
}

function bindComparisonViewport() {
  state.comparisonZoomControllers.forEach(({ controller }) => controller.destroy());
  state.comparisonZoomControllers = [...app.querySelectorAll(".comparison-trace-panel")].map((panel) => {
    const scroll = panel.querySelector(".comparison-trace-scroll");
    const trace = panel.querySelector(".comparison-trace");
    const controller = createWidthZoomController({
      naturalWidth: 630,
      scrollElements: [scroll],
      getFitWidth: () => Math.max(300, scroll.clientWidth),
      applyWidth: (width) => {
      trace.style.width = `${width}px`;
      trace.style.minWidth = `${width}px`;
      },
      fitButton: panel.querySelector("[data-panel-fit]"),
      zoomOutButton: panel.querySelector("[data-panel-zoom-out]"),
      zoomInButton: panel.querySelector("[data-panel-zoom-in]"),
      levelElement: panel.querySelector("[data-panel-zoom-level]"),
      maxScale: 10,
    });
    panel.querySelector("[data-panel-focus]").addEventListener("click", () => {
      const active = document.body.classList.contains("comparison-focus") && panel.classList.contains("is-focused");
      setComparisonFocus(panel, !active);
    });
    return { side: panel.dataset.traceSide, controller };
  });
}

function renderComparison(comparison) {
  if (document.body.classList.contains("trace-focus")) void setTraceFocus(false);
  if (document.body.classList.contains("comparison-focus")) setComparisonFocus(app.querySelector(".comparison-trace-panel.is-focused"), false);
  const fragment = document.querySelector("#comparison-template").content.cloneNode(true);
  fragment.querySelector("[data-comparison-kicker]").textContent = `Chapter 2 · Comparison lab ${String(comparison.order).padStart(2, "0")}`;
  fragment.querySelector("[data-comparison-title]").textContent = comparison.title;
  fragment.querySelector("[data-comparison-summary]").textContent = comparison.summary;
  fragment.querySelector("[data-comparison-tags]").innerHTML = comparison.tags.map((tag) => `<span>${escapeHtml(tag)}</span>`).join("");
  fragment.querySelector("[data-comparison-progress]").textContent = `${comparison.questions.length} dependency questions`;
  fragment.querySelector("[data-comparison-intro]").textContent = comparison.intro;
  fragment.querySelector("[data-comparison-caveat]").textContent = comparison.caveat;
  const prerequisites = fragment.querySelector("[data-comparison-prerequisites]");
  prerequisites.innerHTML = (comparison.prerequisites || []).map((taskId) => {
    const task = state.tasks.find((item) => item.id === taskId);
    return task ? `<a href="#/task/${escapeHtml(task.id)}">Review: ${escapeHtml(task.title)} <span>↗</span></a>` : "";
  }).join("");
  prerequisites.hidden = !prerequisites.children.length;
  const tracePair = fragment.querySelector("[data-trace-pair]");
  tracePair.append(renderReadOnlyTrace(comparison.left, "left", comparison.scaleMs));
  tracePair.append(renderReadOnlyTrace(comparison.right, "right", comparison.scaleMs));
  app.replaceChildren(fragment);
  document.title = `${comparison.title} — Build Your Trace`;
  bindComparisonViewport();

  const savedQuestion = Number(readComparisonJourneys()[comparison.id]?.questionIndex || 0);
  let questionIndex = Math.max(0, Math.min(comparison.questions.length - 1, savedQuestion));
  let selectedBlockId = null;
  let selectedOptionId = null;
  const questionPanel = app.querySelector("[data-dependency-question]");
  const checkButton = questionPanel.querySelector("[data-question-check]");
  const nextButton = questionPanel.querySelector("[data-question-next]");
  const doneLink = questionPanel.querySelector("[data-comparison-done]");
  const nextLabLink = questionPanel.querySelector("[data-comparison-next-lab]");
  const options = questionPanel.querySelector("[data-question-options]");
  const feedback = questionPanel.querySelector("[data-question-feedback]");
  const comparisonIndex = state.comparisons.findIndex((item) => item.id === comparison.id);
  const nextComparison = state.comparisons[comparisonIndex + 1];
  if (nextComparison) nextLabLink.href = `#/compare/${nextComparison.id}`;

  const showQuestion = () => {
    const question = comparison.questions[questionIndex];
    selectedBlockId = null;
    selectedOptionId = null;
    const choiceQuestion = question.kind === "choice";
    app.querySelectorAll(".comparison-trace-block").forEach((block) => {
      block.classList.remove("selected", "correct", "incorrect");
      const answerable = !choiceQuestion && block.dataset.comparisonSide === question.side;
      block.dataset.answerable = String(answerable);
      block.setAttribute("aria-disabled", String(!answerable));
      block.classList.toggle("not-answerable", !answerable);
    });
    app.querySelectorAll(".comparison-trace-panel").forEach((panel) => panel.classList.toggle("question-target", !choiceQuestion && panel.dataset.traceSide === question.side));
    questionPanel.querySelector("[data-question-count]").textContent = `${question.stage || (choiceQuestion ? "Explain" : "Inspect")} · question ${questionIndex + 1} of ${comparison.questions.length}`;
    questionPanel.querySelector("[data-question-prompt]").textContent = question.prompt;
    questionPanel.querySelector("[data-question-instruction]").textContent = choiceQuestion
      ? "Choose the explanation that follows from the dependency, not only from the visible label."
      : `Select one block in the ${question.side === "left" ? comparison.left.label : comparison.right.label} trace.`;
    options.hidden = !choiceQuestion;
    options.innerHTML = choiceQuestion ? question.options.map((option) => `<button type="button" data-question-option="${escapeHtml(option.id)}">${escapeHtml(option.label)}</button>`).join("") : "";
    options.querySelectorAll("[data-question-option]").forEach((optionButton) => optionButton.addEventListener("click", () => {
      options.querySelectorAll("button").forEach((item) => item.classList.remove("selected", "incorrect"));
      optionButton.classList.add("selected");
      selectedOptionId = optionButton.dataset.questionOption;
      feedback.className = "question-feedback";
      feedback.textContent = "";
      checkButton.disabled = false;
    }));
    feedback.className = "question-feedback";
    feedback.textContent = "";
    checkButton.disabled = true;
    checkButton.hidden = false;
    nextButton.hidden = true;
    doneLink.hidden = true;
  };

  app.querySelectorAll(".comparison-trace-block").forEach((block) => {
    block.addEventListener("click", () => {
      if (block.dataset.answerable !== "true" || comparison.questions[questionIndex].kind === "choice") return;
      app.querySelectorAll(".comparison-trace-block").forEach((item) => item.classList.remove("selected", "incorrect"));
      block.classList.add("selected");
      selectedBlockId = block.dataset.comparisonBlock;
      feedback.className = "question-feedback";
      feedback.textContent = "";
      checkButton.disabled = false;
    });
  });

  checkButton.addEventListener("click", () => {
    const question = comparison.questions[questionIndex];
    const choiceQuestion = question.kind === "choice";
    const selected = choiceQuestion
      ? options.querySelector(`[data-question-option="${CSS.escape(selectedOptionId || "")}"]`)
      : app.querySelector(`.comparison-trace-block[data-comparison-side="${question.side}"][data-comparison-block="${CSS.escape(selectedBlockId || "")}"]`);
    const correct = choiceQuestion ? selectedOptionId === question.answerOptionId : selectedBlockId === question.answerBlockId;
    if (!correct) {
      selected?.classList.add("incorrect");
      feedback.className = "question-feedback error";
      feedback.textContent = question.nudge;
      return;
    }
    selected?.classList.add("correct");
    feedback.className = "question-feedback success";
    feedback.textContent = question.explanation;
    checkButton.hidden = true;
    if (questionIndex < comparison.questions.length - 1) {
      nextButton.hidden = false;
    } else {
      state.comparisonsCompleted.add(comparison.id);
      localStorage.setItem(comparisonProgressKey, JSON.stringify([...state.comparisonsCompleted]));
      saveComparisonJourney(comparison.id, 0);
      doneLink.hidden = false;
      nextLabLink.hidden = !nextComparison;
      questionPanel.querySelector("[data-question-count]").textContent = "Lab complete";
      feedback.insertAdjacentHTML("beforeend", completionGithubLink());
    }
  });

  nextButton.addEventListener("click", () => {
    questionIndex += 1;
    saveComparisonJourney(comparison.id, questionIndex);
    showQuestion();
    if (matchMedia("(max-width: 760px)").matches) questionPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  });
  state.cancelInteraction = () => {
    if (!selectedBlockId && !selectedOptionId) return false;
    selectedBlockId = null;
    selectedOptionId = null;
    app.querySelectorAll(".comparison-trace-block, [data-question-option]").forEach((item) => item.classList.remove("selected", "incorrect"));
    feedback.className = "question-feedback";
    feedback.textContent = "Selection cleared.";
    checkButton.disabled = true;
    return true;
  };
  showQuestion();
}

function readCollectivePlacements() {
  try { return JSON.parse(localStorage.getItem(collectivePlacementKey) || "{}"); } catch { return {}; }
}

function renderCollectiveLesson(lesson) {
  if (document.body.classList.contains("trace-focus")) void setTraceFocus(false);
  const fragment = document.querySelector("#collective-template").content.cloneNode(true);
  fragment.querySelector("[data-collective-kicker]").textContent = `Interactive primer · Primitive ${String(lesson.order).padStart(2, "0")}`;
  fragment.querySelector("[data-collective-title]").textContent = lesson.title;
  fragment.querySelector("[data-collective-summary]").textContent = lesson.summary;
  fragment.querySelector("[data-collective-instruction]").textContent = lesson.instruction;
  fragment.querySelector("[data-collective-insight]").textContent = lesson.insight;

  const allInputChunks = lesson.inputRanks.flat();
  const inputById = new Map(allInputChunks.map((chunk) => [chunk.id, chunk]));
  const resultEntries = Object.entries(lesson.resultChunks);
  const sourceRank = (chunkId) => {
    const inputSource = inputById.get(chunkId)?.source;
    if (inputSource !== undefined) return inputSource;
    if (chunkId.startsWith("sum-")) return Number(chunkId.split("-")[1]);
    return Number(chunkId.split("-")[0]) || 0;
  };
  const chunkMarkup = (chunkId, extraClass = "") => {
    const chunk = lesson.resultChunks[chunkId];
    const description = chunk.description ? ` title="${escapeHtml(chunk.description)}"` : "";
    return `<span class="tensor-chunk rank-${sourceRank(chunkId)} ${extraClass}"${description}>${escapeHtml(chunk.label)}</span>`;
  };

  fragment.querySelector("[data-collective-inputs]").innerHTML = lesson.inputRanks.map((chunks, rank) => `
    <div class="collective-rank-row">
      <div class="collective-rank-label"><strong>Rank ${rank}</strong><small>input</small></div>
      <div class="collective-tensor input-tensor">${chunks.map((chunk) => `<span class="tensor-chunk rank-${chunk.source}" title="${escapeHtml(chunk.description || `Input chunk from rank ${chunk.source}`)}">${escapeHtml(chunk.label)}</span>`).join("")}</div>
    </div>`).join("");

  fragment.querySelector("[data-collective-outputs]").innerHTML = lesson.outputRanks.map((chunks, rank) => `
    <div class="collective-rank-row">
      <div class="collective-rank-label"><strong>Rank ${rank}</strong><small>result</small></div>
      <div class="collective-tensor output-tensor">${chunks.map((_, index) => `<button class="tensor-slot" type="button" data-collective-slot="${rank}:${index}" aria-label="Result rank ${rank}, position ${index + 1}"></button>`).join("")}</div>
    </div>`).join("");

  fragment.querySelector("[data-collective-palette]").innerHTML = resultEntries.map(([chunkId, chunk]) => `
    <button class="collective-palette-chunk rank-${sourceRank(chunkId)}" type="button" draggable="true" data-collective-chunk="${escapeHtml(chunkId)}" title="${escapeHtml(chunk.description || chunk.label)}">
      <span>${escapeHtml(chunk.label)}</span><small>${escapeHtml(chunk.description || "Move this chunk")}</small>
    </button>`).join("");

  const nextLesson = [...state.collectives].sort((left, right) => left.order - right.order).find((item) => item.order > lesson.order);
  const nextLink = fragment.querySelector("[data-collective-next]");
  nextLink.href = nextLesson ? `#/collective/${nextLesson.id}` : "#/";
  nextLink.innerHTML = nextLesson ? `Next primitive <span>→</span>` : `Back to challenges <span>→</span>`;
  app.replaceChildren(fragment);
  document.title = `${lesson.title} — Build Your Trace`;

  const placementStore = readCollectivePlacements();
  const placements = new Map(Object.entries(placementStore[lesson.id] || {}));
  const allowsReplication = lesson.operation === "all_gather" || lesson.operation === "all_reduce";
  const totalSlots = lesson.outputRanks.flat().length;
  let activeChunkId = null;
  const selectionFeedback = () => {
    const feedback = app.querySelector("[data-collective-feedback]");
    if (!feedback || feedback.classList.contains("error") || feedback.classList.contains("success")) return;
    feedback.textContent = activeChunkId && allowsReplication
      ? `${lesson.resultChunks[activeChunkId].label} selected — place every copy; it stays selected.`
      : "";
  };

  const save = () => {
    const store = readCollectivePlacements();
    if (placements.size) store[lesson.id] = Object.fromEntries(placements);
    else delete store[lesson.id];
    try { localStorage.setItem(collectivePlacementKey, JSON.stringify(store)); } catch {}
  };

  const draw = () => {
    app.querySelectorAll("[data-collective-slot]").forEach((slot) => {
      slot.classList.remove("occupied", "correct", "incorrect", "missing");
      const chunkId = placements.get(slot.dataset.collectiveSlot);
      slot.innerHTML = chunkId ? chunkMarkup(chunkId) : "";
      slot.classList.toggle("occupied", Boolean(chunkId));
    });
    app.querySelectorAll("[data-collective-chunk]").forEach((button) => {
      const used = [...placements.values()].includes(button.dataset.collectiveChunk);
      button.classList.toggle("active", button.dataset.collectiveChunk === activeChunkId);
      button.classList.toggle("used", used && !allowsReplication);
    });
    app.querySelector("[data-collective-progress]").textContent = `${placements.size}/${totalSlots} placed`;
    app.querySelector("[data-collective-check]").disabled = placements.size === 0;
    selectionFeedback();
  };

  const selectChunk = (chunkId) => {
    const feedback = app.querySelector("[data-collective-feedback]");
    feedback.className = "collective-feedback";
    activeChunkId = activeChunkId === chunkId ? null : chunkId;
    draw();
  };

  const placeChunk = (slotKey, chunkId) => {
    if (!allowsReplication) {
      for (const [placedSlot, placedChunk] of placements) {
        if (placedChunk === chunkId) placements.delete(placedSlot);
      }
    }
    placements.set(slotKey, chunkId);
    if (!allowsReplication) activeChunkId = null;
    app.querySelector("[data-collective-feedback]").className = "collective-feedback";
    app.querySelector("[data-collective-feedback]").textContent = "";
    save();
    draw();
  };

  app.querySelectorAll("[data-collective-chunk]").forEach((button) => {
    button.addEventListener("click", () => selectChunk(button.dataset.collectiveChunk));
    button.addEventListener("dragstart", (event) => {
      event.dataTransfer.setData("text/plain", button.dataset.collectiveChunk);
      event.dataTransfer.effectAllowed = "copyMove";
    });
  });

  app.querySelectorAll("[data-collective-slot]").forEach((slot) => {
    slot.addEventListener("click", () => {
      if (activeChunkId) placeChunk(slot.dataset.collectiveSlot, activeChunkId);
      else if (placements.delete(slot.dataset.collectiveSlot)) { save(); draw(); }
    });
    slot.addEventListener("dragover", (event) => {
      event.preventDefault();
      slot.classList.add("drag-over");
    });
    slot.addEventListener("dragleave", () => slot.classList.remove("drag-over"));
    slot.addEventListener("drop", (event) => {
      event.preventDefault();
      slot.classList.remove("drag-over");
      const chunkId = event.dataTransfer.getData("text/plain");
      if (lesson.resultChunks[chunkId]) placeChunk(slot.dataset.collectiveSlot, chunkId);
    });
  });

  app.querySelector("[data-collective-reset]").addEventListener("click", () => {
    placements.clear();
    activeChunkId = null;
    save();
    app.querySelector("[data-collective-feedback]").className = "collective-feedback";
    app.querySelector("[data-collective-feedback]").textContent = "";
    app.querySelector("[data-collective-next]").hidden = true;
    draw();
  });

  app.querySelector("[data-collective-check]").addEventListener("click", () => {
    let correct = 0;
    let missing = 0;
    let misplaced = 0;
    app.querySelectorAll("[data-collective-slot]").forEach((slot) => {
      const [rank, index] = slot.dataset.collectiveSlot.split(":").map(Number);
      const actual = placements.get(slot.dataset.collectiveSlot);
      const matches = actual === lesson.outputRanks[rank][index];
      slot.classList.add(matches ? "correct" : actual ? "incorrect" : "missing");
      if (matches) correct += 1;
      else if (actual) misplaced += 1;
      else missing += 1;
    });
    const feedback = app.querySelector("[data-collective-feedback]");
    if (correct !== totalSlots) {
      feedback.className = "collective-feedback error";
      feedback.textContent = `${correct} of ${totalSlots} positions are correct; ${misplaced} misplaced and ${missing} still empty. ${lesson.explanation[0].toUpperCase()}${lesson.explanation.slice(1)}`;
      return;
    }
    state.collectivesCompleted.add(lesson.id);
    try { localStorage.setItem(collectiveProgressKey, JSON.stringify([...state.collectivesCompleted])); } catch {}
    feedback.className = "collective-feedback success";
    feedback.innerHTML = `<strong>Correct.</strong> ${escapeHtml(lesson.insight)}${completionGithubLink()}`;
    app.querySelector("[data-collective-next]").hidden = false;
  });

  state.cancelInteraction = () => {
    if (!activeChunkId) return false;
    activeChunkId = null;
    const feedback = app.querySelector("[data-collective-feedback]");
    feedback.className = "collective-feedback";
    feedback.textContent = "Selection cleared.";
    draw();
    return true;
  };

  draw();
}

function currentTask() {
  return state.tasks.find((task) => location.hash.endsWith(`/task/${task.id}`));
}

function route({ restoreScroll = false } = {}) {
  saveScrollPosition();
  state.cancelInteraction = null;
  const nextRouteKey = routeKey();
  scrollTrackingReady = false;
  const collectiveMatch = location.hash.match(/^#\/collective\/([^/]+)$/);
  if (collectiveMatch) {
    const lesson = state.collectives.find((item) => item.id === collectiveMatch[1]);
    if (lesson) renderCollectiveLesson(lesson);
    else renderHome();
    activeRouteKey = nextRouteKey;
    positionRoute(nextRouteKey, restoreScroll);
    app.focus({ preventScroll: true });
    return;
  }
  const comparisonMatch = location.hash.match(/^#\/compare\/([^/]+)$/);
  if (comparisonMatch) {
    const comparison = state.comparisons.find((item) => item.id === comparisonMatch[1]);
    if (comparison) renderComparison(comparison);
    else renderHome();
    activeRouteKey = nextRouteKey;
    positionRoute(nextRouteKey, restoreScroll);
    app.focus({ preventScroll: true });
    return;
  }
  const match = location.hash.match(/^#\/task\/([^/]+)$/);
  if (match) {
    const task = state.tasks.find((item) => item.id === match[1]);
    if (task?.verification?.status === "measured" && !task.catalogHidden) renderGame(task);
    else if (task) app.innerHTML = `<section class="error-page"><p class="eyebrow">Not available</p><h1>${escapeHtml(task.title)}</h1><p>This trace is still being calibrated.</p><p><a href="#/">← Return to challenges</a></p></section>`;
    else renderHome();
  } else {
    if (document.body.classList.contains("trace-focus")) void setTraceFocus(false);
    if (document.body.classList.contains("comparison-focus")) setComparisonFocus(app.querySelector(".comparison-trace-panel.is-focused"), false);
    renderHome();
  }
  activeRouteKey = nextRouteKey;
  positionRoute(nextRouteKey, restoreScroll);
  app.focus({ preventScroll: true });
}

document.addEventListener("click", (event) => {
  if (event.target.closest("[data-theme-toggle]")) setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
  if (event.target.closest("[data-method-open]")) methodDialog.showModal();
  if (event.target.closest("[data-scroll-challenges]")) {
    const challenges = document.querySelector("#challenges");
    if (challenges) challenges.scrollIntoView({ behavior: "smooth" });
    else {
      pendingChallengeScroll = true;
      if (location.hash === "#/") route();
      else location.hash = "#/";
    }
  }
  if (event.target.closest("[data-scroll-top]")) window.scrollTo({ top: 0, behavior: "smooth" });
});
window.addEventListener("hashchange", () => route({ restoreScroll: routeKey() === "#/" }));
window.addEventListener("scroll", () => saveScrollPosition(), { passive: true });
window.addEventListener("pagehide", () => saveScrollPosition());
document.addEventListener("fullscreenchange", () => {
  if (document.fullscreenElement) return;
  document.body.classList.remove("trace-focus");
  document.body.classList.remove("comparison-focus");
  state.focusedComparisonSide = null;
  updateFocusButton(false);
  app.querySelectorAll(".comparison-trace-panel").forEach((panel) => panel.classList.remove("is-focused"));
  app.querySelectorAll("[data-panel-focus]").forEach((button) => {
    button.setAttribute("aria-pressed", "false");
    button.querySelector("[data-focus-label]").textContent = "Focus";
  });
  requestAnimationFrame(() => state.comparisonZoomControllers.forEach(({ controller }) => controller.fit()));
});
document.addEventListener("keydown", (event) => {
  const gamePage = app.querySelector(".game-page");
  const editable = event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement || event.target?.isContentEditable;
  if (gamePage && !editable && (event.key === "Delete" || event.key === "Backspace")) {
    const slot = event.target.closest?.(".trace-slot[data-occupied]:not([data-fixed])");
    if (slot) {
      const paletteBlock = removeBlock(slot.dataset.occupied);
      requestAnimationFrame(() => paletteBlock?.focus());
      event.preventDefault();
      return;
    }
  }
  if (gamePage && state.activeBlock && event.key === "Tab") {
    const slots = orderedEmptySlots(state.activeTask);
    if (slots.length) {
      const currentIndex = slots.indexOf(document.activeElement);
      const direction = event.shiftKey ? -1 : 1;
      const nextIndex = currentIndex === -1
        ? (event.shiftKey ? slots.length - 1 : 0)
        : (currentIndex + direction + slots.length) % slots.length;
      focusKeyboardTarget(slots[nextIndex], slots);
      event.preventDefault();
      return;
    }
  }
  if (gamePage && !state.activeBlock && event.key === "Tab") {
    const paletteBlocks = [...app.querySelectorAll("[data-block]:not(:disabled)")];
    const currentIndex = paletteBlocks.indexOf(document.activeElement);
    const outsideWorkspace = !event.target.closest?.(".workspace");
    if (currentIndex !== -1) {
      const nextIndex = currentIndex + (event.shiftKey ? -1 : 1);
      if (nextIndex >= 0 && nextIndex < paletteBlocks.length) focusPaletteButton(paletteBlocks[nextIndex]);
      else if (event.shiftKey) focusPaletteButton(paletteBlocks[paletteBlocks.length - 1]);
      else app.querySelector("[data-check]")?.focus();
      event.preventDefault();
      return;
    }
    if (outsideWorkspace || event.target === app) {
      if (paletteBlocks.length) focusPaletteButton(paletteBlocks[event.shiftKey ? paletteBlocks.length - 1 : 0]);
      else app.querySelector("[data-check]")?.focus();
      event.preventDefault();
      return;
    }
  }
  if (gamePage && !editable && !event.metaKey && !event.ctrlKey && !event.altKey) {
    if (event.key === "+" || event.key === "=") {
      state.timelineZoomController?.zoomBy(1.2);
      event.preventDefault();
      return;
    }
    if (event.key === "-" || event.key === "_") {
      state.timelineZoomController?.zoomBy(1 / 1.2);
      event.preventDefault();
      return;
    }
    if (event.key === "0") {
      state.timelineZoomController?.fit();
      event.preventDefault();
      return;
    }
    if (event.key.toLowerCase() === "f") {
      void setTraceFocus(!document.body.classList.contains("trace-focus"));
      event.preventDefault();
      return;
    }
  }
  const debrief = app.querySelector("[data-completion-debrief]:not([hidden])");
  if (debrief && (event.key === "ArrowLeft" || event.key === "ArrowRight")) {
    const dependencies = state.activeTask ? debriefDependencies(state.activeTask) : [];
    const direction = event.key === "ArrowLeft" ? -1 : 1;
    const nextIndex = Math.max(0, Math.min(dependencies.length - 1, state.activeDebriefIndex + direction));
    if (nextIndex !== state.activeDebriefIndex) selectDebriefDependency(state.activeTask, nextIndex);
    event.preventDefault();
    return;
  }
  if (event.key !== "Escape") return;
  traceTooltip.hidden = true;
  if (state.cancelInteraction?.()) {
    event.preventDefault();
    return;
  }
  if (document.body.classList.contains("trace-focus")) {
    void setTraceFocus(false);
  }
  if (document.body.classList.contains("comparison-focus")) setComparisonFocus(app.querySelector(".comparison-trace-panel.is-focused"), false);
});

try {
  await loadData();
  configureGithubLinks(document);
  renderMethod();
  route({ restoreScroll: true });
} catch (error) {
  app.innerHTML = `<section class="error-page"><h1>Could not start the game</h1><p>${escapeHtml(error.message)}</p><p>Serve this directory over HTTP instead of opening index.html directly.</p></section>`;
}
