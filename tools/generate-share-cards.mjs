import { mkdir, readFile, writeFile } from "node:fs/promises";

const root = new URL("../", import.meta.url);
const readJson = async (path) => JSON.parse(await readFile(new URL(path, root), "utf8"));
const registry = await readJson("data/tasks.json");
const tasks = [...registry.tasks];
for (const path of registry.taskFiles || []) tasks.push(await readJson(path));
for (const path of registry.taskOverrideFiles || []) {
  const override = await readJson(path);
  Object.assign(tasks.find((task) => task.id === override.id), override);
}

const cardCopy = {
  "single-gpu-forward-backward": ["Single-GPU forward + backward", "One dense Transformer layer"],
  "single-gpu-moe": ["MoE forward on one GPU", "One Qwen3 MoE layer"],
  "moe-expert-parallel": ["MoE + expert parallelism", "Dispatch, expert compute, and combine"],
  "moe-expert-parallel-backward": ["MoE + EP backward", "Expert-backward dependency window"],
  "ddp-terminal": ["DDP with terminal gradient sync", "Two dense Transformer layers"],
  "ddp-bucketed": ["DDP buckets overlapping backward", "Two dense Transformer layers"],
  "tensor-parallel-dense": ["Tensor-parallel dense layer", "Forward and backward"],
  "tensor-sequence-parallel": ["Tensor + sequence parallelism", "Forward and backward"],
  "tensor-parallel-async": ["Async tensor parallelism", "Pipelined SwiGLU gathering"],
  "fsdp-dense": ["FSDP dense layer + prefetch", "Two dense Transformer layers"],
  "fsdp-dense-granular": ["Granular FSDP", "Separate attention and MLP units"],
  "fsdp-moe": ["FSDP over MoE", "Two-layer MoE"],
  "fsdp-moe-ep": ["FSDP + expert parallelism", "Two-layer MoE"],
  "context-parallel-allgather": ["Context parallelism · K/V AllGather", "Ring attention"],
  "context-parallel-alltoall": ["Context parallelism · cyclic AllToAll", "Ring attention"],
  "context-parallel-multirank": ["Context parallelism · multi-rank view", "Attention-forward slice"]
};

const windows = {
  "fsdp-moe": [72, 154.483078],
  "fsdp-moe-ep": [24.8, 51.2]
};

const colors = {
  attention: ["#64b5f6", "#1675bb"],
  mlp: ["#ffb74d", "#b85e00"],
  collective: ["#ba68c8", "#7b2988"],
  routing: ["#4db6ac", "#087d74"],
  loss: ["#e57373", "#a52f2f"],
  bubble: ["#b0bec5", "#52636b"]
};

const escapeXml = (value) => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const compactLabel = (block) => (block.shortLabel || block.label)
  .split("·")[0]
  .replaceAll("Backward", "BWD")
  .replaceAll("Forward", "FWD")
  .replaceAll("Attention", "Attn")
  .replaceAll("All-gather", "AG")
  .replaceAll("all-gather", "AG")
  .replaceAll("Reduce-scatter", "RS")
  .replaceAll("reduce-scatter", "RS")
  .trim();

function renderCard(task) {
  const [title, subtitle] = cardCopy[task.id] || [task.title, task.scope];
  const [startMs, endMs] = windows[task.id] || [0, task.totalMs];
  const traceX = 262;
  const traceWidth = 858;
  const traceY = 238;
  const traceHeight = 284;
  const laneHeight = traceHeight / task.tracks.length;
  const titleSize = title.length > 43 ? 39 : title.length > 34 ? 44 : 50;
  const rects = [];
  const labels = [];
  const clips = [];
  let clipIndex = 0;

  for (const block of task.blocks) {
    const visibleStart = Math.max(startMs, block.startMs);
    const visibleEnd = Math.min(endMs, block.startMs + block.durationMs);
    if (visibleEnd <= visibleStart) continue;
    const trackIndex = task.tracks.findIndex((track) => track.id === block.track);
    if (trackIndex < 0) continue;
    const x = traceX + ((visibleStart - startMs) / (endMs - startMs)) * traceWidth;
    const honestWidth = ((visibleEnd - visibleStart) / (endMs - startMs)) * traceWidth;
    const width = Math.max(3, honestWidth);
    const height = Math.min(36, laneHeight - 18);
    const y = traceY + trackIndex * laneHeight + (laneHeight - height) / 2;
    const [fill, stroke] = colors[block.type] || colors.bubble;
    const clipId = `block-${clipIndex++}`;
    clips.push(`<clipPath id="${clipId}"><rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${width.toFixed(1)}" height="${height.toFixed(1)}" rx="3"/></clipPath>`);
    rects.push(`<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${width.toFixed(1)}" height="${height.toFixed(1)}" rx="3" fill="${fill}" stroke="${stroke}" stroke-width="1.2"/>`);
    const label = compactLabel(block);
    const estimatedWidth = label.length * 6.25 + 14;
    if (width >= Math.max(36, estimatedWidth)) {
      labels.push(`<text x="${(x + width / 2).toFixed(1)}" y="${(y + height / 2 + 4).toFixed(1)}" text-anchor="middle" clip-path="url(#${clipId})">${escapeXml(label)}</text>`);
    }
  }

  const laneLines = task.tracks.slice(1).map((_, index) => {
    const y = traceY + (index + 1) * laneHeight;
    return `<line x1="84" y1="${y.toFixed(1)}" x2="1116" y2="${y.toFixed(1)}"/>`;
  }).join("");
  const trackLabels = task.tracks.map((track, index) => {
    const y = traceY + index * laneHeight + laneHeight / 2 + 5;
    const label = track.label.replace(/^FSDP parameter /, "FSDP ").replace(/^EP activation /, "EP ");
    return `<text x="102" y="${y.toFixed(1)}">${escapeXml(label)}</text>`;
  }).join("");

  return `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630" role="img" aria-labelledby="title desc">
  <title id="title">Completed a distributed LLM training exercise</title>
  <desc id="desc">${escapeXml(title)}, shown with a task-specific trace excerpt.</desc>
  <defs>${clips.join("")}</defs>
  <rect width="1200" height="630" fill="#f4f4f1"/>
  <rect x="42" y="42" width="1116" height="546" rx="9" fill="#fff" stroke="#d7d8d5" stroke-width="2"/>
  <g font-family="Inter, Arial, sans-serif">
    <text x="82" y="91" fill="#5f656a" font-size="17" font-weight="700" letter-spacing="2.2">DISTRIBUTED LLM TRAINING · COMPLETED</text>
    <text x="82" y="158" fill="#202326" font-size="${titleSize}" font-weight="760" letter-spacing="-1.8">${escapeXml(title)}</text>
    <text x="84" y="197" fill="#62676c" font-size="21">${escapeXml(subtitle)}</text>
    <rect x="84" y="238" width="1032" height="284" rx="5" fill="#fafafa" stroke="#dfe0dd"/>
    <g stroke="#e5e6e3" stroke-width="1"><line x1="262" y1="238" x2="262" y2="522"/>${laneLines}</g>
    <g fill="#61676c" font-size="14" font-weight="650">${trackLabels}</g>
    <g>${rects.join("")}</g>
    <g fill="#253035" font-size="11" font-weight="700">${labels.join("")}</g>
    <g transform="translate(84 550)">
      <rect width="18" height="8" rx="2" fill="#64b5f6"/><rect x="23" width="18" height="8" rx="2" fill="#ba68c8"/><rect x="46" width="18" height="8" rx="2" fill="#ffb74d"/>
      <text x="77" y="9" fill="#6b7176" font-size="15">trace.vladsavinov.com</text>
      <text x="1032" y="9" text-anchor="end" fill="#6b7176" font-size="15">Trace exercise</text>
    </g>
  </g>
</svg>\n`;
}

const output = new URL("assets/share-cards/", root);
await mkdir(output, { recursive: true });
for (const task of tasks.filter((task) => !task.catalogHidden)) {
  await writeFile(new URL(`${task.id}.svg`, output), renderCard(task));
}
console.log(`Generated ${tasks.filter((task) => !task.catalogHidden).length} task-specific SVG cards.`);
