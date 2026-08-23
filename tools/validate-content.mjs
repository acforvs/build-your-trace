import { readFile } from "node:fs/promises";

const root = new URL("../", import.meta.url);
const readJson = async (path) => JSON.parse(await readFile(new URL(path, root), "utf8"));
const registry = await readJson("data/tasks.json");
const tasks = [...registry.tasks];
for (const path of registry.taskFiles || []) tasks.push(await readJson(path));
for (const path of registry.taskOverrideFiles || []) {
  const override = await readJson(path);
  const task = tasks.find((item) => item.id === override.id);
  if (!task) throw new Error(`Unknown task override: ${override.id}`);
  Object.assign(task, override);
}
for (const path of registry.taskVariantFiles || []) {
  const record = await readJson(path);
  const task = tasks.find((item) => item.id === record.taskId);
  if (!task) throw new Error(`Unknown task variant target: ${record.taskId}`);
  task.variants = [...(task.variants || []), record.variant];
}
const teaching = await readJson(registry.teachingFile);
for (const [taskId, content] of Object.entries(teaching)) {
  const task = tasks.find((item) => item.id === taskId);
  if (!task) throw new Error(`Teaching content references unknown task: ${taskId}`);
  Object.assign(task, content);
}

const errors = [];
const ids = new Set();
for (const task of tasks) {
  if (ids.has(task.id)) errors.push(`Duplicate task: ${task.id}`);
  ids.add(task.id);
  if (!task.title || !task.totalMs || (!task.catalogHidden && !task.objective)) errors.push(`${task.id}: incomplete task metadata`);
  if (!task.tracks?.length || !task.blocks?.length) errors.push(`${task.id}: missing tracks or blocks`);
  const tracks = new Set((task.tracks || []).map((track) => track.id));
  const blocks = new Set();
  for (const block of task.blocks || []) {
    if (blocks.has(block.id)) errors.push(`${task.id}: duplicate block ${block.id}`);
    blocks.add(block.id);
    if (!tracks.has(block.track)) errors.push(`${task.id}/${block.id}: unknown track`);
    if (!Number.isFinite(block.durationMs) || block.durationMs <= 0) errors.push(`${task.id}/${block.id}: invalid duration`);
  }
  for (const dependency of task.debrief?.dependencies || []) {
    if (!blocks.has(dependency.fromBlockId) || !blocks.has(dependency.toBlockId)) errors.push(`${task.id}: invalid debrief dependency`);
  }
}

const comparisons = (await readJson(registry.comparisonFile)).comparisons || [];
for (const comparison of comparisons) {
  for (const side of ["left", "right"]) {
    const taskId = comparison[side]?.taskId;
    if (taskId && !ids.has(taskId)) errors.push(`${comparison.id}: unknown ${side} task ${taskId}`);
  }
}
const collectives = (await readJson(registry.collectiveFile)).lessons || [];
if (!collectives.length) errors.push("No collective lessons found");
if (errors.length) throw new Error(errors.join("\n"));
console.log(`Validated ${tasks.length} tasks, ${comparisons.length} comparison labs, and ${collectives.length} collective lessons.`);
