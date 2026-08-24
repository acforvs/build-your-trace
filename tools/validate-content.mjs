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
  for (const block of task.blocks || []) {
    for (const slotId of block.acceptedSlotIds || []) {
      const slotBlock = task.blocks.find((item) => item.id === slotId);
      if (task.timelineMode !== "absolute" || !slotBlock || slotBlock.fixed || slotBlock.track !== block.track) errors.push(`${task.id}/${block.id}: invalid accepted slot ${slotId}`);
    }
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
const recognition = (await readJson(registry.recognitionFile)).rounds || [];
const recognitionIds = new Set();
const recognitionOrders = new Set();
for (const round of recognition) {
  if (recognitionIds.has(round.id)) errors.push(`Duplicate recognition round: ${round.id}`);
  if (recognitionOrders.has(round.order)) errors.push(`Duplicate recognition order: ${round.order}`);
  recognitionIds.add(round.id);
  recognitionOrders.add(round.order);
  if (!/^mystery-[0-9]{2}$/.test(round.id) || !/^Mystery trace [0-9]{2}$/.test(round.title)) errors.push(`${round.id}: recognition metadata may reveal the answer`);
  const source = tasks.find((task) => task.id === round.trace?.taskId);
  if (!source) { errors.push(`${round.id}: unknown task ${round.trace?.taskId}`); continue; }
  if (!round.trace.context?.trim()) errors.push(`${round.id}: missing answer-neutral window context`);
  let task = source;
  if (round.trace.variantId) {
    const variant = source.variants?.find((item) => item.id === round.trace.variantId);
    if (!variant) { errors.push(`${round.id}: unknown variant ${round.trace.variantId}`); continue; }
    task = { ...source, ...variant, blocks: source.blocks.map((block) => ({ ...block, ...(variant.blockOverrides?.[block.id] || {}) })) };
  }
  if (task.verification?.status !== "measured") errors.push(`${round.id}: recognition source must be measured`);
  const { startMs, endMs } = round.trace.window || {};
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || startMs < 0 || endMs <= startMs || endMs > task.totalMs) errors.push(`${round.id}: window is outside the source trace`);
  const trackIds = new Set(task.tracks.map((track) => track.id));
  if (new Set(round.trace.tracks).size !== round.trace.tracks.length) errors.push(`${round.id}: duplicate track ids`);
  for (const trackId of round.trace.tracks) if (!trackIds.has(trackId)) errors.push(`${round.id}: unknown track ${trackId}`);
  const visibleBlocks = new Set(task.blocks.filter((block) => round.trace.tracks.includes(block.track) && block.startMs < endMs && block.startMs + block.durationMs > startMs).map((block) => block.id));
  for (const blockId of Object.keys(round.trace.anonymizedBlockLabels || {})) if (!visibleBlocks.has(blockId)) errors.push(`${round.id}: anonymized label block ${blockId} is not visible in the window`);
  if (!round.evidence?.answerBlockIds?.length) errors.push(`${round.id}: no evidence answers`);
  if (new Set(round.evidence?.answerBlockIds || []).size !== (round.evidence?.answerBlockIds || []).length) errors.push(`${round.id}: duplicate evidence answers`);
  for (const blockId of round.evidence?.answerBlockIds || []) if (!visibleBlocks.has(blockId)) errors.push(`${round.id}: evidence block ${blockId} is not visible in the window`);
  const optionIds = (round.choice?.options || []).map((option) => option.id);
  if (optionIds.length < 3 || optionIds.length > 4 || new Set(optionIds).size !== optionIds.length) errors.push(`${round.id}: choice must have 3–4 unique options`);
  if (!optionIds.includes(round.choice?.answerOptionId)) errors.push(`${round.id}: answer option does not exist`);
  for (const option of round.choice?.options || []) if (option.id !== round.choice.answerOptionId && !option.nudge) errors.push(`${round.id}: distractor ${option.id} needs a nudge`);
  const evidenceOptionIds = (round.evidence?.options || []).map((option) => option.id);
  if (evidenceOptionIds.length < 3 || evidenceOptionIds.length > 4 || new Set(evidenceOptionIds).size !== evidenceOptionIds.length) errors.push(`${round.id}: evidence must have 3–4 unique options`);
  if (!evidenceOptionIds.includes(round.evidence?.answerOptionId)) errors.push(`${round.id}: evidence answer option does not exist`);
  for (const option of round.evidence?.options || []) if (option.id !== round.evidence.answerOptionId && !option.nudge) errors.push(`${round.id}: evidence distractor ${option.id} needs a nudge`);
}
const collectives = (await readJson(registry.collectiveFile)).lessons || [];
if (!collectives.length) errors.push("No collective lessons found");
if (errors.length) throw new Error(errors.join("\n"));
console.log(`Validated ${tasks.length} tasks, ${comparisons.length} comparison labs, ${recognition.length} recognition rounds, and ${collectives.length} collective lessons.`);
