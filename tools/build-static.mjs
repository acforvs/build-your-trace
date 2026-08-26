import { cp, lstat, mkdir, readFile, readdir, rm, writeFile } from "node:fs/promises";

const root = new URL("../", import.meta.url);
const output = new URL("../dist/", import.meta.url);
await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
for (const path of ["index.html", "styles.css", "src", "data", "assets"]) {
  await cp(new URL(path, root), new URL(path, output), { recursive: true });
}

const readJson = async (path) => JSON.parse(await readFile(new URL(path, root), "utf8"));
const registry = await readJson("data/tasks.json");
const tasks = [...registry.tasks];
for (const path of registry.taskFiles || []) tasks.push(await readJson(path));
for (const path of registry.taskOverrideFiles || []) {
  const override = await readJson(path);
  Object.assign(tasks.find((task) => task.id === override.id), override);
}
const escapeHtml = (value) => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
for (const task of tasks.filter((task) => !task.catalogHidden)) {
  await lstat(new URL(`assets/share-cards/${task.id}.png`, output));
  const shareDirectory = new URL(`share/${task.id}/`, output);
  await mkdir(shareDirectory, { recursive: true });
  const title = `${task.title} — distributed LLM training exercise`;
  const description = `I completed “${task.title}” — an interactive exercise in distributed LLM training.`;
  await writeFile(new URL("index.html", shareDirectory), `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${escapeHtml(title)}</title><meta name="description" content="${escapeHtml(description)}">
<meta property="og:type" content="website"><meta property="og:title" content="${escapeHtml(title)}"><meta property="og:description" content="${escapeHtml(description)}">
<meta property="og:url" content="https://trace.vladsavinov.com/share/${escapeHtml(task.id)}/"><meta property="og:image" content="https://trace.vladsavinov.com/assets/share-cards/${escapeHtml(task.id)}.png?v=3"><meta property="og:image:width" content="1200"><meta property="og:image:height" content="630">
<meta name="twitter:card" content="summary_large_image"><meta name="twitter:title" content="${escapeHtml(title)}"><meta name="twitter:description" content="${escapeHtml(description)}"><meta name="twitter:image" content="https://trace.vladsavinov.com/assets/share-cards/${escapeHtml(task.id)}.png?v=3">
<script>location.replace("../../#/task/${escapeHtml(task.id)}")</script></head>
<body><p><a href="../../#/task/${escapeHtml(task.id)}">Open the exercise</a></p></body></html>\n`);
}

const allowedExtensions = new Set([".css", ".html", ".js", ".json", ".png", ".svg"]);
async function inspect(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = new URL(entry.isDirectory() ? `${entry.name}/` : entry.name, directory);
    const info = await lstat(path);
    if (info.isSymbolicLink()) throw new Error(`Symlink is not allowed in dist: ${path.pathname}`);
    if (entry.isDirectory()) await inspect(path);
    else if (![...allowedExtensions].some((extension) => entry.name.endsWith(extension))) throw new Error(`Unexpected file in dist: ${path.pathname}`);
  }
}
await inspect(output);
console.log("Built static site in dist/");
