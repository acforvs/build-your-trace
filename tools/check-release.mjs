import { lstat, readFile, readdir } from "node:fs/promises";
import { extname, relative } from "node:path";

const root = new URL("../", import.meta.url);
const ignored = new Set([".git", "dist"]);
const allowedTopLevel = new Set([".github", ".gitignore", "README.md", "assets", "benchmarks", "data", "index.html", "netlify.toml", "package.json", "schemas", "server.mjs", "src", "styles.css", "tools"]);
const rejectedExtensions = new Set([".nsys-rep", ".sqlite", ".pyc", ".pyo"]);
const errors = [];

async function inspect(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    if (ignored.has(entry.name)) continue;
    const path = new URL(entry.isDirectory() ? `${entry.name}/` : entry.name, directory);
    const name = relative(root.pathname, path.pathname).replace(/\/$/, "");
    if (!name.includes("/") && !allowedTopLevel.has(name)) errors.push(`Unexpected top-level entry: ${name}`);
    const info = await lstat(path);
    if (info.isSymbolicLink()) { errors.push(`Symlink is not allowed: ${name}`); continue; }
    if (entry.isDirectory()) { await inspect(path); continue; }
    if (rejectedExtensions.has(extname(entry.name)) || /perfetto.*\.json(?:\.gz)?$/i.test(entry.name)) errors.push(`Generated profiler file: ${name}`);
    const content = await readFile(path);
    if (content.includes(0)) continue;
    const text = content.toString("utf8");
    if (/\/home\/[A-Za-z0-9._-]+\//.test(text)) errors.push(`Developer home path: ${name}`);
    if (/[a-f0-9]{64}/i.test(text)) errors.push(`Embedded digest value: ${name}`);
  }
}
await inspect(root);
if (errors.length) { console.error(errors.join("\n")); process.exit(1); }
console.log("Release tree contains only reviewed source and no generated profiler files.");
