import { cp, lstat, mkdir, readdir, rm } from "node:fs/promises";

const root = new URL("../", import.meta.url);
const output = new URL("../dist/", import.meta.url);
await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
for (const path of ["index.html", "styles.css", "src", "data", "assets"]) {
  await cp(new URL(path, root), new URL(path, output), { recursive: true });
}

const allowedExtensions = new Set([".css", ".html", ".js", ".json", ".svg"]);
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
