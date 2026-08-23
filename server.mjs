import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, isAbsolute, normalize, relative, resolve } from "node:path";

const host = process.env.HOST || "0.0.0.0";
const port = Number(process.env.PORT || 4173);
const root = process.cwd();
const types = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
};

createServer((request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, `http://${request.headers.host || "localhost"}`).pathname);
    const relativePath = normalize(pathname === "/" ? "index.html" : pathname).replace(/^[/\\]+/, "");
    const filePath = resolve(root, relativePath);
    const pathFromRoot = relative(root, filePath);
    if (pathFromRoot.startsWith("..") || isAbsolute(pathFromRoot)) {
      response.writeHead(403, { "Content-Type": "text/plain; charset=utf-8" }).end("Forbidden");
      return;
    }
    if (!statSync(filePath).isFile()) throw new Error("Not a file");
    response.writeHead(200, {
      "Content-Type": types[extname(filePath)] || "application/octet-stream",
      "X-Content-Type-Options": "nosniff",
    });
    createReadStream(filePath).pipe(response);
  } catch {
    response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" }).end("Not found");
  }
}).listen(port, host, () => {
  console.log(`Build Your Trace is running at http://${host}:${port}`);
});
