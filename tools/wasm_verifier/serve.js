const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");

const root = __dirname;
const contentTypes = {
  ".html": "text/html; charset=utf-8",
  ".wasm": "application/wasm",
};

http
  .createServer((request, response) => {
    const pathname = new URL(request.url, "http://localhost").pathname;
    const relativePath = pathname === "/" ? "index.html" : decodeURIComponent(pathname.slice(1));
    const filePath = path.resolve(root, relativePath);
    if (!filePath.startsWith(`${root}${path.sep}`)) {
      response.writeHead(403).end("Forbidden");
      return;
    }
    fs.readFile(filePath, (error, contents) => {
      if (error) {
        response.writeHead(404).end("Not found");
        return;
      }
      response.setHeader("Content-Type", contentTypes[path.extname(filePath)] ?? "application/octet-stream");
      response.end(contents);
    });
  })
  .listen(8000, "127.0.0.1", () => console.log("Open http://127.0.0.1:8000 to run the NEUR Wasm verifier."));
