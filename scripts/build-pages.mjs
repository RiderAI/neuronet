import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import "./build-classic.mjs";

// Publish the current windowed interface, not the alternative dist/index.html.
// Only these standalone files belong to the public website.
const root = resolve(import.meta.dirname, "..");
const html = await readFile(resolve(root, "dist/NeuroNet-Web-Classic.html"));
const output = resolve(root, "dist-pages");
await mkdir(output, { recursive: true });
await Promise.all([
  writeFile(resolve(output, "index.html"), html),
  writeFile(resolve(output, "NeuroNet-Web-Classic.html"), html),
  writeFile(resolve(output, ".nojekyll"), ""),
]);
console.log(`GitHub Pages: dist-pages/index.html (${html.length} bytes)`);
