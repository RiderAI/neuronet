import { build } from "esbuild";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const result = await build({
  entryPoints: [resolve(root, "src/app.ts")],
  bundle: true,
  minify: true,
  format: "iife",
  target: ["es2020"],
  write: false,
  outdir: resolve(root, ".build"),
  legalComments: "none",
});

const js = result.outputFiles?.find((file) => file.path.endsWith(".js"))?.text;
if (!js) throw new Error("esbuild не вернул JavaScript-пакет");

const [template, css] = await Promise.all([
  readFile(resolve(root, "index.template.html"), "utf8"),
  readFile(resolve(root, "src/styles.css"), "utf8"),
]);

const html = template
  .replace("/*__NEURONET_CSS__*/", () => css)
  .replace("/*__NEURONET_JS__*/", () => js);

const outDir = resolve(root, "dist");
await mkdir(outDir, { recursive: true });
await Promise.all([
  writeFile(resolve(outDir, "NeuroNet-Web.html"), html, "utf8"),
  writeFile(resolve(outDir, "index.html"), html, "utf8"),
]);

console.log(`Собрано: dist/NeuroNet-Web.html (${Buffer.byteLength(html)} байт)`);
