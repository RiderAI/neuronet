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
  outdir: resolve(root, ".build-classic"),
  legalComments: "none",
});

const js = result.outputFiles?.find((file) => file.path.endsWith(".js"))?.text;
if (!js) throw new Error("esbuild не вернул JavaScript-пакет");

const [template, baseCss, classicCss, contemporaryCss, studioCss] = await Promise.all([
  readFile(resolve(root, "index.template.html"), "utf8"),
  readFile(resolve(root, "src/styles.css"), "utf8"),
  readFile(resolve(root, "src/styles-classic.css"), "utf8"),
  readFile(resolve(root, "src/styles-contemporary.css"), "utf8"),
  readFile(resolve(root, "src/styles-studio.css"), "utf8"),
]);

const html = template
  .replace('<html lang="ru">', '<html lang="ru" data-ui="classic" data-skin="studio">')
  .replace("NeuroNet Online — учебная лаборатория", "NeuroNet Online")
  .replace("/*__NEURONET_CSS__*/", () => `${baseCss}\n${classicCss}\n${contemporaryCss}\n${studioCss}`)
  .replace("/*__NEURONET_JS__*/", () => js);

const outDir = resolve(root, "dist");
const outFile = resolve(outDir, "NeuroNet-Web-Classic.html");
await mkdir(outDir, { recursive: true });
await Promise.all([
  writeFile(outFile, html, "utf8"),
  writeFile(resolve(outDir, "NeuroNet-Studio.html"), html, "utf8"),
  writeFile(resolve(outDir, "NeuroNet-Online.html"), html, "utf8"),
  writeFile(resolve(outDir, "index.html"), html, "utf8"),
]);

console.log(`Собрано: dist/NeuroNet-Web-Classic.html (${Buffer.byteLength(html)} байт)`);
