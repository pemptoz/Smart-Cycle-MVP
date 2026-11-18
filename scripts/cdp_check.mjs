// Headless Chrome check of a page through the DevTools protocol (no npm dependency, Node >= 22).
// Prints console errors / exceptions, every request host, the dashboard self-test result and
// optionally saves a screenshot.
//   node scripts/cdp_check.mjs --url 'http://127.0.0.1:8000/?selftest=1' --wait-selftest
//   node scripts/cdp_check.mjs --url 'http://127.0.0.1:8000/?view=partner' --shot out.png --dark
//   node scripts/cdp_check.mjs --upload '#csvFile=/abs/ride.csv' --eval 'document.querySelector("#uploadForm").requestSubmit()'
import { spawn } from "node:child_process";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const flag = (k) => args.includes(k);
const url = opt("--url", "http://127.0.0.1:8000/");
const shot = opt("--shot", null);
const waitMs = Number(opt("--wait-ms", "4000"));
const [W, H] = opt("--size", "1400x900").split("x").map(Number);
const chrome = opt("--chrome", process.env.CHROME || "google-chrome");
const port = 9300 + Math.floor(Math.random() * 500);
const prof = mkdtempSync(join(tmpdir(), "cdp-"));

const proc = spawn(chrome, ["--headless=new", "--disable-gpu", "--no-first-run",
  "--no-default-browser-check", `--remote-debugging-port=${port}`, `--user-data-dir=${prof}`,
  `--window-size=${W},${H}`, "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let ws;
for (let i = 0; i < 50 && !ws; i++) {
  await sleep(200);
  try {
    const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    const page = list.find((t) => t.type === "page");
    if (page) ws = new WebSocket(page.webSocketDebuggerUrl);
  } catch { /* not up yet */ }
}
if (!ws) { console.error("Chrome did not start"); proc.kill(); process.exit(2); }
await new Promise((r) => ws.addEventListener("open", r, { once: true }));

let id = 0;
const pending = new Map();
const problems = [];
const hosts = new Map();
ws.addEventListener("message", (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); return; }
  if (msg.method === "Runtime.consoleAPICalled" && ["error", "warning", "assert"].includes(msg.params.type)) {
    problems.push(`console.${msg.params.type}: ${msg.params.args.map((a) => a.value ?? a.description).join(" ")}`);
  } else if (msg.method === "Runtime.exceptionThrown") {
    const d = msg.params.exceptionDetails;
    problems.push(`exception: ${d.exception?.description || d.text}`);
  } else if (msg.method === "Log.entryAdded" && ["error", "warning"].includes(msg.params.entry.level)) {
    problems.push(`log.${msg.params.entry.level}: ${msg.params.entry.text} ${msg.params.entry.url || ""}`);
  } else if (msg.method === "Network.requestWillBeSent") {
    const u = msg.params.request.url;
    if (!u.startsWith("data:")) {
      const h = new URL(u).host;
      hosts.set(h, (hosts.get(h) || 0) + 1);
    }
  }
});
const send = (method, params = {}) => new Promise((res) => {
  const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params }));
});
const evaluate = async (expr) => (await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true })).result?.result?.value;

await send("Runtime.enable"); await send("Log.enable"); await send("Network.enable"); await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 1, mobile: false });
const media = [];
if (flag("--dark")) media.push({ name: "prefers-color-scheme", value: "dark" });
if (flag("--light")) media.push({ name: "prefers-color-scheme", value: "light" });
if (flag("--reduced-motion")) media.push({ name: "prefers-reduced-motion", value: "reduce" });
if (media.length) await send("Emulation.setEmulatedMedia", { features: media });
await send("Page.navigate", { url });

let selftest = null;
if (flag("--wait-selftest")) {
  for (let i = 0; i < 120; i++) {
    await sleep(500);
    const t = await evaluate("document.getElementById('selftest')?.textContent || ''");
    if (t && t.includes("SELFTEST DONE")) { selftest = t; break; }
  }
} else {
  await sleep(waitMs);
}
const upload = opt("--upload", null); // "<css selector>=<file path>" sets a file input
if (upload) {
  const [sel, file] = upload.split("=");
  const doc = await send("DOM.getDocument", {});
  const q = await send("DOM.querySelector", { nodeId: doc.result.root.nodeId, selector: sel });
  await send("DOM.setFileInputFiles", { nodeId: q.result.nodeId, files: [file] });
}
const js = opt("--eval", null);
if (js) { console.log("eval:", JSON.stringify(await evaluate(js))); await sleep(Number(opt("--after-eval-ms", "1500"))); }
if (shot) {
  const r = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(shot, Buffer.from(r.result.data, "base64"));
  console.log(`screenshot: ${shot}`);
}
console.log("requests by host:", JSON.stringify(Object.fromEntries(hosts)));
console.log(problems.length ? `console problems (${problems.length}):\n  ${problems.join("\n  ")}` : "console problems: none");
if (flag("--wait-selftest")) console.log(selftest ?? "SELFTEST: no result (timeout)");
ws.close(); proc.kill();
await sleep(300);
try { rmSync(prof, { recursive: true, force: true }); } catch { /* ignore */ }
const failed = (selftest === null && flag("--wait-selftest")) || /FAIL/.test(selftest || "");
process.exit(failed ? 1 : 0);
