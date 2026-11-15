// Unit tests of the pure logic of web/label.html (the <script id="label-core"> block).
// Run: node --test tests/js/
// Set UPDATE_FIXTURE=1 to rewrite tests/data/label_tool_export.csv (checked by pytest).
import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const html = fs.readFileSync(path.join(root, "web", "label.html"), "utf8");
const src = html.match(/<script id="label-core">([\s\S]*?)<\/script>/)[1];
const ctx = {};
vm.createContext(ctx);
vm.runInContext(src, ctx);
const C = ctx.LabelCore;
const plain = (x) => JSON.parse(JSON.stringify(x));
const S = (s, e, label) => ({ s, e, label });

test("applyInterval inside an old segment splits it in three", () => {
  const out = C.applyInterval([S(0, 10, "Smooth")], 4, 6, "Rough");
  assert.deepEqual(plain(out), [S(0, 4, "Smooth"), S(4, 6, "Rough"), S(6, 10, "Smooth")]);
});

test("applyInterval straddling edges trims both neighbours", () => {
  const segs = [S(0, 5, "Smooth"), S(5, 10, "Rough")];
  const out = C.applyInterval(segs, 3, 7, "Irregularity");
  assert.deepEqual(plain(out), [S(0, 3, "Smooth"), S(3, 7, "Irregularity"), S(7, 10, "Rough")]);
});

test("applyInterval covering several segments removes them; reversed bounds ok", () => {
  const segs = [S(1, 2, "Pothole"), S(3, 4, "Rough"), S(5, 9, "Smooth")];
  const out = C.applyInterval(segs, 6, 0.5, "Smooth");
  assert.deepEqual(plain(out), [S(0.5, 9, "Smooth")]); // merged with the touching Smooth remainder
});

test("applyInterval merges touching same-label segments and drops slivers", () => {
  let segs = C.applyInterval([], 0, 2, "Rough");
  segs = C.applyInterval(segs, 2, 4, "Rough");
  assert.deepEqual(plain(segs), [S(0, 4, "Rough")]);
  segs = C.applyInterval([S(0, 4, "Smooth")], 0.005, 3.999, "Rough"); // slivers < 0.01 s dropped
  assert.deepEqual(plain(segs), [S(0.005, 3.999, "Rough")]);
  assert.throws(() => C.applyInterval([], 0, 1, "Crack"), /Unknown label/);
});

test("result is always sorted and non-overlapping (random ops)", () => {
  let segs = [];
  let seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let k = 0; k < 500; k++) {
    const a = rnd() * 100, b = a + rnd() * 10;
    const label = C.LABELS[Math.floor(rnd() * 4)];
    segs = C.applyInterval(segs, a, b, label);
    for (let i = 1; i < segs.length; i++) assert.ok(segs[i].s >= segs[i - 1].e - 1e-9);
    // the interval just applied is covered by its label from start to end
    if (b - a >= 0.01) for (const q of [a, (a + b) / 2, b - 1e-6]) assert.equal(C.labelAt(segs, q), label);
  }
});

test("fillUnlabelled fills only the gaps", () => {
  const out = C.fillUnlabelled([S(2, 3, "Pothole"), S(5, 6, "Rough")], "Smooth", 8);
  assert.deepEqual(plain(out), [S(0, 2, "Smooth"), S(2, 3, "Pothole"), S(3, 5, "Smooth"),
    S(5, 6, "Rough"), S(6, 8, "Smooth")]);
  assert.deepEqual(plain(C.fillUnlabelled([], "Rough", 4)), [S(0, 4, "Rough")]);
});

test("eventAt is ±0.4 s, clamped to the ride", () => {
  assert.deepEqual(plain(C.eventAt(5, 10)), { s: 4.6, e: 5.4 });
  assert.deepEqual(plain(C.eventAt(0.1, 10)), { s: 0, e: 0.5 });
  assert.deepEqual(plain(C.eventAt(9.9, 10)), { s: 9.5, e: 10 });
});

test("editSeg and deleteSeg", () => {
  const segs = [S(0, 4, "Smooth"), S(4, 6, "Rough"), S(6, 10, "Smooth")];
  assert.deepEqual(plain(C.deleteSeg(segs, 1)), [S(0, 4, "Smooth"), S(6, 10, "Smooth")]);
  assert.deepEqual(plain(C.editSeg(segs, 1, S(4, 6, "Smooth"))), [S(0, 10, "Smooth")]);
});

test("History undo/redo", () => {
  const h = new C.History([]);
  h.push([S(0, 1, "Rough")]);
  h.push([S(0, 2, "Rough")]);
  assert.equal(h.undo()[0].e, 1);
  assert.equal(h.undo().length, 0);
  assert.equal(h.undo(), null);
  assert.equal(h.redo()[0].e, 1);
  h.push([]);
  assert.equal(h.redo(), null); // a new edit clears the redo stack
});

function syntheticRide(fs = 50, dur = 20, bumps = [5, 5.8, 12]) {
  const t = [], ax = [], ay = [], az = [];
  for (let i = 0; i <= dur * fs; i++) {
    const ti = i / fs;
    t.push(ti); ax.push(0.1 * Math.sin(ti * 7)); ay.push(0); az.push(9.81 + 0.2 * Math.sin(ti * 13));
  }
  for (const b of bumps) az[Math.round(b * fs)] += 8;
  ay[Math.round(16 * fs)] += 6; // lateral swerve
  return { t, ax, ay, az };
}

test("findCandidates: vertical + lateral peaks, min separation 1.5 s", () => {
  const r = syntheticRide();
  const c = plain(C.findCandidates(r.t, r.ax, r.ay, r.az, 3, 1.5));
  assert.deepEqual(c.map((x) => [x.t, x.kind]), [[5, "vertical"], [12, "vertical"], [16, "lateral"]]);
  assert.equal(C.findCandidates(r.t, r.ax, r.ay, r.az, 3, 0.5).length, 4); // 5.8 kept when allowed
  assert.equal(C.findCandidates(r.t, r.ax, r.ay, r.az, 50).length, 0);
});

const RIDE_CSV = `timestamp_ms,accel_x,accel_y,accel_z,latitude,longitude,gps_accuracy_m,speed_mps,label
1700000000040,0.1,0.0,9.8,,,,,
1700000000000,0.1,0.0,9.8,53.30,-6.22,4.0,3.1,Smooth
1700000000020,0.1,0.0,9.8,,,,,Smooth
1700000000020,9,9,9,,,,,Rough
1700000000060,0.2,0.1,9.9,53.31,-6.21,4.0,3.3,Rough
`;

test("parseCanonicalCSV: sorts, de-duplicates, t from first sample, empty GPS cells", () => {
  const r = C.parseCanonicalCSV(RIDE_CSV);
  assert.deepEqual(plain(r.t), [0, 0.02, 0.04, 0.06]);
  assert.equal(r.duration, 0.06);
  assert.equal(r.ax[1], 0.1); // first of the duplicates kept
  assert.ok(Number.isNaN(r.lat[1]) && r.lat[0] === 53.3);
  assert.deepEqual(plain(r.labelCol), ["Smooth", "Smooth", "", "Rough"]);
  const segs = plain(C.labelsFromColumn(r.t, r.labelCol));
  assert.deepEqual(segs.map((g) => g.label), ["Smooth", "Rough"]);
  assert.throws(() => C.parseCanonicalCSV("Time (s),x\n0,1\n"), /missing timestamp_ms.*Found columns: time \(s\), x/);
});

test("toCSV: offset, clamping, 3 decimals, sorted; merge replaces only this ride", () => {
  const segs = [S(3, 4.5, "Rough"), S(0, 3, "Smooth"), S(9.8, 10, "Pothole")];
  assert.equal(C.toCSV("r1", segs, 0, 10),
    "ride_id,t_start_s,t_end_s,label\nr1,0.000,3.000,Smooth\nr1,3.000,4.500,Rough\nr1,9.800,10.000,Pothole\n");
  // +0.5 s: last event pushed past the end is dropped, first one clamped at 0 when negative offset
  assert.equal(C.toCSV("r1", segs, 0.5, 10).trim().split("\n").length, 3);
  assert.match(C.toCSV("r1", segs, -0.25, 10), /r1,0\.000,2\.750,Smooth/);
  const base = C.parseLabelsCSV("ride_id,t_start_s,t_end_s,label\nr2,0,1,smooth\nr1,0,5,Rough\nr0,1,2,Pothole\n");
  const merged = C.rowsToCSV(C.mergeRows(base, "r1", C.exportRows("r1", segs, 0, 10)));
  assert.equal(merged, "ride_id,t_start_s,t_end_s,label\nr0,1,2,Pothole\nr1,0.000,3.000,Smooth\n" +
    "r1,3.000,4.500,Rough\nr1,9.800,10.000,Pothole\nr2,0,1,Smooth\n");
  assert.throws(() => C.parseLabelsCSV("ride_id,t_start_s,t_end_s,label\nr1,0,1,Crack\n"), /unknown label/);
});

test("fixture export used by the Python test is up to date", () => {
  // A realistic labelling session: fill Smooth, a Rough stretch, events, an overwrite.
  const dur = 60;
  let segs = C.fillUnlabelled([], "Smooth", dur);
  segs = C.applyInterval(segs, 20, 35, "Rough");
  for (const t of [8.3, 41.0]) { const e = C.eventAt(t, dur); segs = C.applyInterval(segs, e.s, e.e, "Pothole"); }
  { const e = C.eventAt(27.5, dur); segs = C.applyInterval(segs, e.s, e.e, "Irregularity"); }
  { const e = C.eventAt(59.9, dur); segs = C.applyInterval(segs, e.s, e.e, "Irregularity"); }
  segs = C.applyInterval(segs, 33, 36, "Smooth"); // overwrite the end of the Rough stretch
  const csv = C.toCSV("tool_ride", segs, 0, dur);
  const fixture = path.join(root, "tests", "data", "label_tool_export.csv");
  if (process.env.UPDATE_FIXTURE) fs.writeFileSync(fixture, csv);
  assert.equal(fs.readFileSync(fixture, "utf8"), csv);
});
