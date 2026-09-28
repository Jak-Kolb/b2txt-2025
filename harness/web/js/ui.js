// Shared DOM, formatting, flag, and word-diff helpers. All data text goes through textContent.

export async function getJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  return response.json();
}

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

// replaceChildren that flattens arrays and skips null/false (like el()).
export function fill(node, ...children) {
  node.replaceChildren(...children.flat(Infinity).filter((c) => c != null && c !== false));
  return node;
}

export const fmt = (v, digits = 2) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(digits));
export const fmtCI = (ci, digits = 2) => (ci ? `[${fmt(ci[0], digits)}, ${fmt(ci[1], digits)}]` : "–");
export const shortTime = (iso) => (iso ? iso.replace("T", " ").slice(0, 16) : "");

export const FLAG_TEXT = {
  acoustic_selected_on_scope: "Checkpoint chosen by val PER over sessions that include this scope",
  acoustic_selection_unknown: "Checkpoint selection procedure is not recorded",
  decode_tuned_on_scope: "Decode settings were swept on this partition",
  decode_tuned_for_other_acoustic: "Decode settings were tuned for a different acoustic model",
  decode_tuning_unrecorded: "Decode overrides without a tuning record",
  preprocess_mismatch: "Inference smoothing differs from the model's training smoothing",
  offline_noncausal: "Non-causal smoothing: offline accuracy reference only, no timing",
  exposed_partition: "Former val-test: influenced checkpoint selection (exposed)",
  smoke_limited: "Smoke run on a limited trial subset",
};
const WARN_FLAGS = new Set(["preprocess_mismatch", "offline_noncausal", "exposed_partition", "smoke_limited",
                            "decode_tuned_for_other_acoustic", "decode_tuning_unrecorded"]);

export function flagBadges(flags = []) {
  return flags.map((flag) => el("span", {
    class: `badge${WARN_FLAGS.has(flag) ? " warn" : ""}`, title: FLAG_TEXT[flag] || flag, text: flag.replaceAll("_", " "),
  }));
}

// Mirrors stream_lm.norm_words: lowercase, punctuation (except apostrophes) to space.
export function normWords(text) {
  return (text || "").toLowerCase().replace(/[^\p{L}\p{N}_\s']/gu, " ").split(/\s+/).filter(Boolean);
}

// Minimum-edit alignment of hypothesis to reference: [{op, ref, hyp}], op in ok/sub/ins/del.
export function alignWords(ref, hyp, { prefix = false } = {}) {
  const n = ref.length, m = hyp.length;
  const d = Array.from({ length: n + 1 }, (_, i) => Array.from({ length: m + 1 }, (_, j) => (j === 0 ? i : i === 0 ? j : 0)));
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1));
    }
  }
  // Prefix mode (live display only): unspoken reference tail is free, so end at the best row
  // (ties prefer the longer prefix, so the newest word reads as a substitution, not an insertion).
  let i = n;
  if (prefix) {
    let best = 0;
    for (let r = 0; r <= n; r++) if (d[r][m] <= d[best][m]) best = r;
    i = best;
  }
  let j = m;
  const out = [];
  while (i > 0 || j > 0) {
    if (i > 0 && j > 0 && d[i][j] === d[i - 1][j - 1] + (ref[i - 1] === hyp[j - 1] ? 0 : 1)) {
      out.push({ op: ref[i - 1] === hyp[j - 1] ? "ok" : "sub", ref: ref[i - 1], hyp: hyp[j - 1] }); i--; j--;
    } else if (j > 0 && d[i][j] === d[i][j - 1] + 1) {
      out.push({ op: "ins", hyp: hyp[j - 1] }); j--;
    } else {
      out.push({ op: "del", ref: ref[i - 1] }); i--;
    }
  }
  return out.reverse();
}

// Hypothesis words marked against the reference (substitution / insertion washes, deletions struck).
export function diffLine(reference, hypothesis, opts) {
  const ops = alignWords(normWords(reference), normWords(hypothesis), opts);
  return el("div", { class: "words" }, ops.map((o) => {
    if (o.op === "ok") return [el("span", { class: "w", text: o.hyp }), " "];
    if (o.op === "sub") return [el("span", { class: "w sub", title: `reference: ${o.ref}`, text: o.hyp }), " "];
    if (o.op === "ins") return [el("span", { class: "w ins", title: "insertion", text: o.hyp }), " "];
    return [el("span", { class: "w del", title: "deletion", text: o.ref }), " "];
  }));
}

export function diffLegend() {
  return el("div", { class: "legend" },
    el("span", {}, el("span", { class: "key", style: "background: var(--wash-sub)" }), "substitution"),
    el("span", {}, el("span", { class: "key", style: "background: var(--wash-ins)" }), "insertion"),
    el("span", {}, el("span", { class: "w del", text: "word" }), " deletion"));
}

export function tile(label, value, detail) {
  return el("div", { class: "tile" }, el("div", { class: "label", text: label }),
    el("div", { class: "value", text: value }), detail ? el("div", { class: "detail", text: detail }) : null);
}

export function kv(pairs) {
  return el("dl", { class: "kv" }, pairs.map(([k, v]) => [el("dt", { text: k }), el("dd", {}, v)]));
}
