import React, { useEffect, useMemo, useRef, useState } from "react";

/**
 * Single-cell RNA-seq ML pipeline (frontend-only shell)
 * - No external UI libraries
 * - Works on a fresh Vite + React + TS template
 * - Can optionally call a backend if you set an API base URL in the UI
 */

const APP_VERSION = "0.2.0";

type PhaseKey =
  | "upload"
  | "quality_control"
  | "normalization"
  | "batch_correction"
  | "clustering"
  | "ml_training"
  | "results";

type Phase = {
  key: PhaseKey;
  title: string;
  subtitle: string;
};

type StepStatus = "idle" | "running" | "done" | "error";

type RunResult = {
  ok: boolean;
  message: string;
  payload?: unknown;
};


function isMissingRunId422(payload: any) {
  const detail = payload?.detail;
  if (!Array.isArray(detail)) return false;
  return detail.some((d: any) => {
    const loc = Array.isArray(d?.loc) ? d.loc.join(".") : "";
    const msg = typeof d?.msg === "string" ? d.msg.toLowerCase() : "";
    const typ = typeof d?.type === "string" ? d.type.toLowerCase() : "";
    return loc === "body.run_id" && (typ === "missing" || msg.includes("field required"));
  });
}



function isMissingFile422(payload: any) {
  const detail = payload?.detail;
  if (!Array.isArray(detail)) return false;
  return detail.some((d: any) => {
    const loc = Array.isArray(d?.loc) ? d.loc.join(".") : "";
    const msg = typeof d?.msg === "string" ? d.msg.toLowerCase() : "";
    const typ = typeof d?.type === "string" ? d.type.toLowerCase() : "";
    return loc === "body.file" && (typ === "missing" || msg.includes("field required"));
  });
}

function isRunIdNotFound(payload: any) {
  const d = payload?.detail;
  return typeof d === "string" && d.toLowerCase().includes("run_id not found");
}

const PHASES: Phase[] = [
  { key: "upload", title: "Upload Data", subtitle: "Load counts matrix" },
  { key: "quality_control", title: "Quality Control", subtitle: "Checks and filtering" },
  { key: "normalization", title: "Normalization", subtitle: "Log-normalize and scale" },
  { key: "batch_correction", title: "Batch Correction", subtitle: "Harmony in PC space" },
  { key: "clustering", title: "Clustering", subtitle: "Dimensionality and clusters" },
  { key: "ml_training", title: "ML Training", subtitle: "Train and inspect" },
  { key: "results", title: "Results", subtitle: "Export reports" },
];

function clamp(n: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, n));
}

function prettyBytes(n: number) {
  const units = ["B", "KB", "MB", "GB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function sleep(ms: number) {
  return new Promise((r) => setTimeout(r, ms));
}

async function safeJson(res: Response) {
  const text = await res.text().catch(() => "");
  try {
    return text ? JSON.parse(text) : null;
  } catch {
    return { raw: text };
  }
}

/**
 * Small helpers used by parseQuickMatrixInfo.
 * Kept pure so they can be sanity-checked without File APIs.
 */
function splitNonEmptyLines(text: string) {
  // Correct newline handling for Windows (\r\n) and Unix (\n)
  return text.split(/\r?\n/).filter((x) => x.trim().length > 0);
}

function inferSeparator(headerLine: string) {
  return headerLine.includes("\t") ? "\t" : ",";
}

function estimateMatrixShape(lines: string[]) {
  if (lines.length === 0) return { rows: undefined as number | undefined, cols: undefined as number | undefined };

  const sep = inferSeparator(lines[0]);
  const first = lines[0].split(sep);
  const cols = Math.max(0, first.length - 1); // assume first col is gene id
  const rows = Math.max(0, lines.length - 1); // minus header
  return { rows, cols };
}

function sanitizeApiBase(apiBase: string) {
  // Allow pasting the Swagger UI URL. Strip fragments and common doc suffixes.
  let base = apiBase.trim();
  base = base.replace(/#.*$/, "");
  base = base.replace(/\/openapi\.json\/?$/, "");
  base = base.replace(/\/docs\/?$/, "");
  base = base.replace(/\/+$/, "");
  return base;
}

function buildUrl(apiBase: string, path: string) {
  const base = sanitizeApiBase(apiBase);
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${base}${p}`;
}

let __didSelfTest = false;
function __devSelfTest() {
  // Minimal sanity checks to prevent regressions like broken regex literals.
  const t1 = "gene,cell1,cell2\nG1,1,2\nG2,3,4\n";
  const l1 = splitNonEmptyLines(t1);
  if (l1.length !== 3) throw new Error(`selftest: expected 3 lines, got ${l1.length}`);
  const s1 = estimateMatrixShape(l1);
  if (s1.rows !== 2 || s1.cols !== 2) throw new Error(`selftest: expected rows=2 cols=2, got ${s1.rows} ${s1.cols}`);

  const t2 = "gene\tcell1\tcell2\r\nG1\t1\t2\r\n";
  const l2 = splitNonEmptyLines(t2);
  if (l2.length !== 2) throw new Error(`selftest: expected 2 lines, got ${l2.length}`);
  const s2 = estimateMatrixShape(l2);
  if (s2.rows !== 1 || s2.cols !== 2) throw new Error(`selftest: expected rows=1 cols=2, got ${s2.rows} ${s2.cols}`);
  const u1 = buildUrl("https://x.test/", "/normalize");
  if (u1 !== "https://x.test/normalize") throw new Error(`selftest: bad url build u1=${u1}`);

  const u2 = buildUrl(" https://x.test ", "health");
  if (u2 !== "https://x.test/health") throw new Error(`selftest: bad url build u2=${u2}`);
}

async function runWithBackend(opts: {
  apiBase: string;
  path: string;
  method?: "GET" | "POST";
  json?: any;
  form?: FormData;
  timeoutMs?: number;
}): Promise<RunResult> {
  const { apiBase, path, method = "POST", json, form, timeoutMs = 120000 } = opts;

  if (!apiBase.trim()) {
    await sleep(600 + Math.random() * 900);
    return { ok: true, message: "Completed (simulated). No API base URL set." };
  }

  const url = buildUrl(apiBase, path);

  const controller = new AbortController();
  const t = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    // Important: when sending FormData, do NOT set Content-Type manually.
    // Also avoid passing `headers: undefined`.
    const init: RequestInit = {
      method,
      // For multipart FormData, do not set Content-Type. Let the browser set multipart boundaries.
      headers: form
        ? { Accept: "application/json" }
        : { Accept: "application/json", "Content-Type": "application/json" },
      body: form ? form : json ? JSON.stringify(json) : undefined,
      signal: controller.signal,
      mode: "cors",
      redirect: "follow",
    };

    const res = await fetch(url, init);

    const data = await safeJson(res);
    if (!res.ok) {
      return {
        ok: false,
        message: `API error ${res.status}: ${typeof data === "string" ? data : JSON.stringify(data)}`,
        payload: data,
      };
    }

    return { ok: true, message: "Completed (API).", payload: data };
  } catch (e: any) {
    const msg =
      e?.name === "AbortError"
        ? "Request timed out."
        : e?.message
        ? e.message
        : "Network error.";
    return { ok: false, message: msg };
  } finally {
    window.clearTimeout(t);
  }
}

function Icon({ name, active }: { name: PhaseKey; active?: boolean }) {
  const stroke = active ? "#1d4ed8" : "#94a3b8";
  const fill = "none";

  const common = {
    width: 22,
    height: 22,
    viewBox: "0 0 24 24",
    fill,
    stroke,
    strokeWidth: 1.9,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
  };

  switch (name) {
    case "upload":
      return (
        <svg {...common}>
          <path d="M12 3v12" />
          <path d="M7 8l5-5 5 5" />
          <path d="M4 21h16" />
        </svg>
      );
    case "quality_control":
      return (
        <svg {...common}>
          <path d="M10 3h4" />
          <path d="M12 3v7" />
          <path d="M8 10h8" />
          <path d="M7 10l-2 8h14l-2-8" />
        </svg>
      );
    case "normalization":
      return (
        <svg {...common}>
          <path d="M4 6h16" />
          <path d="M7 6v14" />
          <path d="M17 6v14" />
          <path d="M10 12h4" />
        </svg>
      );
    case "batch_correction":
      return (
        <svg {...common}>
          <path d="M7 7h10v10H7z" />
          <path d="M7 12h10" />
          <path d="M12 7v10" />
        </svg>
      );
    case "clustering":
      return (
        <svg {...common}>
          <circle cx="7" cy="7" r="2" />
          <circle cx="17" cy="7" r="2" />
          <circle cx="12" cy="17" r="2" />
          <path d="M9 8.5l2 6" />
          <path d="M15 8.5l-2 6" />
        </svg>
      );
    case "ml_training":
      return (
        <svg {...common}>
          <path d="M12 3l9 5-9 5-9-5 9-5z" />
          <path d="M3 8v8l9 5 9-5V8" />
        </svg>
      );
    case "results":
      return (
        <svg {...common}>
          <path d="M12 3v12" />
          <path d="M7 10l5 5 5-5" />
          <path d="M5 21h14" />
        </svg>
      );
    default:
      return null;
  }
}

function LockIcon() {
  return (
    <svg
      width={14}
      height={14}
      viewBox="0 0 24 24"
      fill="none"
      stroke="#94a3b8"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M7 11V8a5 5 0 0 1 10 0v3" />
      <path d="M6 11h12v10H6z" />
    </svg>
  );
}

function InfoIcon() {
  return (
    <svg
      width={18}
      height={18}
      viewBox="0 0 24 24"
      fill="none"
      stroke="#2563eb"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="10" />
      <path d="M12 16v-4" />
      <path d="M12 8h.01" />
    </svg>
  );
}

export default function App() {
  const [phaseIndex, setPhaseIndex] = useState<number>(0);

  const [apiBase, setApiBase] = useState<string>(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const v = (import.meta as any)?.env?.VITE_API_BASE_URL;
    return typeof v === "string" ? v : "";
  });

  const [file, setFile] = useState<File | null>(null);
  const [fileInfo, setFileInfo] = useState<{ name: string; size: number; rows?: number; cols?: number } | null>(null);

  const [uploaded, setUploaded] = useState<boolean>(false);

  const [qcStatus, setQcStatus] = useState<StepStatus>("idle");
  const [normStatus, setNormStatus] = useState<StepStatus>("idle");
  const [harmStatus, setHarmStatus] = useState<StepStatus>("idle");
  const [clusStatus, setClusStatus] = useState<StepStatus>("idle");
  const [trainStatus, setTrainStatus] = useState<StepStatus>("idle");
  const [exportStatus, setExportStatus] = useState<StepStatus>("idle");

  // Backend currently exposes POST /normalize only. We store its returned content for export.
  const [normalizedText, setNormalizedText] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);

  const [logLines, setLogLines] = useState<string[]>([]);
  const logRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const currentPhase = useMemo(() => PHASES[clamp(phaseIndex, 0, PHASES.length - 1)], [phaseIndex]);

  // Run small sanity checks once in dev.
  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const isDev = !!(import.meta as any)?.env?.DEV;
    if (!isDev || __didSelfTest) return;
    __didSelfTest = true;
    try {
      __devSelfTest();
    } catch (e) {
      // eslint-disable-next-line no-console
      console.error(e);
    }
  }, []);

  function log(line: string) {
    setLogLines((prev) => [...prev, `${new Date().toLocaleTimeString()}  ${line}`]);
  }

  useEffect(() => {
    if (!logRef.current) return;
    logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logLines]);

  function isUnlocked(idx: number) {
    const key = PHASES[idx]?.key;
    if (!key) return false;
    if (key === "upload") return true;
    if (key === "quality_control") return uploaded;
    if (key === "normalization") return qcStatus === "done";
    if (key === "batch_correction") return normStatus === "done";
    if (key === "clustering") return harmStatus === "done";
    if (key === "ml_training") return clusStatus === "done";
    if (key === "results") return trainStatus === "done";
    return false;
  }

  async function parseQuickMatrixInfo(f: File) {
    const maxBytes = 256 * 1024;
    const blob = f.slice(0, maxBytes);
    const text = await blob.text();
    const lines = splitNonEmptyLines(text);
    return estimateMatrixShape(lines);
  }

  async function onChooseFile(f: File | null) {
    setFile(f);
    setUploaded(false);

    setQcStatus("idle");
    setNormStatus("idle");
    setHarmStatus("idle");
    setClusStatus("idle");
    setTrainStatus("idle");
    setExportStatus("idle");
    setNormalizedText(null);
    setRunId(null);

    setLogLines([]);

    if (!f) {
      setFileInfo(null);
      return;
    }

    log(`Selected file: ${f.name} (${prettyBytes(f.size)})`);
    const meta = await parseQuickMatrixInfo(f).catch(() => ({ rows: undefined, cols: undefined }));
    setFileInfo({ name: f.name, size: f.size, rows: meta.rows, cols: meta.cols });
  }

  async function doUpload() {
    if (!file) return;

    const base = sanitizeApiBase(apiBase);

    // Prefer backend /upload if available (so the backend can mint a run_id and persist artifacts).
    if (base.trim()) {
      const form = new FormData();
    form.append("file", file, file.name);
    if (runId) form.append("run_id", runId);
      log(`Calling backend: POST ${buildUrl(base, "/upload")}  file=${file.name} (${prettyBytes(file.size)})`);

      const res = await runWithBackend({ apiBase: base, path: "/upload", form });

      if (res.ok) {
        setUploaded(true);
        const payload = res.payload;
        if (payload && typeof payload === "object" && "run_id" in (payload as any)) {
          const rid = String((payload as any).run_id);
          setRunId(rid);
          log(`Captured run_id: ${rid}`);
        }
        log("Upload done (backend). ");
        return;
      }

      // If the backend does not implement /upload, fall back to local upload.
      if (String(res.message).includes("API error 404")) {
        log("Upload set locally. Backend has no /upload route.");
        setUploaded(true);
        return;
      }

      log(`Upload failed: ${res.message}. Falling back to local upload.`);
      setUploaded(true);
      return;
    }

    log("Upload set locally. No API base URL set.");
    setUploaded(true);
  }

  async function doQC() {
    if (!uploaded) {
      log("QC is locked. Upload data first.");
      return;
    }
    if (!file) {
      log("No file selected.");
      return;
    }

    setQcStatus("running");
    log("Quality control started…");

    const base = sanitizeApiBase(apiBase);

    // Backend expects: POST /qc with multipart/form-data and required field name 'file'.
    // If no API base is set, we simulate QC to keep the UI usable.
    if (!base.trim()) {
      await sleep(500);
      setQcStatus("done");
      log("Quality control done (simulated). No API base URL set.");
      return;
    }

    const form = new FormData();
    form.append("file", file, file.name);
    if (runId) form.append("run_id", runId);
    log(
      `Calling backend: POST ${buildUrl(base, "/qc")}  file=${file.name} (${prettyBytes(file.size)})${runId ? `  run_id=${runId}` : ""}`
    );

    let res = await runWithBackend({ apiBase: base, path: "/qc", form });

    // Some deployments define /qc/ with a trailing slash. Avoid redirects that can drop multipart bodies.
    if (!res.ok && isMissingFile422(res.payload)) {
      log("Backend reports missing form field 'file'. Retrying with /qc/ …");
      res = await runWithBackend({ apiBase: base, path: "/qc/", form });
    }

    if (res.ok) {
      setQcStatus("done");
      const payload = res.payload ?? { message: res.message };
      if (payload && typeof payload === "object" && "run_id" in (payload as any)) {
        const rid = String((payload as any).run_id);
        setRunId(rid);
        log(`Captured run_id: ${rid}`);
      }
      const txt = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
      log("Quality control done (backend). QC summary received.");
      log(`QC summary: ${txt.length > 220 ? `${txt.slice(0, 220)}…` : txt}`);
    } else {
      setQcStatus("error");
      log(`Quality control failed: ${res.message}`);
    }
  }

  async function doNormalization() {
    if (qcStatus !== "done") {
      log("Normalization is locked. Complete QC first.");
      return;
    }
    if (!file) {
      log("No file selected.");
      return;
    }

    setNormStatus("running");
    log("Normalization started…");

    // Backend expects: POST /normalize with multipart/form-data and required field name 'file'.
    // Include run_id if available so the backend can persist artifacts for the run.
    const form = new FormData();
    form.append("file", file, file.name);
    if (runId) form.append("run_id", runId);

    const base = sanitizeApiBase(apiBase);

    if (base.trim()) {
      log(
        `Calling backend: POST ${buildUrl(base, "/normalize")}  file=${file.name} (${prettyBytes(file.size)})${runId ? `  run_id=${runId}` : ""}`
      );
    } else {
      log("No API base URL set. Running simulated normalization.");
    }

    let res = await runWithBackend({ apiBase: base, path: "/normalize", form });

    // Some deployments define /normalize/ with a trailing slash. Avoid redirects that can drop multipart bodies.
    if (!res.ok && base.trim() && isMissingFile422(res.payload)) {
      log("Backend reports missing form field 'file'. Retrying with /normalize/ …");
      res = await runWithBackend({ apiBase: base, path: "/normalize/", form });
    }

    if (res.ok) {
      setNormStatus("done");

      const payload = res.payload ?? { message: res.message };
      const textMaybe = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
      setNormalizedText(textMaybe ?? JSON.stringify({ message: res.message }, null, 2));

      log(
        res.message.includes("simulated")
          ? "Normalization done (simulated). Output stored for export."
          : "Normalization done (backend). Output stored for export."
      );
    } else {
      setNormStatus("error");
      log(`Normalization failed: ${res.message}`);
    }
  }

  async function doHarmony() {
    if (normStatus !== "done") {
      log("Batch correction is locked. Complete normalization first.");
      return;
    }
    if (!runId) {
      setHarmStatus("error");
      log("Harmony requires run_id from QC. Run QC again.");
      return;
    }

    setHarmStatus("running");
    log("Harmony batch correction started…");

    const base = sanitizeApiBase(apiBase);

    // Backend expects: POST /harmony with JSON body { run_id }.
    // If no API base is set, we simulate to keep the UI usable.
    if (!base.trim()) {
      await sleep(700);
      setHarmStatus("done");
      log("Harmony batch correction done (simulated). No API base URL set.");
      return;
    }

    log(`Calling backend: POST ${buildUrl(base, "/harmony")}  run_id=${runId}`);

    let res = await runWithBackend({ apiBase: base, path: "/harmony", json: { run_id: runId } });

    // Fallback for backends that implemented /harmony as multipart.
    if (!res.ok && (isMissingRunId422(res.payload) || isRunIdNotFound(res.payload))) {
      log("Backend reports missing run_id in JSON. Retrying as multipart…");
      const form = new FormData();
      form.append("run_id", runId);
      if (file) form.append("file", file, file.name);
      res = await runWithBackend({ apiBase: base, path: "/harmony", form });
    }

    if (res.ok) {
      setHarmStatus("done");
      const payload = res.payload ?? { message: res.message };
      const txt = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
      log("Harmony batch correction done (backend). Harmony output received.");
      log(`Harmony output: ${txt.length > 220 ? `${txt.slice(0, 220)}…` : txt}`);
    } else {
      setHarmStatus("error");
      log(`Harmony batch correction failed: ${res.message}`);
    }
  }

  async function doClustering() {
    if (harmStatus !== "done") {
      log("Clustering is locked. Complete batch correction first.");
      return;
    }
    if (!runId) {
      setClusStatus("error");
      log("Clustering requires run_id. Run QC again.");
      return;
    }

    setClusStatus("running");
    log("Clustering started…");

    const base = sanitizeApiBase(apiBase);

    if (!base.trim()) {
      await sleep(700);
      setClusStatus("done");
      log("Clustering done (simulated). No API base URL set.");
      return;
    }

    const form = new FormData();
    form.append("run_id", runId);

    log(`Calling backend: POST ${buildUrl(base, "/cluster")}  run_id=${runId}`);

    const res = await runWithBackend({ apiBase: base, path: "/cluster", json: { run_id: runId } });

    if (res.ok) {
      setClusStatus("done");
      const payload = res.payload ?? { message: res.message };
      const txt = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
      log("Clustering done (backend). Cluster labels received.");
      log(`Cluster output: ${txt.length > 220 ? `${txt.slice(0, 220)}…` : txt}`);
    } else {
      setClusStatus("error");
      log(`Clustering failed: ${res.message}`);
    }
  }

  async function doTraining() {
    if (clusStatus !== "done") {
      log("Training is locked. Complete clustering first.");
      return;
    }
    if (!runId) {
      setTrainStatus("error");
      log("Training requires run_id. Run QC again.");
      return;
    }

    setTrainStatus("running");
    log("ML training started…");

    const base = sanitizeApiBase(apiBase);

    if (!base.trim()) {
      await sleep(700);
      setTrainStatus("done");
      log("ML training done (simulated). No API base URL set.");
      return;
    }

    log(`Calling backend: POST ${buildUrl(base, "/train")}  run_id=${runId}`);

    // Prefer JSON first. Swagger indicates /train expects a JSON body { run_id }.
    let res = await runWithBackend({
      apiBase: base,
      path: "/train",
      json: { run_id: runId },
      timeoutMs: 180000,
    });

    // Some backends implement /train as multipart. Retry if JSON was rejected.
    if (!res.ok) {
      log("Retrying /train as multipart…");
      const form = new FormData();
      form.append("run_id", runId);
      res = await runWithBackend({ apiBase: base, path: "/train", form, timeoutMs: 180000 });
    }

    // Some deployments define /train/ with a trailing slash.
    if (!res.ok) {
      log("Retrying /train/ as JSON…");
      res = await runWithBackend({
        apiBase: base,
        path: "/train/",
        json: { run_id: runId },
        timeoutMs: 180000,
      });
    }

    if (!res.ok) {
      log("Retrying /train/ as multipart…");
      const form = new FormData();
      form.append("run_id", runId);
      res = await runWithBackend({ apiBase: base, path: "/train/", form, timeoutMs: 180000 });
    }

    if (res.ok) {
      setTrainStatus("done");
      const payload = res.payload ?? { message: res.message };
      const txt = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
      log("ML training done (backend). Metrics received.");
      log(`Training output: ${txt.length > 220 ? `${txt.slice(0, 220)}…` : txt}`);
    } else {
      setTrainStatus("error");
      log(`ML training failed: ${res.message}`);
    }
  }

  async function doExport() {
    if (trainStatus !== "done") {
      log("Results are locked. Complete training first.");
      return;
    }
    if (!runId) {
      setExportStatus("error");
      log("Export requires run_id. Run QC again.");
      return;
    }

    setExportStatus("running");
    log("Preparing export…");

    const base = sanitizeApiBase(apiBase);

    if (!base.trim()) {
      // Fallback: export locally as before.
      await sleep(300);

      if (normalizedText) {
        const looksLikeTable = normalizedText.includes("\n") && (normalizedText.includes(",") || normalizedText.includes("\t"));
        const mime = looksLikeTable ? "text/csv" : "application/json";
        const ext = looksLikeTable ? "csv" : "json";
        const blob = new Blob([normalizedText], { type: mime });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `normalized_output.${ext}`;
        a.click();
        URL.revokeObjectURL(a.href);
        log(`Downloaded normalized_output.${ext}.`);
      } else {
        log("No normalized output stored yet. Run Normalization first.");
      }

      setExportStatus("done");
      log("Export downloaded.");
      return;
    }

    // Backend export: download a zip bundle.
    const url = buildUrl(base, "/export");
    const form = new FormData();
    form.append("run_id", runId);

    try {
      const res = await fetch(url, {
        method: "POST",
        headers: { Accept: "application/zip" },
        body: form,
        mode: "cors",
        redirect: "follow",
      });

      if (!res.ok) {
        const data = await safeJson(res);
        setExportStatus("error");
        log(`Export failed: API error ${res.status}: ${typeof data === "string" ? data : JSON.stringify(data)}`);
        return;
      }

      const blob = await res.blob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "rnaseq_export.zip";
      a.click();
      URL.revokeObjectURL(a.href);

      setExportStatus("done");
      log("Export downloaded.");
    } catch (e: any) {
      setExportStatus("error");
      log(`Export failed: ${e?.message ?? "Network error."}`);
    }
  }

  async function testHealth() {
    if (!apiBase.trim()) {
      log("API base URL is empty.");
      return;
    }

    log("Health check started…");
    const base = sanitizeApiBase(apiBase);
    const res = await runWithBackend({ apiBase: base, path: "/health", method: "GET" });

    if (res.ok) {
      log("Health check OK.");
    } else {
      log(`Health check failed: ${res.message}`);
    }
  }

  function resetAll() {
    setPhaseIndex(0);
    setFile(null);
    setFileInfo(null);
    setUploaded(false);
    setQcStatus("idle");
    setNormStatus("idle");
    setHarmStatus("idle");
    setClusStatus("idle");
    setTrainStatus("idle");
    setExportStatus("idle");
    setNormalizedText(null);
    setRunId(null);
    setLogLines([]);
  }

  function goTo(idx: number) {
    const i = clamp(idx, 0, PHASES.length - 1);
    if (!isUnlocked(i)) {
      log(`Step locked: ${PHASES[i].title}. Complete previous steps first.`);
      return;
    }
    setPhaseIndex(i);
  }

  function next() {
    const i = clamp(phaseIndex + 1, 0, PHASES.length - 1);
    goTo(i);
  }

  function prev() {
    const i = clamp(phaseIndex - 1, 0, PHASES.length - 1);
    goTo(i);
  }

  const stepLabel = `Step ${phaseIndex + 1}: ${currentPhase.title}`;

  function StepCallout({ children }: { children: React.ReactNode }) {
    return (
      <div className="callout">
        <div className="calloutIcon">
          <InfoIcon />
        </div>
        <div className="calloutBody">{children}</div>
      </div>
    );
  }

  function BackendSettings() {
    const base = sanitizeApiBase(apiBase);
    const changed = apiBase.trim() && base !== apiBase.trim();

    return (
      <details className="settings">
        <summary>Backend API, optional</summary>
        <div className="settingsInner">
          <div className="settingsLabel">API base URL</div>
          <input
            className="settingsInput"
            value={apiBase}
            onChange={(e) => setApiBase(e.target.value)}
            placeholder="https://rnaseq-backend-xxxxx.europe-west1.run.app"
          />
          <div className="settingsHint">
            Leave empty to run in browser simulated mode. If set, Normalization calls POST /normalize with multipart/form-data field name <b>file</b>.
          </div>
          <div className="settingsActions">
            <button className="ghostBtn" type="button" onClick={testHealth} disabled={!apiBase.trim()}>
              Test /health
            </button>
            <div className="settingsMini">Resolved normalize URL: {base ? buildUrl(base, "/normalize") : "—"}
            {changed ? <div className="settingsMini">Note: stripped "/docs" from the base URL.</div> : null}</div>
          </div>
        </div>
      </details>
    );
  }

  function UploadView() {
    return (
      <div className="content">
        <StepCallout>
          <div className="calloutTitle">Upload a count matrix</div>
          <div className="calloutText">
            This demo parses CSV or TSV in the browser and computes QC summaries. Assumes first column is gene IDs,
            remaining columns are cells.
          </div>
        </StepCallout>

        <h2 className="stepTitle">{stepLabel}</h2>

        <div
          className="dropzone"
          role="button"
          tabIndex={0}
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click();
          }}
          onDragOver={(e) => {
            e.preventDefault();
            e.stopPropagation();
          }}
          onDrop={(e) => {
            e.preventDefault();
            e.stopPropagation();
            const f = e.dataTransfer.files?.[0];
            if (f) onChooseFile(f);
          }}
        >
          <div className="dropTitle">Upload CSV or TSV count matrix</div>
          <div className="dropSub">Genes as rows, cells as columns.</div>
          <div className="dropTiny">First column must contain gene IDs.</div>

          <input
            ref={fileInputRef}
            className="hiddenFile"
            type="file"
            accept=".csv,.tsv,.txt"
            onChange={(e) => onChooseFile(e.target.files?.[0] ?? null)}
          />
        </div>

        <div className="metaRow">
          <div className="metaBox">
            <div className="metaLabel">Selected file</div>
            <div className="metaValue">{fileInfo ? fileInfo.name : "None"}</div>
            <div className="metaHint">
              {fileInfo
                ? `${prettyBytes(fileInfo.size)}${fileInfo.rows ? `, ~${fileInfo.rows} genes` : ""}${
                    fileInfo.cols ? `, ~${fileInfo.cols} cells` : ""
                  }`
                : "Select a counts matrix to begin."}
            </div>
          </div>

          <div className="metaBox">
            <div className="metaLabel">Upload</div>
            <div className="metaValue">{uploaded ? "Done" : "Pending"}</div>
            <div className="metaHint">Upload enables Quality Control.</div>
          </div>
        </div>

        <div className="actionsRow">
          <button className="primaryBtn" type="button" onClick={doUpload} disabled={!file || uploaded}>
            {uploaded ? "Uploaded" : "Upload"}
          </button>
          <button className="ghostBtn" type="button" onClick={resetAll}>
            Reset
          </button>
        </div>

        <BackendSettings />
        <LogPanel />
      </div>
    );
  }

  function SimpleStepView(opts: {
    title: string;
    description: string;
    status: StepStatus;
    actionLabel: string;
    onRun: () => void;
    hint?: string;
  }) {
    const statusText =
      opts.status === "idle"
        ? "Idle"
        : opts.status === "running"
        ? "Running"
        : opts.status === "done"
        ? "Done"
        : "Error";

    const statusCls =
      opts.status === "done"
        ? "statusChip statusDone"
        : opts.status === "running"
        ? "statusChip statusRun"
        : opts.status === "error"
        ? "statusChip statusErr"
        : "statusChip";

    return (
      <div className="content">
        <StepCallout>
          <div className="calloutTitle">{opts.title}</div>
          <div className="calloutText">{opts.description}</div>
        </StepCallout>

        <h2 className="stepTitle">{stepLabel}</h2>

        <div className="panel">
          <div className="panelTop">
            <div className="panelTitle">Run step</div>
            <span className={statusCls}>{statusText}</span>
          </div>
          <div className="panelHint">{opts.hint ?? "Configure this later to match your backend."}</div>
          <div className="panelActions">
            <button className="primaryBtn" type="button" onClick={opts.onRun}>
              {opts.actionLabel}
            </button>
          </div>
        </div>

        <BackendSettings />
        <LogPanel />
      </div>
    );
  }

  function ResultsView() {
    return (
      <div className="content">
        <StepCallout>
          <div className="calloutTitle">Results and export</div>
          <div className="calloutText">Download outputs and reports. If API is empty, a demo JSON file is produced.</div>
        </StepCallout>

        <h2 className="stepTitle">{stepLabel}</h2>

        <div className="panel">
          <div className="panelTop">
            <div className="panelTitle">Export</div>
            <span
              className={
                exportStatus === "done"
                  ? "statusChip statusDone"
                  : exportStatus === "running"
                  ? "statusChip statusRun"
                  : exportStatus === "error"
                  ? "statusChip statusErr"
                  : "statusChip"
              }
            >
              {exportStatus === "idle"
                ? "Idle"
                : exportStatus === "running"
                ? "Running"
                : exportStatus === "done"
                ? "Done"
                : "Error"}
            </span>
          </div>
          <div className="panelHint">Exports are generated after training.</div>
          <div className="panelActions">
            <button className="primaryBtn" type="button" onClick={doExport}>
              Export
            </button>
          </div>
        </div>

        <BackendSettings />
        <LogPanel />
      </div>
    );
  }

  function LogPanel() {
    return (
      <details className="logWrap">
        <summary>Run log</summary>
        <div ref={logRef} className="log">
          {logLines.length === 0 ? "No events yet." : logLines.join("\n")}
        </div>
        <div className="logActions">
          <button
            className="ghostBtn"
            type="button"
            onClick={() => {
              navigator.clipboard?.writeText(logLines.join("\n")).catch(() => undefined);
            }}
            disabled={logLines.length === 0}
          >
            Copy log
          </button>
          <button className="ghostBtn" type="button" onClick={() => setLogLines([])} disabled={logLines.length === 0}>
            Clear
          </button>
        </div>
      </details>
    );
  }

  function renderBody() {
    switch (currentPhase.key) {
      case "upload":
        return <UploadView />;
      case "quality_control":
        return (
          <SimpleStepView
            title="Quality control"
            description="Basic checks, missing values, filtering."
            status={qcStatus}
            actionLabel="Run quality control"
            onRun={doQC}
            hint="Calls POST /qc on your backend using the selected file. If API base URL is empty, QC is simulated. Unlocks normalization."
          />
        );
      case "normalization":
        return (
          <SimpleStepView
            title="Normalization"
            description="Log-normalize and scale."
            status={normStatus}
            actionLabel="Run normalization"
            onRun={doNormalization}
            hint="Calls POST /normalize on your backend. Upload uses the selected file."
          />
        );
      case "batch_correction":
        return (
          <SimpleStepView
            title="Batch correction"
            description="Harmony batch correction in PCA space."
            status={harmStatus}
            actionLabel="Run batch correction"
            onRun={doHarmony}
            hint="Calls POST /harmony on your backend using run_id from QC. If API base URL is empty, Harmony is simulated. Unlocks clustering."
          />
        );
      case "clustering":
        return (
          <SimpleStepView
            title="Clustering"
            description="Reduce dimensions and cluster."
            status={clusStatus}
            actionLabel="Run clustering"
            onRun={doClustering}
            hint="Calls POST /cluster on your backend using run_id. If API base URL is empty, clustering is simulated. Unlocks ML training."
          />
        );
      case "ml_training":
        return (
          <SimpleStepView
            title="ML training"
            description="Train a simple model on derived features, then inspect."
            status={trainStatus}
            actionLabel="Run training"
            onRun={doTraining}
            hint="Calls POST /train on your backend using run_id. If API base URL is empty, training is simulated. Unlocks results."
          />
        );
      case "results":
        return <ResultsView />;
      default:
        return null;
    }
  }

  return (
    <div className="page">
      <style>{`
        :root { color-scheme: light; }

        .page {
          min-height: 100vh;
          background: #eef3ff;
          padding: 24px;
          font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
          color: #0f172a;
        }

        .frame {
          max-width: 1180px;
          margin: 0 auto;
          background: #ffffff;
          border-radius: 14px;
          box-shadow: 0 18px 50px rgba(15, 23, 42, 0.18);
          overflow: hidden;
          border: 1px solid rgba(15, 23, 42, 0.06);
        }

        .header {
          padding: 26px 28px;
          background: linear-gradient(90deg, #6d28d9 0%, #7c3aed 40%, #3b82f6 100%);
          color: #ffffff;
        }

        .headerTitle {
          font-size: 34px;
          font-weight: 800;
          letter-spacing: -0.4px;
        }

        .headerSub {
          margin-top: 6px;
          font-size: 14px;
          opacity: 0.92;
        }

        .tabs {
          display: grid;
          grid-template-columns: repeat(7, 1fr);
          gap: 0;
          background: #ffffff;
          border-bottom: 1px solid rgba(15, 23, 42, 0.08);
        }

        .tab {
          position: relative;
          padding: 14px 10px 12px;
          cursor: pointer;
          background: transparent;
          border: none;
          text-align: center;
          color: #64748b;
          transition: background 120ms ease;
        }

        .tab:hover { background: rgba(59, 130, 246, 0.04); }

        .tabActive {
          color: #1d4ed8;
        }

        .tabActive::after {
          content: "";
          position: absolute;
          left: 10px;
          right: 10px;
          bottom: 0;
          height: 3px;
          background: #1d4ed8;
          border-radius: 999px;
        }

        .tabInner {
          display: grid;
          justify-items: center;
          gap: 6px;
        }

        .tabLabel {
          font-size: 12px;
          font-weight: 650;
        }

        .tabLock {
          position: absolute;
          top: 10px;
          right: 10px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          width: 22px;
          height: 22px;
          border-radius: 999px;
          background: rgba(148, 163, 184, 0.18);
        }

        .tabLocked {
          cursor: not-allowed;
          color: #94a3b8;
        }

        .body {
          padding: 26px 28px;
        }

        .content { max-width: 980px; }

        .callout {
          display: grid;
          grid-template-columns: 22px 1fr;
          gap: 12px;
          padding: 14px 16px;
          border-radius: 12px;
          background: #eff6ff;
          border: 1px solid #cfe2ff;
          color: #0f2a5f;
        }

        .calloutTitle {
          font-weight: 750;
          font-size: 14px;
        }

        .calloutText {
          margin-top: 2px;
          font-size: 13px;
          line-height: 1.35;
          color: #1e3a8a;
        }

        .stepTitle {
          margin-top: 22px;
          margin-bottom: 16px;
          font-size: 28px;
          font-weight: 800;
          letter-spacing: -0.3px;
          color: #0b1220;
        }

        .dropzone {
          border: 2px dashed rgba(100, 116, 139, 0.35);
          border-radius: 12px;
          padding: 44px 18px;
          background: #ffffff;
          text-align: center;
          cursor: pointer;
          user-select: none;
        }

        .dropzone:hover {
          background: #f8fbff;
          border-color: rgba(37, 99, 235, 0.35);
        }

        .dropTitle {
          font-size: 15px;
          font-weight: 750;
          color: #1d4ed8;
        }

        .dropSub {
          margin-top: 8px;
          font-size: 13px;
          color: #475569;
        }

        .dropTiny {
          margin-top: 6px;
          font-size: 12px;
          color: #94a3b8;
        }

        .hiddenFile { display: none; }

        .metaRow {
          margin-top: 18px;
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 12px;
        }

        .metaBox {
          border: 1px solid rgba(15, 23, 42, 0.08);
          border-radius: 12px;
          background: #ffffff;
          padding: 12px 14px;
          box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06);
        }

        .metaLabel {
          font-size: 12px;
          color: #64748b;
        }

        .metaValue {
          margin-top: 6px;
          font-size: 14px;
          font-weight: 750;
          color: #0f172a;
        }

        .metaHint {
          margin-top: 4px;
          font-size: 12px;
          color: #64748b;
          line-height: 1.3;
        }

        .actionsRow {
          margin-top: 16px;
          display: flex;
          gap: 10px;
          align-items: center;
        }

        .primaryBtn {
          border: none;
          background: #3b82f6;
          color: #ffffff;
          font-weight: 750;
          padding: 10px 16px;
          border-radius: 10px;
          cursor: pointer;
          box-shadow: 0 10px 20px rgba(59, 130, 246, 0.22);
          transition: transform 100ms ease, filter 120ms ease;
        }

        .primaryBtn:hover { filter: brightness(0.98); transform: translateY(-1px); }
        .primaryBtn:disabled { opacity: 0.55; cursor: not-allowed; box-shadow: none; transform: none; }

        .ghostBtn {
          border: 1px solid rgba(15, 23, 42, 0.12);
          background: #ffffff;
          color: #0f172a;
          font-weight: 700;
          padding: 10px 14px;
          border-radius: 10px;
          cursor: pointer;
        }

        .ghostBtn:disabled { opacity: 0.55; cursor: not-allowed; }

        .panel {
          border: 1px solid rgba(15, 23, 42, 0.08);
          border-radius: 12px;
          background: #ffffff;
          padding: 14px 14px;
          box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
        }

        .panelTop {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 10px;
        }

        .panelTitle { font-weight: 800; }

        .panelHint {
          margin-top: 8px;
          color: #64748b;
          font-size: 13px;
          line-height: 1.35;
        }

        .panelActions { margin-top: 12px; }

        .statusChip {
          font-size: 12px;
          font-weight: 750;
          padding: 6px 10px;
          border-radius: 999px;
          background: rgba(100, 116, 139, 0.10);
          color: #475569;
          border: 1px solid rgba(100, 116, 139, 0.20);
        }

        .statusRun { background: rgba(59, 130, 246, 0.10); color: #1d4ed8; border-color: rgba(59, 130, 246, 0.22); }
        .statusDone { background: rgba(16, 185, 129, 0.10); color: #047857; border-color: rgba(16, 185, 129, 0.20); }
        .statusErr { background: rgba(239, 68, 68, 0.10); color: #b91c1c; border-color: rgba(239, 68, 68, 0.20); }

        .settings {
          margin-top: 16px;
          border: 1px solid rgba(15, 23, 42, 0.08);
          border-radius: 12px;
          background: #fbfdff;
          padding: 10px 12px;
        }

        .settings > summary {
          cursor: pointer;
          font-weight: 750;
          color: #334155;
          list-style: none;
        }

        .settingsInner { margin-top: 10px; }

        .settingsLabel { font-size: 12px; color: #64748b; }

        .settingsInput {
          margin-top: 6px;
          width: 100%;
          border: 1px solid rgba(15, 23, 42, 0.12);
          border-radius: 10px;
          padding: 10px 10px;
          outline: none;
        }

        .settingsInput:focus {
          border-color: rgba(59, 130, 246, 0.45);
          box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.14);
        }

        .settingsHint {
          margin-top: 8px;
          font-size: 12px;
          color: #64748b;
          line-height: 1.35;
        }

        .settingsActions {
          margin-top: 10px;
          display: flex;
          align-items: center;
          gap: 12px;
          flex-wrap: wrap;
        }

        .settingsMini {
          font-size: 12px;
          color: #64748b;
        }

        .logWrap {
          margin-top: 14px;
          border: 1px solid rgba(15, 23, 42, 0.08);
          border-radius: 12px;
          background: #ffffff;
          padding: 10px 12px;
        }

        .logWrap > summary {
          cursor: pointer;
          font-weight: 750;
          color: #334155;
          list-style: none;
        }

        .log {
          margin-top: 10px;
          height: 160px;
          overflow: auto;
          padding: 10px;
          border-radius: 10px;
          border: 1px solid rgba(15, 23, 42, 0.08);
          background: #0b1220;
          color: rgba(255,255,255,0.86);
          font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
          font-size: 12px;
          white-space: pre-wrap;
          line-height: 1.35;
        }

        .logActions { display: flex; gap: 10px; margin-top: 10px; }

        .footer {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 16px 18px;
          border-top: 1px solid rgba(15, 23, 42, 0.08);
          background: #fbfdff;
        }

        .footerMid { font-size: 12px; color: #64748b; }

        .prevBtn {
          border: 1px solid rgba(15, 23, 42, 0.12);
          background: #ffffff;
          color: #64748b;
          font-weight: 750;
          padding: 10px 18px;
          border-radius: 10px;
          cursor: pointer;
        }

        .prevBtn:disabled { opacity: 0.5; cursor: not-allowed; }

        .nextBtn {
          border: none;
          background: #93c5fd;
          color: #ffffff;
          font-weight: 800;
          padding: 10px 18px;
          border-radius: 10px;
          cursor: pointer;
          box-shadow: 0 10px 20px rgba(59, 130, 246, 0.16);
        }

        .nextBtn:hover { filter: brightness(0.98); transform: translateY(-1px); }

        @media (max-width: 980px) {
          .tabs { grid-template-columns: repeat(3, 1fr); }
          .metaRow { grid-template-columns: 1fr; }
        }
      `}</style>

      <div className="frame">
        <div className="header">
          <div className="headerTitle">Single-cell RNA-seq ML pipeline</div>
          <div className="headerSub">Upload, QC, normalize, Harmony in PC space, cluster, train, inspect. v{APP_VERSION}</div>
        </div>

        <div className="tabs">
          {PHASES.map((p, idx) => {
            const locked = !isUnlocked(idx);
            const active = idx === phaseIndex;
            return (
              <button
                key={p.key}
                className={`tab ${active ? "tabActive" : ""} ${locked ? "tabLocked" : ""}`}
                type="button"
                onClick={() => {
                  if (locked) {
                    log(`Step locked: ${p.title}. Complete previous steps first.`);
                    return;
                  }
                  setPhaseIndex(idx);
                }}
                aria-disabled={locked}
              >
                {locked ? (
                  <span className="tabLock" aria-hidden="true">
                    <LockIcon />
                  </span>
                ) : null}
                <div className="tabInner">
                  <Icon name={p.key} active={active} />
                  <div className="tabLabel">{p.title}</div>
                </div>
              </button>
            );
          })}
        </div>

        <div className="body">{renderBody()}</div>

        <div className="footer">
          <button className="prevBtn" type="button" onClick={prev} disabled={phaseIndex === 0}>
            Previous
          </button>
          <div className="footerMid">
            Step {phaseIndex + 1} of {PHASES.length}
          </div>
          <button className="nextBtn" type="button" onClick={next}>
            Next
          </button>
        </div>
      </div>
    </div>
  );
}
