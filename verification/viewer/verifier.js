"use strict";

const STORED_SIGNATURE_FIELDS = new Set([
  "signature_alg",
  "signature",
  "public_key",
  "canonical_payload_sha256",
  "receipt_version",
  "key_provenance",
  "attestation",
  "canonicalization",
]);

const RUNTIME_FIELDS = new Set([
  "signature_verified",
  "verified_at_utc",
  "verifier_version",
  "verification_error",
  "public_claim_eligible",
]);

function canonicalize(value) {
  if (value === null || typeof value === "string" || typeof value === "boolean") {
    return JSON.stringify(value);
  }
  if (Number.isInteger(value)) {
    return String(value);
  }
  if (typeof value === "number") {
    throw new Error("floats are not supported in signed receipt canonicalization");
  }
  if (Array.isArray(value)) {
    return `[${value.map(canonicalize).join(",")}]`;
  }
  if (typeof value === "object") {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalize(value[key])}`)
      .join(",")}}`;
  }
  throw new Error(`unsupported JSON type: ${typeof value}`);
}

async function sha256Hex(text) {
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function b64ToBytes(value) {
  const raw = atob(value);
  return Uint8Array.from(raw, (char) => char.charCodeAt(0));
}

function signableReceipt(receipt) {
  const payload = {};
  for (const [key, value] of Object.entries(receipt)) {
    if (!STORED_SIGNATURE_FIELDS.has(key) && !RUNTIME_FIELDS.has(key)) {
      payload[key] = value;
    }
  }
  return payload;
}

function walk(value, output = []) {
  output.push(value);
  if (Array.isArray(value)) {
    for (const item of value) walk(item, output);
  } else if (value && typeof value === "object") {
    for (const item of Object.values(value)) walk(item, output);
  }
  return output;
}

function collectReceipts(payload) {
  return walk(payload).filter((value) =>
    value &&
    typeof value === "object" &&
    ("signature_alg" in value || "receipt_version" in value || "canonical_payload_sha256" in value),
  );
}

async function verifyReceipt(receipt) {
  if (receipt.receipt_version === "unsigned_legacy" || !receipt.signature) {
    return { signature_verified: false, verification_error: "unsigned_legacy" };
  }
  const payload = signableReceipt(receipt);
  const canonical = canonicalize(payload);
  const digest = await sha256Hex(canonical);
  if (receipt.canonical_payload_sha256 !== digest) {
    return { signature_verified: false, verification_error: "canonical_payload_sha256 mismatch" };
  }
  try {
    const key = await crypto.subtle.importKey("raw", b64ToBytes(receipt.public_key), { name: "Ed25519" }, false, ["verify"]);
    const ok = await crypto.subtle.verify({ name: "Ed25519" }, key, b64ToBytes(receipt.signature), new TextEncoder().encode(canonical));
    return { signature_verified: ok, verification_error: ok ? null : "invalid signature" };
  } catch (error) {
    return {
      signature_verified: null,
      verification_error: `browser_ed25519_unavailable: ${error.message}`,
    };
  }
}

async function verifyArtifact(file) {
  const text = await file.text();
  const payload = JSON.parse(text);
  const receipts = collectReceipts(payload);
  const receiptResults = [];
  for (const receipt of receipts) {
    receiptResults.push(await verifyReceipt(receipt));
  }
  const boundaries = walk(payload)
    .filter((value) => typeof value === "string" && /claim_boundary|not |does not/i.test(value))
    .slice(0, 20);
  const verified = receiptResults.filter((item) => item.signature_verified === true).length;
  const unsupported = receiptResults.filter((item) => item.signature_verified === null).length;
  const failed = receiptResults.filter((item) => item.signature_verified === false && item.verification_error !== "unsigned_legacy").length;
  const unsigned = receiptResults.filter((item) => item.verification_error === "unsigned_legacy").length;
  return {
    artifact: "helix-browser-verifier-report",
    browser_verifier_version: "helix-browser-verifier-v0",
    file_name: file.name,
    artifact_file_sha256: await sha256Hex(text),
    receipt_count: receipts.length,
    signature_verified_count: verified,
    signature_unsupported_count: unsupported,
    signature_failed_count: failed,
    unsigned_legacy_count: unsigned,
    claim_boundaries: boundaries,
    status: failed > 0 ? "failed" : "verified_with_browser_capabilities",
    claim_boundary: "Browser verification checks artifact integrity and signed receipts; it does not prove model behavior.",
  };
}

const input = document.getElementById("artifact-file");
const statusEl = document.getElementById("status");
const reportEl = document.getElementById("report");

input.addEventListener("change", async () => {
  const [file] = input.files || [];
  if (!file) return;
  statusEl.textContent = "Verifying...";
  statusEl.className = "status warn";
  try {
    const report = await verifyArtifact(file);
    reportEl.textContent = JSON.stringify(report, null, 2);
    statusEl.textContent = report.status;
    statusEl.className = `status ${report.signature_failed_count ? "bad" : "good"}`;
  } catch (error) {
    statusEl.textContent = "failed";
    statusEl.className = "status bad";
    reportEl.textContent = JSON.stringify({ error: error.message }, null, 2);
  }
});
