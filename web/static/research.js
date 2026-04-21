(function () {
  const state = {
    locale: window.localStorage.getItem("helix-research-locale") || "es",
    dataSource: "backend",
    manifest: [],
    artifacts: {},
    datasets: null,
    reports: [],
    activeReportId: null,
    selectedTag: "all",
    selectedArtifactName: null,
    search: "",
  };

  const byId = (id) => document.getElementById(id);
  const isEs = () => state.locale === "es";
  const s = (es, en) => (isEs() ? es : en);

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatNumber(value, digits = 2) {
    if (value == null || Number.isNaN(Number(value))) return "--";
    return new Intl.NumberFormat(isEs() ? "es-AR" : "en-US", {
      maximumFractionDigits: digits,
    }).format(Number(value));
  }

  function formatRatio(value, digits = 2) {
    if (value == null || Number.isNaN(Number(value))) return "--";
    return `${formatNumber(value, digits)}x`;
  }

  function fetchJson(url) {
    return fetch(url).then((response) => {
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return response.json();
    });
  }

  function preferredSources() {
    const backend = {
      mode: "backend",
      manifest: "/research/artifacts",
      artifact: (name) => `/research/artifacts/${encodeURIComponent(name)}`,
    };
    const statik = {
      mode: "static",
      manifest: "/research-data/manifest.json",
      artifact: (name) => `/research-data/artifacts/${encodeURIComponent(name)}`,
    };
    return state.dataSource === "static" ? [statik, backend] : [backend, statik];
  }

  async function fetchManifestWithFallback() {
    let lastError = null;
    for (const source of preferredSources()) {
      try {
        const payload = await fetchJson(source.manifest);
        state.dataSource = source.mode;
        return payload.artifacts || [];
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError || new Error("manifest unavailable");
  }

  async function fetchArtifactWithFallback(name) {
    let lastError = null;
    for (const source of preferredSources()) {
      try {
        const payload = await fetchJson(source.artifact(name));
        state.dataSource = source.mode;
        return payload.payload || payload;
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError || new Error(`artifact unavailable: ${name}`);
  }

  function svgShell(viewBox, body) {
    return `<svg viewBox="${viewBox}" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">${body}</svg>`;
  }

  function compactModelLabel(modelRef) {
    return String(modelRef || "--")
      .replace("Qwen/", "")
      .replace("HuggingFaceTB/", "")
      .replace("Zyphra/", "")
      .replace("EchoLabs33/", "");
  }

  function getData() {
    const frontierSummary = state.datasets.frontierSummary || {};
    const promptSuite = state.datasets.promptSuite || {};
    const hxqDiagnostics = state.datasets.hxqDiagnostics || {};
    const gemmaAttempts = state.datasets.gemmaAttempts || {};
    const stressDashboard = state.datasets.stressDashboard || {};
    const claimsMatrix = state.datasets.claimsMatrix || {};
    const prefixReuse = state.datasets.prefixReuse || {};
    const hybridPrefix = state.datasets.hybridPrefix || {};
    const sessionBranching = state.datasets.sessionBranching || {};
    const agentFramework = state.datasets.agentFramework || {};
    const openaiSmoke = state.datasets.openaiSmoke || {};
    const memoryCatalog = state.datasets.memoryCatalog || {};
    const layerSlice = state.datasets.layerSlice || {};
    const airllmBridge = state.datasets.airllmBridge || {};
    const memoryOpenai = state.datasets.memoryOpenai || {};
    const inferenceOs = state.datasets.inferenceOs || {};
    const stackCatalog = state.datasets.stackCatalog || {};
    const blueprintMeta = state.datasets.blueprintMeta || {};
    const transformerModels = frontierSummary?.transformer_gpu?.models || [];
    const bestTransformer = transformerModels.reduce((best, item) => {
      if ((item.best_compression_kv_ratio_vs_native || 0) > (best.best_compression_kv_ratio_vs_native || 0)) return item;
      return best;
    }, transformerModels[0] || {});
    const hybridLocal = frontierSummary?.hybrid_local || {};
    const promptAggregates = hybridLocal?.prompt_category_aggregates?.vanilla || promptSuite?.aggregates?.vanilla || {};

    return {
      frontierSummary,
      promptSuite,
      hxqDiagnostics,
      gemmaAttempts,
      transformerModels,
      bestTransformer,
      hybridLocal,
      promptAggregates,
      stressDashboard,
      claimsMatrix,
      prefixReuse,
      hybridPrefix,
      sessionBranching,
      agentFramework,
      openaiSmoke,
      memoryCatalog,
      layerSlice,
      airllmBridge,
      memoryOpenai,
      inferenceOs,
      stackCatalog,
      blueprintMeta,
    };
  }

  function heroVisual(data) {
    const prefixModel = (data.prefixReuse?.models || [])[0] || {};
    return svgShell(
      "0 0 760 540",
      `
        <line x1="80" y1="488" x2="392" y2="160" stroke="rgba(28,28,26,0.18)" stroke-width="1.3" />
        <line x1="688" y1="488" x2="392" y2="160" stroke="rgba(28,28,26,0.18)" stroke-width="1.3" />
        <line x1="392" y1="160" x2="392" y2="40" stroke="rgba(28,28,26,0.14)" stroke-width="1.1" />
        <circle cx="392" cy="160" r="110" stroke="rgba(28,28,26,0.13)" stroke-width="1.2" />
        <circle cx="392" cy="160" r="18" fill="#D64933" />
        <line x1="42" y1="342" x2="724" y2="342" stroke="rgba(28,28,26,0.12)" stroke-width="1" stroke-dasharray="8 8" />
        <line x1="42" y1="430" x2="724" y2="430" stroke="rgba(28,28,26,0.1)" stroke-width="1" stroke-dasharray="8 8" />
        <text x="56" y="94" fill="rgba(28,28,26,0.6)" font-size="12" font-family="Inter, sans-serif" letter-spacing="2">INFERENCE SESSION OS</text>
        <text x="56" y="156" fill="rgba(28,28,26,0.18)" font-size="92" font-family="Cormorant Garamond, serif">${escapeHtml(formatRatio(data.hybridPrefix?.best_speedup))}</text>
        <text x="58" y="184" fill="rgba(28,28,26,0.66)" font-size="13" font-family="Inter, sans-serif">hybrid exact-checkpoint TTFT speedup</text>
        <rect x="438" y="286" width="244" height="88" rx="22" fill="rgba(255,255,255,0.48)" stroke="rgba(28,28,26,0.16)" />
        <text x="462" y="318" fill="rgba(28,28,26,0.56)" font-size="11" font-family="Inter, sans-serif" letter-spacing="2">TRANSFORMER PREFIX</text>
        <text x="462" y="360" fill="#1C1C1A" font-size="46" font-family="Cormorant Garamond, serif">${escapeHtml(formatRatio(prefixModel.claim_speedup_including_restore))}</text>
        <text x="462" y="386" fill="rgba(28,28,26,0.72)" font-size="13" font-family="Inter, sans-serif">top-1 stable compressed path</text>
        <rect x="90" y="386" width="228" height="96" rx="22" fill="rgba(255,255,255,0.44)" stroke="rgba(28,28,26,0.16)" />
        <text x="114" y="420" fill="rgba(28,28,26,0.56)" font-size="11" font-family="Inter, sans-serif" letter-spacing="2">SESSION BRANCHING</text>
        <text x="114" y="454" fill="#1C1C1A" font-size="36" font-family="Cormorant Garamond, serif">${escapeHtml(formatNumber(data.sessionBranching?.branch_count, 0))}</text>
        <text x="168" y="454" fill="rgba(28,28,26,0.72)" font-size="13" font-family="Inter, sans-serif">verified branches</text>
        <rect x="492" y="112" width="154" height="56" rx="18" fill="rgba(255,255,255,0.38)" stroke="rgba(28,28,26,0.12)" />
        <text x="512" y="136" fill="rgba(28,28,26,0.5)" font-size="11" font-family="Inter, sans-serif" letter-spacing="2">ARTIFACT LEDGER</text>
        <text x="512" y="158" fill="#1C1C1A" font-size="28" font-family="Cormorant Garamond, serif">${escapeHtml(formatNumber(state.manifest.length, 0))}</text>
      `
    );
  }

  function reportCover(note, stat, subtitle) {
    return svgShell(
      "0 0 900 300",
      `
        <rect x="34" y="46" width="832" height="210" rx="30" stroke="rgba(235,230,223,0.24)" />
        <text x="68" y="110" fill="rgba(235,230,223,0.72)" font-size="14" font-family="Inter, sans-serif" letter-spacing="2">${escapeHtml(note)}</text>
        <text x="68" y="176" fill="#EBE6DF" font-size="72" font-family="Cormorant Garamond, serif">${escapeHtml(stat)}</text>
        <text x="68" y="220" fill="rgba(235,230,223,0.84)" font-size="16" font-family="Inter, sans-serif">${escapeHtml(subtitle)}</text>
      `
    );
  }

  function reportListFigure(kind, data) {
    if (kind === "transformer") {
      const rows = data.transformerModels
        .map((item, index) => {
          const ratio = item.best_compression_kv_ratio_vs_native;
          const width = 320 * ((ratio || 0) / 3.1);
          const y = 54 + index * 70;
          return `
            <text x="34" y="${y}" fill="#1C1C1A" font-size="15" font-family="Inter, sans-serif">${escapeHtml(compactModelLabel(item.model_ref))}</text>
            <rect x="34" y="${y + 14}" width="388" height="20" rx="10" fill="rgba(28,28,26,0.08)" stroke="rgba(28,28,26,0.16)" />
            <rect x="34" y="${y + 14}" width="${Math.max(28, width)}" height="20" rx="10" fill="${index === 1 ? "#1C1C1A" : "#D64933"}" />
            <text x="438" y="${y + 30}" fill="#1C1C1A" font-size="20" font-family="Cormorant Garamond, serif">${escapeHtml(formatRatio(ratio))}</text>
          `;
        })
        .join("");
      return svgShell("0 0 520 270", `<rect x="12" y="12" width="496" height="246" rx="24" fill="rgba(255,255,255,0.44)" stroke="rgba(28,28,26,0.18)" />${rows}`);
    }

    if (kind === "hybrid") {
      const kv = data.hybridLocal?.kv_only_gain?.hybrid_total_runtime_cache_ratio_vs_native;
      const stateOnly = data.hybridLocal?.mamba_state_only_gain?.hybrid_total_runtime_cache_ratio_vs_native;
      const combined = data.hybridLocal?.combined_hybrid_gain?.hybrid_total_runtime_cache_ratio_vs_native;
      return svgShell(
        "0 0 520 280",
        `
          <rect x="12" y="12" width="496" height="256" rx="24" fill="rgba(255,255,255,0.44)" stroke="rgba(28,28,26,0.18)" />
          <text x="34" y="54" fill="#1C1C1A" font-size="13" font-family="Inter, sans-serif">KV-only ${escapeHtml(formatRatio(kv))}</text>
          <rect x="34" y="64" width="360" height="18" rx="9" fill="rgba(28,28,26,0.08)" /><rect x="34" y="64" width="${Math.max(28, 260 * ((kv || 0) / 4.2))}" height="18" rx="9" fill="#F1D8D3" />
          <text x="34" y="134" fill="#1C1C1A" font-size="13" font-family="Inter, sans-serif">State-only ${escapeHtml(formatRatio(stateOnly))}</text>
          <rect x="34" y="144" width="360" height="18" rx="9" fill="rgba(28,28,26,0.08)" /><rect x="34" y="144" width="${Math.max(28, 260 * ((stateOnly || 0) / 4.2))}" height="18" rx="9" fill="#F4E5B5" />
          <text x="34" y="214" fill="#1C1C1A" font-size="13" font-family="Inter, sans-serif">Combined ${escapeHtml(formatRatio(combined))}</text>
          <rect x="34" y="224" width="360" height="18" rx="9" fill="rgba(28,28,26,0.08)" /><rect x="34" y="224" width="${Math.max(28, 260 * ((combined || 0) / 4.2))}" height="18" rx="9" fill="#D64933" />
        `
      );
    }

    if (kind === "session-os") {
      const prefixModel = (data.prefixReuse?.models || [])[0] || {};
      return svgShell(
        "0 0 520 300",
        `
          <rect x="12" y="12" width="496" height="276" rx="24" fill="rgba(255,255,255,0.44)" stroke="rgba(28,28,26,0.18)" />
          <text x="34" y="48" fill="#1C1C1A" font-size="12" font-family="Inter, sans-serif" letter-spacing="2">SESSION LIFECYCLE</text>
          <rect x="34" y="74" width="134" height="74" rx="18" fill="#1C1C1A" />
          <text x="52" y="102" fill="rgba(235,230,223,0.66)" font-size="10" font-family="Inter, sans-serif">prefix</text>
          <text x="52" y="130" fill="#EBE6DF" font-size="26" font-family="Cormorant Garamond, serif">${escapeHtml(formatRatio(prefixModel.claim_speedup_including_restore))}</text>
          <rect x="194" y="74" width="134" height="74" rx="18" fill="#D64933" />
          <text x="212" y="102" fill="rgba(235,230,223,0.76)" font-size="10" font-family="Inter, sans-serif">hybrid</text>
          <text x="212" y="130" fill="#EBE6DF" font-size="26" font-family="Cormorant Garamond, serif">${escapeHtml(formatRatio(data.hybridPrefix?.best_speedup))}</text>
          <rect x="354" y="74" width="120" height="74" rx="18" fill="#FFFFFF" stroke="#1C1C1A" />
          <text x="372" y="102" fill="rgba(28,28,26,0.62)" font-size="10" font-family="Inter, sans-serif">branches</text>
          <text x="372" y="130" fill="#1C1C1A" font-size="26" font-family="Cormorant Garamond, serif">${escapeHtml(formatNumber(data.sessionBranching?.branch_count, 0))}</text>
          <line x1="76" y1="204" x2="442" y2="204" stroke="#1C1C1A" stroke-dasharray="7 6" />
          <circle cx="76" cy="204" r="14" fill="#1C1C1A" /><circle cx="260" cy="204" r="14" fill="#D64933" /><circle cx="442" cy="204" r="14" fill="#1C1C1A" />
          <text x="34" y="252" fill="#1C1C1A" font-size="13" font-family="Inter, sans-serif">catalog -> restore -> audit -> OpenAI-compatible clients</text>
        `
      );
    }

    return svgShell(
      "0 0 520 260",
      `
        <rect x="12" y="12" width="496" height="236" rx="24" fill="rgba(255,255,255,0.44)" stroke="rgba(28,28,26,0.18)" />
        <text x="34" y="58" fill="#1C1C1A" font-size="14" font-family="Inter, sans-serif">code ${escapeHtml(formatRatio(data.promptAggregates?.code?.avg_speedup_vs_native))}</text>
        <text x="34" y="102" fill="#1C1C1A" font-size="14" font-family="Inter, sans-serif">daily ${escapeHtml(formatRatio(data.promptAggregates?.daily?.avg_speedup_vs_native))}</text>
        <text x="34" y="164" fill="#1C1C1A" font-size="14" font-family="Inter, sans-serif">HXQ finite ${escapeHtml(String(data.hxqDiagnostics?.logits_finite === true))}</text>
        <text x="34" y="208" fill="#1C1C1A" font-size="14" font-family="Inter, sans-serif">Gemma attempts ${escapeHtml(formatNumber((data.gemmaAttempts?.models || []).length, 0))}</text>
      `
    );
  }

  function buildReports(data) {
    const stressMissions = data.stressDashboard?.missions || [];
    const restoreMission = stressMissions.find((item) => item.mission_id === "restore-equivalence") || {};
    const stateMission = stressMissions.find((item) => item.mission_id === "state-juggler") || {};
    const contextMission = stressMissions.find((item) => item.mission_id === "context-switcher") || {};
    const longMission = stressMissions.find((item) => item.mission_id === "long-context-coder") || {};
    const prefixModel = (data.prefixReuse?.models || [])[0] || {};
    const hybridPrefix = data.hybridPrefix || {};
    const branching = data.sessionBranching || {};
    const framework = data.agentFramework || {};
    const openaiSmoke = data.openaiSmoke || {};
    const memoryCatalog = data.memoryCatalog || {};
    const layerSlice = data.layerSlice || {};
    const airllmBridge = data.airllmBridge || {};
    const memoryOpenai = data.memoryOpenai || {};
    const blueprintMeta = data.blueprintMeta || {};
    const stackCatalog = data.stackCatalog || {};
    const claims = data.claimsMatrix?.claims || [];
    const claimCounts = claims.reduce(
      (acc, claim) => {
        acc[claim.status] = (acc[claim.status] || 0) + 1;
        return acc;
      },
      { verified: 0, promising: 0, blocked: 0 }
    );
    const verifiedClaims = claims.filter((claim) => claim.status === "verified");
    const blockedClaims = claims.filter((claim) => claim.status === "blocked");
    const verifiedClaimList = verifiedClaims
      .slice(0, 5)
      .map((claim) => `<li><strong>${escapeHtml(claim.id)}</strong> - ${escapeHtml(claim.public_wording)}</li>`)
      .join("");
    const blockedClaimList = blockedClaims
      .slice(0, 4)
      .map((claim) => `<li><strong>${escapeHtml(claim.id)}</strong> - ${escapeHtml(claim.caveat)}</li>`)
      .join("");
    return [
      {
        id: "transformer",
        number: "01",
        category: s("Verified GPU", "Verified GPU"),
        title: "TRANSFORMER BASELINE",
        teaser: s(
          `La primera historia sigue viva: hasta ${formatRatio(data.bestTransformer?.best_compression_kv_ratio_vs_native)} de compresion de KV con match=true en la suite publica.`,
          `The original story still holds: up to ${formatRatio(data.bestTransformer?.best_compression_kv_ratio_vs_native)} of KV compression with match=true across the public suite.`
        ),
        action: s("Read report", "Read report"),
        summary: s(
          "La capa Transformer sigue siendo la base publica mas limpia del proyecto.",
          "The Transformer layer remains the cleanest public foundation of the project."
        ),
        tags: ["GPU", "match=true"],
        coverArt: reportCover("verified GPU suite", formatRatio(data.bestTransformer?.best_compression_kv_ratio_vs_native), "Transformer baseline"),
        content:
          `<div class="modal-visual">${reportListFigure("transformer", data)}</div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("Why it matters", "Why it matters"))}</h4><p>${escapeHtml(
            s(
              "Qwen y SmolLM sostienen la parte mas estable del proyecto: compresion real de KV, fidelidad fuerte y JSONs reproducibles.",
              "Qwen and SmolLM sustain the most stable part of the project: real KV compression, strong fidelity, and reproducible JSONs."
            )
          )}</p></div>`,
      },
      {
        id: "hybrid",
        number: "02",
        category: s("Hybrid frontier", "Hybrid frontier"),
        title: "HYBRID MEMORY FRONTIER",
        teaser: s(
          `El claim que cambia todo: combined llega a ${formatRatio(data.hybridLocal?.combined_hybrid_gain?.hybrid_total_runtime_cache_ratio_vs_native)} con ${formatRatio(data.hybridLocal?.combined_hybrid_gain?.speedup_vs_native)} de speedup.`,
          `The claim that changes everything: combined reaches ${formatRatio(data.hybridLocal?.combined_hybrid_gain?.hybrid_total_runtime_cache_ratio_vs_native)} with ${formatRatio(data.hybridLocal?.combined_hybrid_gain?.speedup_vs_native)} speedup.`
        ),
        action: s("Open frontier", "Open frontier"),
        summary: s(
          "La historia ya no es solo KV: el cuello hibrido vive sobre todo en recurrent state.",
          "The story is no longer only about KV: the hybrid bottleneck mostly lives inside recurrent state."
        ),
        tags: ["Zamba2", "hybrid"],
        coverArt: reportCover("local Zamba2 frontier", formatRatio(data.hybridLocal?.combined_hybrid_gain?.hybrid_total_runtime_cache_ratio_vs_native), "Hybrid frontier"),
        content:
          `<div class="modal-visual">${reportListFigure("hybrid", data)}</div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("Interpretation", "Interpretation"))}</h4><p>${escapeHtml(
            s(
              "KV-only casi no mueve el total, state-only si, y combined es el argumento mas fuerte de toda la microsite.",
              "KV-only barely moves the total, state-only does, and combined is the strongest argument across the microsite."
            )
          )}</p></div>`,
      },
      {
        id: "session-os",
        number: "03",
        category: s("Session OS", "Session OS"),
        title: "INFERENCE SESSION OS",
        teaser: s(
          `La capa nueva conecta catalogo, prefix reuse, checkpoint hibrido, branching y endpoint OpenAI-compatible en una historia de lifecycle.`,
          `The new layer connects catalog, prefix reuse, hybrid checkpointing, branching, and an OpenAI-compatible endpoint into one lifecycle story.`
        ),
        action: s("Open Session OS", "Open Session OS"),
        summary: s(
          "Aca Helix deja de verse como un benchmark aislado: empieza a administrar estados por agente/modelo.",
          "Here Helix stops looking like an isolated benchmark: it starts managing state per agent/model."
        ),
        tags: ["catalog", "prefix", "OpenAI", "branching"],
        coverArt: reportCover("session lifecycle", "OS v1", "Catalog + prefix + clients"),
        content:
          `<div class="modal-inline-stats"><div><span>Transformer prefix</span><strong>${escapeHtml(formatRatio(prefixModel.claim_speedup_including_restore))}</strong></div><div><span>Hybrid checkpoint</span><strong>${escapeHtml(formatRatio(hybridPrefix.best_speedup))}</strong></div><div><span>Branches</span><strong>${escapeHtml(formatNumber(branching.branch_count, 0))}</strong></div></div>` +
          `<div class="modal-inline-stats"><div><span>Recall</span><strong>${escapeHtml(memoryCatalog.status === "completed" ? "SQLite" : "--")}</strong></div><div><span>Layer slice</span><strong>${escapeHtml(layerSlice.read_mode || "--")}</strong></div><div><span>AirLLM seam</span><strong>${escapeHtml(airllmBridge.all_layer_injections_hit === true ? "hit" : "--")}</strong></div></div>` +
          `<div class="modal-visual">${reportListFigure("session-os", data)}</div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("Artifact-backed pieces", "Artifact-backed pieces"))}</h4><ul class="modal-list"><li>local-prefix-reuse-ttft-summary.json: ${escapeHtml(String(prefixModel.claim_top1_match_all === true))} top-1 stable</li><li>local-hybrid-prefix-checkpoint-summary.json: ${escapeHtml(formatRatio(hybridPrefix.best_speedup))} exact-checkpoint speedup</li><li>local-session-branching-summary.json: ${escapeHtml(formatNumber(branching.rewrite_avoided_bytes_estimate, 0))} avoided rewrite bytes estimated</li><li>local-agent-framework-showcase.json: ${escapeHtml(framework.client || "--")} / ${escapeHtml(framework.status || "--")}</li><li>local-openai-compatible-smoke.json: ${escapeHtml(openaiSmoke.status || "--")}</li></ul></div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("AgentMemory + AirLLM lane", "AgentMemory + AirLLM lane"))}</h4><p>${escapeHtml(
            s(
              "AgentMemory se absorbe como recall propio en SQLite, no como dependencia pesada. AirLLM se prepara como adapter seam: HeliX lee slices por capa desde .hlx para encajar con un lifecycle layer-by-layer sin forkear todavia.",
              "AgentMemory is absorbed as native SQLite recall, not as a heavy dependency. AirLLM is prepared as an adapter seam: HeliX reads per-layer slices from .hlx to fit a layer-by-layer lifecycle without forking yet."
            )
          )}</p><ul class="modal-list"><li>local-agent-memory-catalog-smoke.json: ${escapeHtml(memoryCatalog.privacy_redaction_ok === true ? "privacy redaction ok" : memoryCatalog.status || "--")}</li><li>local-hlx-layer-slice-smoke.json: ${escapeHtml(layerSlice.unrelated_array_loaded === false ? "selected layer only" : layerSlice.status || "--")}</li><li>local-airllm-bridge-smoke.json: ${escapeHtml(airllmBridge.bridge_mode || "--")}</li><li>local-memory-augmented-openai-smoke.json: ${escapeHtml(memoryOpenai.memory_context_injected === true ? "recall injected" : memoryOpenai.status || "--")}</li></ul></div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("Boundary", "Boundary"))}</h4><p>${escapeHtml(
            s(
              "Los estados no se comparten entre modelos distintos. La coordinacion multimodelo ocurre por mensajes, artifacts y sesiones propias de cada modelo.",
              "States are not shared across different models. Multimodel coordination happens through messages, artifacts, and model-specific sessions."
            )
          )}</p></div>`,
      },
      {
        id: "blueprints",
        number: "04",
        category: s("Blueprints", "Blueprints"),
        title: "BLUEPRINTS + META BUILD",
        teaser: s(
          `Meta Microsite: ${blueprintMeta?.quality_checks?.status || "--"} quality gate, ${blueprintMeta?.final_audit_status || "--"} final audit.`,
          `Meta Microsite: ${blueprintMeta?.quality_checks?.status || "--"} quality gate, ${blueprintMeta?.final_audit_status || "--"} final audit.`
        ),
        action: s("Open blueprints", "Open blueprints"),
        summary: s(
          "La capa de producto aparece como workloads versionados: no chats sueltos, sino equipos, handoffs, memoria, session policy y outputs reproducibles.",
          "The product layer appears as versioned workloads: not loose chats, but teams, handoffs, memory, session policy, and reproducible outputs."
        ),
        tags: ["blueprint", "inference-os", "meta-demo"],
        coverArt: reportCover("quality-first meta build", blueprintMeta?.quality_checks?.status || "--", "Blueprints"),
        content:
          `<div class="modal-inline-stats"><div><span>Quality</span><strong>${escapeHtml(blueprintMeta?.quality_checks?.status || "--")}</strong></div><div><span>Audit</span><strong>${escapeHtml(blueprintMeta?.final_audit_status || "--")}</strong></div><div><span>Stacks</span><strong>${escapeHtml(formatNumber((stackCatalog.stacks || []).length, 0))}</strong></div></div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("Evidence", "Evidence"))}</h4><ul class="modal-list"><li>local-blueprint-meta-microsite-demo.json: ${escapeHtml(blueprintMeta?.public_claim_level || "--")}</li><li>local-inference-os-architecture-summary.json: four-layer architecture</li><li>local-blueprint-stack-catalog.json: ${(stackCatalog.stacks || []).map((item) => escapeHtml(item.name || item.id)).join(", ")}</li></ul></div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("Public surface", "Public surface"))}</h4><p><a class="hero-link hero-link-accent" href="/meta-demo">Open the generated Meta Microsite</a></p></div>`,
      },
      {
        id: "prompts",
        number: "05",
        category: s("Prompt suite", "Prompt suite"),
        title: "PROMPT FAMILY READOUT",
        teaser: s(
          `La familia code quedo en ${formatRatio(data.promptAggregates?.code?.avg_speedup_vs_native)} frente a ${formatRatio(data.promptAggregates?.daily?.avg_speedup_vs_native)} en daily.`,
          `The code family landed at ${formatRatio(data.promptAggregates?.code?.avg_speedup_vs_native)} versus ${formatRatio(data.promptAggregates?.daily?.avg_speedup_vs_native)} on daily prompts.`
        ),
        action: s("See prompt readout", "See prompt readout"),
        summary: s(
          "No es un benchmark universal, pero si una lectura local util para hablar de tareas mas agentic o de codigo.",
          "It is not a universal benchmark, but it is a useful local read when talking about more code-heavy or agentic tasks."
        ),
        tags: ["code", "daily"],
        coverArt: reportCover("prompt suite", formatRatio(data.promptAggregates?.code?.avg_speedup_vs_native), "Prompt readout"),
        content:
          `<div class="modal-visual">${reportListFigure("prompts", data)}</div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("Reading", "Reading"))}</h4><p>${escapeHtml(
            s(
              "El resultado se cuenta mejor como senal exploratoria: code se mantuvo mejor que daily con el mismo ratio total de memoria hibrida.",
              "The result is best framed as an exploratory signal: code held up better than daily at the same total hybrid memory ratio."
            )
          )}</p></div>`,
      },
      {
        id: "stress",
        number: "06",
        category: s("Stress lab", "Stress lab"),
        title: "STRESS MISSIONS V1",
        teaser: s(
          `Restore equivalence: hash=${String(restoreMission?.headline_metrics?.hash_match === true)}, tokens=${String(restoreMission?.headline_metrics?.generated_ids_match === true)}, logit delta=${formatNumber(restoreMission?.headline_metrics?.max_abs_logit_delta, 4)}.`,
          `Restore equivalence: hash=${String(restoreMission?.headline_metrics?.hash_match === true)}, tokens=${String(restoreMission?.headline_metrics?.generated_ids_match === true)}, logit delta=${formatNumber(restoreMission?.headline_metrics?.max_abs_logit_delta, 4)}.`
        ),
        action: s("Open stress lab", "Open stress lab"),
        summary: s(
          "La capa de stress conecta integridad de snapshot, continuidad deterministica y degradacion segura ante distribuciones raras.",
          "The stress layer connects snapshot integrity, deterministic continuation, and safe degradation under odd distributions."
        ),
        tags: ["restore", "stress", "receipts", "TUI"],
        coverArt: reportCover("stress missions v1", `${formatNumber(stressMissions.length, 0)} missions`, "Restore equivalence"),
        content:
          `<div class="modal-inline-stats"><div><span>State hash</span><strong>${escapeHtml(String(stateMission?.headline_metrics?.hash_match === true))}</strong></div><div><span>Restore tokens</span><strong>${escapeHtml(String(restoreMission?.headline_metrics?.generated_ids_match === true))}</strong></div><div><span>Logit delta</span><strong>${escapeHtml(formatNumber(restoreMission?.headline_metrics?.max_abs_logit_delta, 4))}</strong></div></div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("What is verified", "What is verified"))}</h4><p>${escapeHtml(
            s(
              "State Juggler prueba integridad bit-perfect del snapshot serializado/restaurado. Restore Equivalence agrega una continuacion corta y deterministica donde generated_ids, top-1 y logits matchean antes y despues del restore.",
              "State Juggler proves bit-perfect integrity of the serialized/restored snapshot. Restore Equivalence adds a short deterministic continuation where generated_ids, top-1, and logits match before and after restore."
            )
          )}</p></div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("Stress results", "Stress results"))}</h4><ul class="modal-list"><li>${escapeHtml(s("Context Switcher promoted blocks", "Context Switcher promoted blocks"))}: ${escapeHtml(formatNumber(contextMission?.headline_metrics?.promoted_block_total, 0))}</li><li>${escapeHtml(s("Context Switcher finite logits", "Context Switcher finite logits"))}: ${escapeHtml(String(contextMission?.headline_metrics?.logits_finite === true))}</li><li>${escapeHtml(s("Long-Context Coder identifier recall", "Long-Context Coder identifier recall"))}: ${escapeHtml(formatNumber(longMission?.headline_metrics?.identifier_recall_passes, 0))}/${escapeHtml(formatNumber(longMission?.headline_metrics?.task_count, 0))}</li></ul></div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("Caveat", "Caveat"))}</h4><p>${escapeHtml(
            s(
              "El hash por si solo no prueba comprension semantica. La equivalencia restore es un probe corto, no una evaluacion amplia de calidad o contexto largo.",
              "The hash alone does not prove semantic understanding. Restore equivalence is a short probe, not a broad quality or long-context evaluation."
            )
          )}</p></div>`,
      },
      {
        id: "claims",
        number: "07",
        category: s("Claims matrix", "Claims matrix"),
        title: "VERIFIED, PROMISING, BLOCKED",
        teaser: s(
          `${formatNumber(claimCounts.verified, 0)} verified, ${formatNumber(claimCounts.promising, 0)} promising, ${formatNumber(claimCounts.blocked, 0)} blocked. Cada claim apunta a artifacts reales.`,
          `${formatNumber(claimCounts.verified, 0)} verified, ${formatNumber(claimCounts.promising, 0)} promising, ${formatNumber(claimCounts.blocked, 0)} blocked. Every claim points to real artifacts.`
        ),
        action: s("Open claims matrix", "Open claims matrix"),
        summary: s(
          "La matriz evita sobreclaiming: separa lo publicable, lo exploratorio y lo bloqueado.",
          "The matrix prevents overclaiming: it separates what is publishable, exploratory, and blocked."
        ),
        tags: ["claims", "verified", "blocked"],
        coverArt: reportCover("claims matrix", `${formatNumber(claimCounts.verified, 0)} verified`, "Evidence wording"),
        content:
          `<div class="modal-inline-stats"><div><span>Verified</span><strong>${escapeHtml(formatNumber(claimCounts.verified, 0))}</strong></div><div><span>Promising</span><strong>${escapeHtml(formatNumber(claimCounts.promising, 0))}</strong></div><div><span>Blocked</span><strong>${escapeHtml(formatNumber(claimCounts.blocked, 0))}</strong></div></div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("Verified wording", "Verified wording"))}</h4><ul class="modal-list">${verifiedClaimList}</ul></div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("Blocked or limited", "Blocked or limited"))}</h4><ul class="modal-list">${blockedClaimList}</ul></div>`,
      },
      {
        id: "side",
        number: "08",
        category: s("Side lanes", "Side lanes"),
        title: "HXQ + GEMMA",
        teaser: s(
          "HXQ Zamba2 sigue bloqueado por logits no finitos y Gemma permanece como lane secundaria.",
          "HXQ Zamba2 remains blocked by non-finite logits and Gemma stays a secondary lane."
        ),
        action: s("Check side lanes", "Check side lanes"),
        summary: s(
          "Importantes para el roadmap, pero no para forzar claims que todavia no estan listos.",
          "Important for the roadmap, but not for forcing claims that are not ready yet."
        ),
        tags: ["HXQ", "Gemma", "blocked"],
        coverArt: reportCover("side lanes", data.hxqDiagnostics?.logits_finite === true ? "finite" : "non-finite", "HXQ + Gemma"),
        content:
          `<div class="modal-visual">${reportListFigure("side", data)}</div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("What remains blocked", "What remains blocked"))}</h4><p>${escapeHtml(
            s(
              "El checkpoint HXQ de Zamba2 sigue dando logits no finitos en local. Gemma 3 quedo gated y Gemma 4 encontro limites de carga local.",
              "The Zamba2 HXQ checkpoint still gives non-finite logits locally. Gemma 3 stayed gated and Gemma 4 hit local loading limits."
            )
          )}</p></div>`,
      },
      {
        id: "method",
        number: "09",
        category: s("Method", "Method"),
        title: "METHOD + NEXT MOVES",
        teaser: s(
          "La direccion fuerte hoy es architecture-aware inference memory, con GPU hybrid validation y un ledger de artifacts que mantenga la historia honesta.",
          "The strong direction today is architecture-aware inference memory, with GPU hybrid validation and an artifact ledger that keeps the story honest."
        ),
        action: s("Read method", "Read method"),
        summary: s(
          "Lo siguiente no es otro tweak visual: es validar el frontier hibrido en GPU y seguir ordenando la historia publica alrededor de evidencia.",
          "The next move is not another visual tweak: it is validating the hybrid frontier on GPU and continuing to organize the public story around evidence."
        ),
        tags: ["roadmap", "artifacts"],
        coverArt: reportCover("next experiments", "GPU next", "Method + roadmap"),
        content:
          `<div class="modal-section"><h4>${escapeHtml(s("Where this goes", "Where this goes"))}</h4><p>${escapeHtml(
            s(
              "La narrativa fuerte es unir Transformer KV, recurrent-state compression y sesiones persistentes dentro de una misma historia de memory compression.",
              "The strong narrative is to unify Transformer KV, recurrent-state compression, and persistent sessions inside the same memory compression story."
            )
          )}</p></div>`,
      },
    ];
  }

  function renderStaticCopy() {
    document.documentElement.lang = state.locale;
    byId("research-brand-subtitle").textContent = s("Research Dossier", "Research Dossier");
    byId("research-nav-home").textContent = "Home";
    byId("research-nav-reports").textContent = "Reports";
    byId("research-nav-artifacts").textContent = "Artifacts";
    byId("research-hero-eyebrow").textContent = s("Editorial evidence", "Editorial evidence");
    byId("research-hero-subtitle").textContent = s(
      "Session OS, hybrid checkpoints, benchmarks, blocked lanes, and raw JSON.",
      "Session OS, hybrid checkpoints, benchmarks, blocked lanes, and raw JSON."
    );
    byId("research-hero-summary").textContent = s(
      "La home resume la tesis nueva: Helix como OS local de sesiones de inferencia. Esta pagina abre cada artifact, separa claims y deja visibles los caveats.",
      "The home page summarizes the new thesis: Helix as a local inference-session OS. This page opens every artifact, separates claims, and keeps caveats visible."
    );
    byId("research-hero-quote-title").textContent = s(
      "Un dossier para leer como pasamos de compresion KV a lifecycle de estados por agente y modelo.",
      "A dossier for reading how we moved from KV compression to state lifecycle per agent and model."
    );
    byId("research-hero-quote-copy").textContent = s(
      "Aca la narrativa se ordena en reportes: baseline Transformer, hybrid memory frontier, Session OS, stress missions, side lanes y artifact ledger completo.",
      "The narrative here is organized into reports: Transformer baseline, hybrid memory frontier, Session OS, stress missions, side lanes, and the full artifact ledger."
    );
    byId("research-hero-link-reports").textContent = s("Leer reports", "Read reports");
    byId("research-hero-link-artifacts").textContent = s("Abrir artifact ledger", "Open artifact ledger");
    byId("reports-title").textContent = "INDEX OF REPORTS";
    byId("reports-subtitle").textContent = s("Canonical evidence + lab notebook", "Canonical evidence + lab notebook");
    byId("artifact-ledger-title").textContent = "ARTIFACT LEDGER";
    byId("artifact-ledger-subtitle").textContent = s(
      "Browse every JSON behind the public story.",
      "Browse every JSON behind the public story."
    );
    byId("artifact-search-label").textContent = "Search";
    byId("artifact-search").placeholder = "remote, zamba2, hxq, gemma...";
    byId("artifact-code-title").textContent = "Raw JSON";
    byId("copy-json-button").textContent = "Copy JSON";
    byId("research-footer-status").textContent = s("Research surface online", "Research surface online");
    byId("research-footer-copy").textContent = "Helix Memory - JSON-backed microsite";

    document.querySelectorAll(".locale-btn").forEach((button) => {
      const active = button.dataset.locale === state.locale;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
  }

  function renderHero(data) {
    const restoreMission = (data.stressDashboard?.missions || []).find((item) => item.mission_id === "restore-equivalence") || {};
    const prefixModel = (data.prefixReuse?.models || [])[0] || {};
    byId("research-hero-metrics").innerHTML = [
      ["Hybrid prefix", formatRatio(data.hybridPrefix?.best_speedup)],
      ["Hybrid cache", formatRatio(data.hybridLocal?.combined_hybrid_gain?.hybrid_total_runtime_cache_ratio_vs_native)],
      ["Transformer prefix", formatRatio(prefixModel.claim_speedup_including_restore)],
      ["Recall", data.memoryCatalog?.status === "completed" ? "ok" : "--"],
      ["Layer slice", data.layerSlice?.unrelated_array_loaded === false ? "ok" : "--"],
      ["Restore", restoreMission?.headline_metrics?.generated_ids_match === true ? "match" : "--"],
      ["Artifacts", formatNumber(state.manifest.length, 0)],
    ]
      .map(([label, value]) => `<div class="metric-chip"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
      .join("");
    byId("research-hero-visual").innerHTML = heroVisual(data);
  }

  function renderReports() {
    byId("report-list").innerHTML = state.reports
      .map(
        (report) => `
      <article class="index-row" data-report-id="${escapeHtml(report.id)}">
        <div class="index-meta"><span class="index-number">${escapeHtml(report.number)}</span><span class="index-category">${escapeHtml(report.category)}</span></div>
        <div class="index-main"><p class="index-tagline">${escapeHtml(report.action)}</p><h3 class="index-title">${escapeHtml(report.title)}</h3><p class="index-teaser">${escapeHtml(report.teaser)}</p></div>
        <div class="index-action"><span class="index-action-label">${escapeHtml(s("Abrir", "Open"))}</span><span class="action-orb">↗</span></div>
      </article>
    `
      )
      .join("");

    document.querySelectorAll("[data-report-id]").forEach((node) => {
      node.addEventListener("click", () => openReport(node.getAttribute("data-report-id")));
    });
  }

  function openReport(reportId) {
    const report = state.reports.find((item) => item.id === reportId);
    if (!report) return;
    state.activeReportId = reportId;
    byId("research-modal-kicker").textContent = `${s("Report", "Report")} ${report.number}`;
    byId("research-modal-title").textContent = report.title;
    byId("research-modal-summary").textContent = report.summary;
    byId("research-modal-tags").innerHTML = report.tags.map((tag) => `<span class="modal-tag">${escapeHtml(tag)}</span>`).join("");
    byId("research-modal-cover-art").innerHTML = report.coverArt;
    byId("research-modal-content").innerHTML = report.content;
    byId("research-modal").hidden = false;
    document.body.style.overflow = "hidden";
  }

  function closeReport() {
    byId("research-modal").hidden = true;
    document.body.style.overflow = "";
  }

  function renderFilters() {
    const tags = new Set(["all"]);
    state.manifest.forEach((artifact) => {
      tags.add(artifact.category === "canonical evidence" ? "canonical" : "lab");
      (artifact.tags || []).forEach((tag) => tags.add(tag));
    });
    const order = ["all", "canonical", "lab", "verified", "blueprint", "inference-os", "stress", "claims", "gpu", "local", "hybrid", "hxq", "gemma", "blocked", "exploratory"];
    byId("artifact-filters").innerHTML = order
      .filter((tag) => tags.has(tag))
      .map((tag) => `<button type="button" class="filter-chip ${state.selectedTag === tag ? "active" : ""}" data-tag="${escapeHtml(tag)}">${escapeHtml(tag)}</button>`)
      .join("");
    document.querySelectorAll(".filter-chip").forEach((button) => {
      button.addEventListener("click", () => {
        state.selectedTag = button.dataset.tag;
        renderArtifactBrowser().catch((error) => console.error(error));
      });
    });
  }

  function artifactMatches(artifact) {
    const haystack = [artifact.name, artifact.title, artifact.category, artifact.status, ...(artifact.tags || []), ...(artifact.model_refs || [])]
      .join(" ")
      .toLowerCase();
    const matchesSearch = haystack.includes(state.search.toLowerCase());
    const matchesTag =
      state.selectedTag === "all" ||
      (state.selectedTag === "canonical" && artifact.category === "canonical evidence") ||
      (state.selectedTag === "lab" && artifact.category === "lab notebook") ||
      (artifact.tags || []).includes(state.selectedTag) ||
      artifact.status === state.selectedTag;
    return matchesSearch && matchesTag;
  }

  async function ensureArtifact(name) {
    if (state.artifacts[name]) return state.artifacts[name];
    const artifact = await fetchArtifactWithFallback(name);
    state.artifacts[name] = artifact;
    return artifact;
  }

  function metricCard(label, value) {
    return `<div class="artifact-summary-card"><div class="widget-title">${escapeHtml(label)}</div><div>${escapeHtml(String(value))}</div></div>`;
  }

  async function renderArtifactDetail(name) {
    const meta = state.manifest.find((item) => item.name === name);
    if (!meta) return;
    const artifact = await ensureArtifact(name);
    const payload = artifact.payload || {};
    byId("artifact-detail-header").innerHTML = `<div class="eyebrow">${escapeHtml(meta.category)}</div><h3>${escapeHtml(meta.title)}</h3><p class="artifact-meta">${escapeHtml(name)}</p><div class="filter-row">${(meta.tags || []).map((tag) => `<span class="artifact-badge">${escapeHtml(tag)}</span>`).join("")}</div>`;
    const headlineCards = Object.entries(meta.headline_metrics || {})
      .slice(0, 6)
      .map(([key, value]) => metricCard(key.replace(/_/g, " "), typeof value === "number" ? formatNumber(value) : value));
    byId("artifact-detail-summary").innerHTML = [metricCard("status", meta.status), metricCard("benchmark", meta.benchmark_kind || "--")]
      .concat(headlineCards)
      .join("");
    const raw = JSON.stringify(payload, null, 2);
    byId("artifact-json").textContent = raw;
    byId("copy-json-button").onclick = async () => {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(raw);
      byId("copy-json-button").textContent = "Copied";
      window.setTimeout(() => {
        byId("copy-json-button").textContent = "Copy JSON";
      }, 1000);
    };
  }

  async function renderArtifactBrowser() {
    renderFilters();
    const filtered = state.manifest.filter(artifactMatches);
    byId("artifact-list").innerHTML = filtered.length
      ? filtered
          .map(
            (artifact) =>
              `<article class="artifact-card ${artifact.name === state.selectedArtifactName ? "active" : ""}" data-artifact-name="${escapeHtml(artifact.name)}"><div class="widget-title">${escapeHtml(artifact.category)}</div><h3>${escapeHtml(artifact.title)}</h3><p class="artifact-meta">${escapeHtml((artifact.model_refs || []).slice(0, 3).join(" - ") || artifact.name)}</p><div class="filter-row">${(artifact.tags || []).slice(0, 4).map((tag) => `<span class="artifact-badge">${escapeHtml(tag)}</span>`).join("")}</div></article>`
          )
          .join("")
      : `<div class="artifact-detail detail-empty">${escapeHtml(s("No hay artifacts para esos filtros.", "No artifacts match those filters."))}</div>`;

    document.querySelectorAll("[data-artifact-name]").forEach((node) => {
      node.addEventListener("click", async () => {
        state.selectedArtifactName = node.getAttribute("data-artifact-name");
        await renderArtifactBrowser();
      });
    });

    const fallbackName = filtered.some((item) => item.name === state.selectedArtifactName) ? state.selectedArtifactName : filtered[0]?.name;
    if (!fallbackName) {
      byId("artifact-detail-header").innerHTML = "";
      byId("artifact-detail-summary").innerHTML = "";
      byId("artifact-json").textContent = "";
      return;
    }
    state.selectedArtifactName = fallbackName;
    await renderArtifactDetail(fallbackName);
  }

  async function loadResearch() {
    state.manifest = await fetchManifestWithFallback();
    const [
      frontierSummary,
      promptSuite,
      hxqDiagnostics,
      gemmaAttempts,
      stressDashboard,
      claimsMatrix,
      prefixReuse,
      hybridPrefix,
      sessionBranching,
      agentFramework,
      openaiSmoke,
      memoryCatalog,
      layerSlice,
      airllmBridge,
      memoryOpenai,
      inferenceOs,
      stackCatalog,
      blueprintMeta,
    ] = await Promise.all([
      fetchArtifactWithFallback("hybrid-memory-frontier-summary.json"),
      fetchArtifactWithFallback("local-zamba2-prompt-suite-code-daily.json").catch(() => ({})),
      fetchArtifactWithFallback("local-zamba2-hxq-direct-diagnostics.json").catch(() => ({})),
      fetchArtifactWithFallback("local-gemma-attempts.json").catch(() => ({})),
      fetchArtifactWithFallback("local-zamba2-stress-dashboard.json").catch(() => ({})),
      fetchArtifactWithFallback("helix-claims-matrix.json").catch(() => ({})),
      fetchArtifactWithFallback("local-prefix-reuse-ttft-summary.json").catch(() => ({})),
      fetchArtifactWithFallback("local-hybrid-prefix-checkpoint-summary.json").catch(() => ({})),
      fetchArtifactWithFallback("local-session-branching-summary.json").catch(() => ({})),
      fetchArtifactWithFallback("local-agent-framework-showcase.json").catch(() => ({})),
      fetchArtifactWithFallback("local-openai-compatible-smoke.json").catch(() => ({})),
      fetchArtifactWithFallback("local-agent-memory-catalog-smoke.json").catch(() => ({})),
      fetchArtifactWithFallback("local-hlx-layer-slice-smoke.json").catch(() => ({})),
      fetchArtifactWithFallback("local-airllm-bridge-smoke.json").catch(() => ({})),
      fetchArtifactWithFallback("local-memory-augmented-openai-smoke.json").catch(() => ({})),
      fetchArtifactWithFallback("local-inference-os-architecture-summary.json").catch(() => ({})),
      fetchArtifactWithFallback("local-blueprint-stack-catalog.json").catch(() => ({})),
      fetchArtifactWithFallback("local-blueprint-meta-microsite-demo.json").catch(() => ({})),
    ]);
    state.datasets = {
      frontierSummary,
      promptSuite,
      hxqDiagnostics,
      gemmaAttempts,
      stressDashboard,
      claimsMatrix,
      prefixReuse,
      hybridPrefix,
      sessionBranching,
      agentFramework,
      openaiSmoke,
      memoryCatalog,
      layerSlice,
      airllmBridge,
      memoryOpenai,
      inferenceOs,
      stackCatalog,
      blueprintMeta,
    };
    const data = getData();
    state.reports = buildReports(data);
    renderHero(data);
    renderReports();
    await renderArtifactBrowser();
    if (state.activeReportId) openReport(state.activeReportId);
  }

  function bindLocale() {
    document.querySelectorAll(".locale-btn").forEach((button) => {
      button.addEventListener("click", () => {
        state.locale = button.dataset.locale;
        window.localStorage.setItem("helix-research-locale", state.locale);
        renderStaticCopy();
        if (state.datasets) {
          const data = getData();
          state.reports = buildReports(data);
          renderHero(data);
          renderReports();
          renderArtifactBrowser().catch((error) => console.error(error));
          if (state.activeReportId) openReport(state.activeReportId);
        }
      });
    });
  }

  function bindEvents() {
    byId("research-modal-close").addEventListener("click", closeReport);
    byId("research-modal").addEventListener("click", (event) => {
      if (event.target === byId("research-modal")) closeReport();
    });
    byId("artifact-search").addEventListener("input", (event) => {
      state.search = event.target.value || "";
      renderArtifactBrowser().catch((error) => console.error(error));
    });
  }

  renderStaticCopy();
  bindLocale();
  bindEvents();
  loadResearch().catch((error) => {
    console.error(error);
    byId("report-list").innerHTML = `<article class="index-row"><div class="index-meta"><span class="index-number">!!</span><span class="index-category">Error</span></div><div class="index-main"><h3 class="index-title">${escapeHtml(s("No pude cargar la research surface.", "Could not load the research surface."))}</h3></div></article>`;
  });
})();
