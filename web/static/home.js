(function () {
  const state = {
    locale: window.localStorage.getItem("helix-research-locale") || "es",
    dataSource: "backend",
    datasets: null,
    chapters: [],
    activeChapterId: null,
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

  function formatPercent(value, digits = 2) {
    if (value == null || Number.isNaN(Number(value))) return "--";
    return `${formatNumber(value, digits)}%`;
  }

  function compactModelLabel(modelRef) {
    return String(modelRef || "--")
      .replace("Qwen/", "")
      .replace("HuggingFaceTB/", "")
      .replace("Zyphra/", "")
      .replace("EchoLabs33/", "");
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

  function getData() {
    const manifest = state.datasets.manifest || [];
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
    const promptEntries = Array.from(
      new Set((promptSuite.prompt_suite || []).map((entry) => entry.prompt_key).filter(Boolean))
    );

    return {
      manifest,
      transformerModels,
      bestTransformer,
      hybridLocal,
      promptAggregates,
      promptEntries,
      canonicalCount: manifest.filter((item) => item.category === "canonical evidence").length,
      labCount: manifest.filter((item) => item.category === "lab notebook").length,
      hxqDiagnostics,
      gemmaModels: gemmaAttempts?.models || [],
      blockedGemma: (gemmaAttempts?.models || []).filter((item) => String(item.status || "").includes("blocked")).length,
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

  function promptName(key) {
    const map = {
      code_completion: "merge_intervals completion",
      code_debug: "LRU cache debugging",
      daily_planning: "Buenos Aires planning",
      daily_buying: "used laptop checklist",
    };
    return map[key] || key;
  }

  function renderHeroVisual(data) {
    const transformer = formatRatio(data.bestTransformer?.best_compression_kv_ratio_vs_native);
    const hybrid = formatRatio(data.hybridLocal?.combined_hybrid_gain?.hybrid_total_runtime_cache_ratio_vs_native);
    const hybridPrefixSpeedup = formatRatio(data.hybridPrefix?.best_speedup);
    const prefixModel = (data.prefixReuse?.models || [])[0] || {};
    const prefixClaim = formatRatio(prefixModel.claim_speedup_including_restore);
    const branchCount = formatNumber(data.sessionBranching?.branch_count, 0);

    byId("hero-visual").innerHTML = svgShell(
      "0 0 760 540",
      `
        <line x1="70" y1="488" x2="382" y2="172" stroke="rgba(28,28,26,0.18)" stroke-width="1.3" />
        <line x1="690" y1="488" x2="382" y2="172" stroke="rgba(28,28,26,0.18)" stroke-width="1.3" />
        <line x1="382" y1="172" x2="382" y2="36" stroke="rgba(28,28,26,0.14)" stroke-width="1.1" />
        <line x1="26" y1="350" x2="734" y2="350" stroke="rgba(28,28,26,0.12)" stroke-width="1" stroke-dasharray="8 8" />
        <line x1="26" y1="432" x2="734" y2="432" stroke="rgba(28,28,26,0.10)" stroke-width="1" stroke-dasharray="8 8" />
        <circle cx="382" cy="172" r="114" stroke="rgba(28,28,26,0.14)" stroke-width="1.2" />
        <circle cx="382" cy="172" r="18" fill="#D64933" />
        <circle cx="382" cy="172" r="6" fill="#EBE6DF" />
        <circle cx="650" cy="128" r="30" stroke="rgba(28,28,26,0.14)" stroke-width="1" />
        <line x1="540" y1="128" x2="620" y2="128" stroke="rgba(28,28,26,0.18)" stroke-width="1" />
        <text x="60" y="98" fill="rgba(28,28,26,0.6)" font-size="12" font-family="Inter, sans-serif" letter-spacing="2">HELIX SESSION OS</text>
        <text x="56" y="154" fill="rgba(28,28,26,0.12)" font-size="96" font-family="Cormorant Garamond, serif">${escapeHtml(hybrid)}</text>
        <text x="58" y="182" fill="rgba(28,28,26,0.66)" font-size="13" font-family="Inter, sans-serif" letter-spacing="1.4">combined hybrid runtime-cache ratio</text>
        <rect x="58" y="206" width="250" height="10" rx="5" fill="rgba(28,28,26,0.08)" />
        <rect x="58" y="206" width="212" height="10" rx="5" fill="#D64933" />
        <rect x="430" y="286" width="252" height="92" rx="22" fill="rgba(255,255,255,0.48)" stroke="rgba(28,28,26,0.16)" />
        <text x="454" y="318" fill="rgba(28,28,26,0.56)" font-size="11" font-family="Inter, sans-serif" letter-spacing="2">HYBRID PREFIX</text>
        <text x="454" y="360" fill="#1C1C1A" font-size="48" font-family="Cormorant Garamond, serif">${escapeHtml(hybridPrefixSpeedup)}</text>
        <text x="454" y="392" fill="rgba(28,28,26,0.72)" font-size="13" font-family="Inter, sans-serif">exact checkpoint TTFT speedup</text>
        <rect x="430" y="408" width="252" height="72" rx="22" fill="rgba(255,255,255,0.44)" stroke="rgba(28,28,26,0.16)" />
        <text x="454" y="436" fill="rgba(28,28,26,0.56)" font-size="11" font-family="Inter, sans-serif" letter-spacing="2">TRANSFORMER PREFIX</text>
        <text x="454" y="466" fill="#1C1C1A" font-size="34" font-family="Cormorant Garamond, serif">${escapeHtml(prefixClaim)}</text>
        <text x="538" y="466" fill="rgba(28,28,26,0.72)" font-size="13" font-family="Inter, sans-serif">top-1 stable</text>
        <rect x="94" y="390" width="232" height="96" rx="22" fill="rgba(255,255,255,0.44)" stroke="rgba(28,28,26,0.16)" />
        <text x="120" y="422" fill="rgba(28,28,26,0.56)" font-size="11" font-family="Inter, sans-serif" letter-spacing="2">SESSION BRANCHES</text>
        <text x="120" y="458" fill="#1C1C1A" font-size="38" font-family="Cormorant Garamond, serif">${escapeHtml(branchCount)}</text>
        <text x="176" y="458" fill="rgba(28,28,26,0.72)" font-size="13" font-family="Inter, sans-serif">verified branches</text>
        <text x="120" y="482" fill="rgba(28,28,26,0.72)" font-size="13" font-family="Inter, sans-serif">SQLite catalog + .hlx sessions</text>
        <rect x="430" y="184" width="252" height="78" rx="22" fill="rgba(28,28,26,0.92)" stroke="rgba(28,28,26,0.16)" />
        <text x="454" y="214" fill="rgba(235,230,223,0.66)" font-size="11" font-family="Inter, sans-serif" letter-spacing="2">TRANSFORMER KV</text>
        <text x="454" y="246" fill="#EBE6DF" font-size="38" font-family="Cormorant Garamond, serif">${escapeHtml(transformer)}</text>
        <text x="552" y="246" fill="rgba(235,230,223,0.72)" font-size="13" font-family="Inter, sans-serif">match=true</text>
      `
    );
  }

  function simpleFigure(kind, data) {
    if (kind === "architecture") {
      return svgShell(
        "0 0 760 260",
        `
          <rect x="16" y="16" width="728" height="228" rx="24" fill="rgba(255,255,255,0.44)" stroke="rgba(28,28,26,0.15)" />
          <rect x="42" y="72" width="180" height="116" rx="24" fill="#FFFFFF" stroke="#1C1C1A" />
          <text x="64" y="106" fill="#1C1C1A" font-size="14" font-family="Inter, sans-serif" letter-spacing="2">TRANSFORMER</text>
          <text x="64" y="152" fill="#1C1C1A" font-size="38" font-family="Cormorant Garamond, serif">KV cache</text>
          <line x1="222" y1="130" x2="344" y2="130" stroke="#1C1C1A" stroke-dasharray="7 6" />
          <circle cx="378" cy="130" r="22" fill="#D64933" />
          <line x1="402" y1="130" x2="522" y2="130" stroke="#1C1C1A" stroke-dasharray="7 6" />
          <rect x="522" y="50" width="180" height="62" rx="22" fill="#FFFFFF" stroke="#1C1C1A" />
          <text x="546" y="88" fill="#1C1C1A" font-size="32" font-family="Cormorant Garamond, serif">KV lane</text>
          <rect x="522" y="148" width="180" height="62" rx="22" fill="#FFFFFF" stroke="#1C1C1A" />
          <text x="538" y="186" fill="#1C1C1A" font-size="28" font-family="Cormorant Garamond, serif">state lane</text>
        `
      );
    }

    if (kind === "transformer") {
      const rows = data.transformerModels
        .map((item, index) => {
          const ratio = item.best_compression_kv_ratio_vs_native;
          const width = 340 * ((ratio || 0) / 3.1);
          const y = 54 + index * 72;
          return `
            <text x="38" y="${y}" fill="#1C1C1A" font-size="15" font-family="Inter, sans-serif">${escapeHtml(compactModelLabel(item.model_ref))}</text>
            <rect x="38" y="${y + 16}" width="404" height="22" rx="11" fill="rgba(28,28,26,0.08)" stroke="rgba(28,28,26,0.16)" />
            <rect x="38" y="${y + 16}" width="${Math.max(30, width)}" height="22" rx="11" fill="${index === 1 ? "#1C1C1A" : "#D64933"}" />
            <text x="462" y="${y + 34}" fill="#1C1C1A" font-size="22" font-family="Cormorant Garamond, serif">${escapeHtml(formatRatio(ratio))}</text>
          `;
        })
        .join("");
      return svgShell("0 0 560 280", `<rect x="12" y="12" width="536" height="256" rx="24" fill="rgba(255,255,255,0.44)" stroke="rgba(28,28,26,0.18)" />${rows}`);
    }

    if (kind === "hybrid") {
      const items = [
        ["KV-only", data.hybridLocal?.kv_only_gain?.hybrid_total_runtime_cache_ratio_vs_native, "#F1D8D3"],
        ["State-only", data.hybridLocal?.mamba_state_only_gain?.hybrid_total_runtime_cache_ratio_vs_native, "#F4E5B5"],
        ["Combined", data.hybridLocal?.combined_hybrid_gain?.hybrid_total_runtime_cache_ratio_vs_native, "#D64933"],
      ];
      const bars = items
        .map(([label, value, fill], index) => {
          const width = 260 * ((value || 0) / 4.1);
          const y = 66 + index * 74;
          return `
            <text x="34" y="${y}" fill="#1C1C1A" font-size="14" font-family="Inter, sans-serif">${label}</text>
            <rect x="34" y="${y + 16}" width="300" height="22" rx="11" fill="rgba(28,28,26,0.08)" stroke="rgba(28,28,26,0.16)" />
            <rect x="34" y="${y + 16}" width="${Math.max(28, width)}" height="22" rx="11" fill="${fill}" />
            <text x="358" y="${y + 34}" fill="#1C1C1A" font-size="22" font-family="Cormorant Garamond, serif">${escapeHtml(formatRatio(value))}</text>
          `;
        })
        .join("");
      const speedup = data.hybridLocal?.combined_hybrid_gain?.speedup_vs_native;
      return svgShell(
        "0 0 500 330",
        `<rect x="12" y="12" width="476" height="306" rx="24" fill="rgba(255,255,255,0.44)" stroke="rgba(28,28,26,0.18)" /><text x="34" y="42" fill="#1C1C1A" font-size="12" font-family="Inter, sans-serif" letter-spacing="2">Hybrid total runtime-cache vs native</text>${bars}<rect x="34" y="264" width="188" height="38" rx="18" fill="#1C1C1A" /><text x="54" y="288" fill="#EBE6DF" font-size="12" font-family="Inter, sans-serif" letter-spacing="2">combined speedup</text><text x="236" y="289" fill="#1C1C1A" font-size="24" font-family="Cormorant Garamond, serif">${escapeHtml(formatRatio(speedup))}</text>`
      );
    }

    if (kind === "prompts") {
      const code = data.promptAggregates?.code?.avg_speedup_vs_native;
      const daily = data.promptAggregates?.daily?.avg_speedup_vs_native;
      return svgShell(
        "0 0 500 250",
        `<rect x="12" y="12" width="476" height="226" rx="24" fill="rgba(255,255,255,0.44)" stroke="rgba(28,28,26,0.18)" /><text x="34" y="44" fill="#1C1C1A" font-size="12" font-family="Inter, sans-serif" letter-spacing="2">Prompt family speedup vs native</text><text x="34" y="96" fill="#1C1C1A" font-size="14" font-family="Inter, sans-serif">code</text><rect x="34" y="112" width="300" height="22" rx="11" fill="rgba(28,28,26,0.08)" stroke="rgba(28,28,26,0.16)" /><rect x="34" y="112" width="${Math.max(28, 260 * ((code || 0) / 1.3))}" height="22" rx="11" fill="#D64933" /><text x="358" y="130" fill="#1C1C1A" font-size="22" font-family="Cormorant Garamond, serif">${escapeHtml(formatRatio(code))}</text><text x="34" y="174" fill="#1C1C1A" font-size="14" font-family="Inter, sans-serif">daily</text><rect x="34" y="190" width="300" height="22" rx="11" fill="rgba(28,28,26,0.08)" stroke="rgba(28,28,26,0.16)" /><rect x="34" y="190" width="${Math.max(28, 260 * ((daily || 0) / 1.3))}" height="22" rx="11" fill="#1C1C1A" /><text x="358" y="208" fill="#1C1C1A" font-size="22" font-family="Cormorant Garamond, serif">${escapeHtml(formatRatio(daily))}</text>`
      );
    }

    if (kind === "session-os") {
      const prefixModel = (data.prefixReuse?.models || [])[0] || {};
      const hybridPrefix = data.hybridPrefix || {};
      const branching = data.sessionBranching || {};
      const openaiStatus = data.openaiSmoke?.status || data.agentFramework?.status || "--";
      return svgShell(
        "0 0 620 330",
        `<rect x="12" y="12" width="596" height="306" rx="26" fill="rgba(255,255,255,0.44)" stroke="rgba(28,28,26,0.18)" />
        <text x="34" y="46" fill="#1C1C1A" font-size="12" font-family="Inter, sans-serif" letter-spacing="2">SESSION OS CONTROL PLANE</text>
        <rect x="34" y="72" width="168" height="92" rx="20" fill="#1C1C1A" />
        <text x="54" y="104" fill="rgba(235,230,223,0.68)" font-size="10" font-family="Inter, sans-serif" letter-spacing="2">PREFIX RESOLVER</text>
        <text x="54" y="138" fill="#EBE6DF" font-size="34" font-family="Cormorant Garamond, serif">${escapeHtml(formatRatio(prefixModel.claim_speedup_including_restore))}</text>
        <rect x="226" y="72" width="168" height="92" rx="20" fill="#D64933" />
        <text x="246" y="104" fill="rgba(235,230,223,0.78)" font-size="10" font-family="Inter, sans-serif" letter-spacing="2">HYBRID CHECKPOINT</text>
        <text x="246" y="138" fill="#EBE6DF" font-size="34" font-family="Cormorant Garamond, serif">${escapeHtml(formatRatio(hybridPrefix.best_speedup))}</text>
        <rect x="418" y="72" width="156" height="92" rx="20" fill="#FFFFFF" stroke="#1C1C1A" />
        <text x="438" y="104" fill="rgba(28,28,26,0.62)" font-size="10" font-family="Inter, sans-serif" letter-spacing="2">BRANCHES</text>
        <text x="438" y="138" fill="#1C1C1A" font-size="34" font-family="Cormorant Garamond, serif">${escapeHtml(formatNumber(branching.branch_count, 0))}</text>
        <line x1="118" y1="204" x2="500" y2="204" stroke="#1C1C1A" stroke-width="1.4" stroke-dasharray="8 7" />
        <circle cx="118" cy="204" r="18" fill="#1C1C1A" /><circle cx="310" cy="204" r="18" fill="#D64933" /><circle cx="500" cy="204" r="18" fill="#1C1C1A" />
        <text x="54" y="256" fill="#1C1C1A" font-size="14" font-family="Inter, sans-serif">catalog</text>
        <text x="260" y="256" fill="#1C1C1A" font-size="14" font-family="Inter, sans-serif">checkpoint</text>
        <text x="454" y="256" fill="#1C1C1A" font-size="14" font-family="Inter, sans-serif">OpenAI API</text>
        <text x="34" y="296" fill="rgba(28,28,26,0.68)" font-size="13" font-family="Inter, sans-serif">client status: ${escapeHtml(openaiStatus)} · exact-prefix only · audit-backed artifacts</text>`
      );
    }

    const total = data.manifest.length;
    return svgShell(
      "0 0 500 240",
      `<rect x="12" y="12" width="476" height="216" rx="24" fill="rgba(255,255,255,0.44)" stroke="rgba(28,28,26,0.18)" /><text x="34" y="46" fill="#1C1C1A" font-size="12" font-family="Inter, sans-serif" letter-spacing="2">Artifact ledger</text><text x="34" y="92" fill="#1C1C1A" font-size="60" font-family="Cormorant Garamond, serif">${escapeHtml(formatNumber(total, 0))}</text><text x="34" y="116" fill="rgba(28,28,26,0.72)" font-size="14" font-family="Inter, sans-serif">visible JSON artifacts</text><rect x="248" y="56" width="208" height="42" rx="20" fill="#FFFFFF" stroke="#1C1C1A" /><text x="266" y="82" fill="#1C1C1A" font-size="13" font-family="Inter, sans-serif">canonical evidence - ${escapeHtml(formatNumber(data.canonicalCount, 0))}</text><rect x="248" y="110" width="208" height="42" rx="20" fill="#FFFFFF" stroke="#1C1C1A" /><text x="266" y="136" fill="#1C1C1A" font-size="13" font-family="Inter, sans-serif">lab notebook - ${escapeHtml(formatNumber(data.labCount, 0))}</text>`
    );
  }

  function chapterCover(title, stat, note) {
    return svgShell(
      "0 0 900 300",
      `<rect x="34" y="46" width="832" height="210" rx="30" stroke="rgba(235,230,223,0.24)" /><text x="68" y="110" fill="rgba(235,230,223,0.72)" font-size="14" font-family="Inter, sans-serif" letter-spacing="2">${escapeHtml(note)}</text><text x="68" y="176" fill="#EBE6DF" font-size="72" font-family="Cormorant Garamond, serif">${escapeHtml(stat)}</text><text x="68" y="220" fill="rgba(235,230,223,0.84)" font-size="16" font-family="Inter, sans-serif">${escapeHtml(title)}</text>`
    );
  }

  function buildChapters(data) {
    const bestRatio = data.bestTransformer?.best_compression_kv_ratio_vs_native;
    const kvRatio = data.hybridLocal?.kv_only_gain?.hybrid_total_runtime_cache_ratio_vs_native;
    const stateRatio = data.hybridLocal?.mamba_state_only_gain?.hybrid_total_runtime_cache_ratio_vs_native;
    const combinedRatio = data.hybridLocal?.combined_hybrid_gain?.hybrid_total_runtime_cache_ratio_vs_native;
    const combinedSpeedup = data.hybridLocal?.combined_hybrid_gain?.speedup_vs_native;
    const code = data.promptAggregates?.code?.avg_speedup_vs_native;
    const daily = data.promptAggregates?.daily?.avg_speedup_vs_native;
    const hxqFinite = data.hxqDiagnostics?.logits_finite === true;
    const stressMissions = data.stressDashboard?.missions || [];
    const restoreMission = stressMissions.find((item) => item.mission_id === "restore-equivalence") || {};
    const stateMission = stressMissions.find((item) => item.mission_id === "state-juggler") || {};
    const contextMission = stressMissions.find((item) => item.mission_id === "context-switcher") || {};
    const prefixModel = (data.prefixReuse?.models || [])[0] || {};
    const prefixClaimSpeedup = prefixModel.claim_speedup_including_restore;
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
    const transformerList = data.transformerModels
      .map(
        (item) =>
          `<li><strong>${escapeHtml(compactModelLabel(item.model_ref))}</strong> - ${escapeHtml(item.best_compression_variant)} - ${escapeHtml(formatRatio(item.best_compression_kv_ratio_vs_native))} - delta PPL ${escapeHtml(formatPercent(item.best_compression_prompt_perplexity_delta_pct_vs_native))} - match=${escapeHtml(String(item.best_compression_match_vs_baseline))}</li>`
      )
      .join("");
    const promptList = data.promptEntries.map((key) => `<li>${escapeHtml(promptName(key))}</li>`).join("");

    return [
      {
        id: "problem",
        number: "01",
        category: s("Story", "Story"),
        title: "WHY THIS STARTED",
        teaser: s(
          "Helix arranco como una forma de comprimir KV cache para correr modelos serios en hardware limitado.",
          "Helix started as a way to compress KV cache so serious models could run on limited hardware."
        ),
        action: s("Open chapter", "Open chapter"),
        tags: [s("problem", "problem"), s("simple", "simple"), s("architecture", "architecture")],
        summary: s(
          "La pregunta inicial era concreta: como reducir la memoria de inferencia sin romper la salida. La pregunta actual es mas grande: que memoria importa para cada arquitectura.",
          "The initial question was concrete: how do we reduce inference memory without breaking the output. The current question is larger: which memory matters for each architecture."
        ),
        coverArt: chapterCover("from KV compression to architecture-aware memory", "KV -> state", "Helix Memory"),
        content:
          `<div class="modal-section"><h4>${escapeHtml(s("For dummies", "For dummies"))}</h4><p>${escapeHtml(
            s(
              "En un Transformer puro, la memoria de inferencia vive sobre todo en el KV cache. En un hibrido como Zamba2, una parte grande del cuello cambia de lugar y se muda a recurrent state.",
              "In a pure Transformer, inference memory mostly lives inside KV cache. In a hybrid like Zamba2, a large part of the bottleneck moves and ends up inside recurrent state."
            )
          )}</p></div>` +
          `<div class="modal-visual">${simpleFigure("architecture", data)}</div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("Why the story changes", "Why the story changes"))}</h4><p>${escapeHtml(
            s(
              "Eso obliga a abandonar la idea de una receta unica. La historia fuerte pasa a ser comprimir la memoria correcta para cada modelo.",
              "That forces us to abandon the idea of one single recipe. The strong story becomes compressing the right memory for each model."
            )
          )}</p></div>`,
      },
      {
        id: "session-os",
        number: "02",
        category: s("Session OS", "Session OS"),
        title: "WHEN MEMORY BECAME A LIFECYCLE",
        teaser: s(
          `El salto nuevo: catalogo SQLite, sesiones .hlx, prefix resolver, branching y clientes OpenAI-compatible sobre el mismo runtime.`,
          `The new jump: SQLite catalog, .hlx sessions, prefix resolver, branching, and OpenAI-compatible clients on the same runtime.`
        ),
        action: s("Open Session OS", "Open Session OS"),
        tags: ["catalog", "prefix", "OpenAI API", "branching"],
        summary: s(
          "Helix ya no es solo un compresor. Empieza a parecer un sistema operativo local para sesiones de inferencia: registra, restaura, verifica, ramifica y expone memoria de trabajo por agente/modelo.",
          "Helix is no longer only a compressor. It starts to look like a local operating system for inference sessions: cataloging, restoring, verifying, branching, and exposing working memory per agent/model."
        ),
        coverArt: chapterCover("catalog + prefix + checkpoint + API", "Session OS", "Inference sessions"),
        content:
          `<div class="modal-inline-stats"><div><span>Transformer prefix</span><strong>${escapeHtml(formatRatio(prefixClaimSpeedup))}</strong></div><div><span>Hybrid checkpoint</span><strong>${escapeHtml(formatRatio(hybridPrefix.best_speedup))}</strong></div><div><span>Branches</span><strong>${escapeHtml(formatNumber(branching.branch_count, 0))}</strong></div></div>` +
          `<div class="modal-inline-stats"><div><span>Recall</span><strong>${escapeHtml(memoryCatalog.status === "completed" ? "SQLite" : "--")}</strong></div><div><span>Layer slice</span><strong>${escapeHtml(layerSlice.read_mode || "--")}</strong></div><div><span>AirLLM seam</span><strong>${escapeHtml(airllmBridge.all_layer_injections_hit === true ? "hit" : "--")}</strong></div></div>` +
          `<div class="modal-visual">${simpleFigure("session-os", data)}</div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("What changed", "What changed"))}</h4><p>${escapeHtml(
            s(
              "La categoria nueva es session lifecycle: el sistema sabe que agente/modelo tiene que estado, si esta pending o verified, cuanto cuesta despertarlo y que prefijo puede reutilizar.",
              "The new category is session lifecycle: the system knows which agent/model owns which state, whether it is pending or verified, how much it costs to wake it up, and which prefix can be reused."
            )
          )}</p></div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("AgentMemory + AirLLM direction", "AgentMemory + AirLLM direction"))}</h4><p>${escapeHtml(
            s(
              "La memoria tipo AgentMemory entra como recall propio en SQLite, sin vector DB pesada. AirLLM entra como seam: HeliX ya puede leer solo la capa N desde .hlx y alimentar un lifecycle layer-by-layer, todavia sin fork ni punteros crudos.",
              "AgentMemory-style memory enters as native SQLite recall, without a heavy vector DB. AirLLM enters as a seam: HeliX can already read only layer N from .hlx and feed a layer-by-layer lifecycle, still without a fork or raw pointers."
            )
          )}</p><ul class="modal-list"><li>local-agent-memory-catalog-smoke.json: ${escapeHtml(memoryCatalog.privacy_redaction_ok === true ? "privacy redaction ok" : memoryCatalog.status || "--")}</li><li>local-hlx-layer-slice-smoke.json: ${escapeHtml(layerSlice.unrelated_array_loaded === false ? "selected layer only" : layerSlice.status || "--")}</li><li>local-airllm-bridge-smoke.json: ${escapeHtml(airllmBridge.bridge_mode || "--")}</li><li>local-memory-augmented-openai-smoke.json: ${escapeHtml(memoryOpenai.memory_context_injected === true ? "recall injected" : memoryOpenai.status || "--")}</li></ul></div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("Killer demo path", "Killer demo path"))}</h4><ul class="modal-list"><li>OpenAI SDK: ${escapeHtml(framework.status || "--")} / ${escapeHtml(openaiSmoke.status || "--")}</li><li>LangChain: ${escapeHtml(s("opcional, skip limpio si falta dependencia", "optional, clean skip if dependency is missing"))}</li><li>CrewAI: ${escapeHtml(s("preparado como adaptador opcional", "prepared as an optional adapter"))}</li></ul></div>`,
      },
      {
        id: "blueprints",
        number: "03",
        category: s("Blueprints", "Blueprints"),
        title: "WORKLOADS, NOT CHAT FLOWS",
        teaser: s(
          `El primer workload visible es Meta Microsite: quality=${blueprintMeta?.quality_checks?.status || "--"}, audit=${blueprintMeta?.final_audit_status || "--"}.`,
          `The first visible workload is Meta Microsite: quality=${blueprintMeta?.quality_checks?.status || "--"}, audit=${blueprintMeta?.final_audit_status || "--"}.`
        ),
        action: s("Open Meta Build", "Open Meta Build"),
        tags: ["blueprints", "meta-demo", "Inference OS"],
        summary: s(
          "HeliX empieza a empaquetarse como Blueprints: configuraciones reproducibles que definen modelos, agentes, tareas, handoffs, memoria y outputs.",
          "HeliX starts packaging itself as Blueprints: reproducible configs that define models, agents, tasks, handoffs, memory, and outputs."
        ),
        coverArt: chapterCover("blueprints + quality renderer", blueprintMeta?.quality_checks?.status || "--", "Meta Microsite"),
        content:
          `<div class="modal-inline-stats"><div><span>Quality</span><strong>${escapeHtml(blueprintMeta?.quality_checks?.status || "--")}</strong></div><div><span>Final audit</span><strong>${escapeHtml(blueprintMeta?.final_audit_status || "--")}</strong></div><div><span>Stacks</span><strong>${escapeHtml(formatNumber((stackCatalog.stacks || []).length, 0))}</strong></div></div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("What it proves", "What it proves"))}</h4><p>${escapeHtml(
            s(
              "El Meta Microsite no deja que el modelo controle el DOM. Los agentes aportan slots de contenido; HeliX renderiza una plantilla editorial curada, registra hmem, guarda .hlx y publica un build log real.",
              "The Meta Microsite does not let the model control the DOM. Agents provide content slots; HeliX renders a curated editorial template, records hmem, saves .hlx, and publishes a real build log."
            )
          )}</p></div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("Stacks catalog", "Stacks catalog"))}</h4><ul class="modal-list">${(stackCatalog.stacks || [])
            .map((item) => `<li><strong>${escapeHtml(item.name || item.id)}</strong> - ${escapeHtml(item.status || "planned")}</li>`)
            .join("")}</ul></div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("Open the artifact", "Open the artifact"))}</h4><p><a class="hero-link hero-link-accent" href="/meta-demo">site-dist/meta-demo.html</a></p></div>`,
      },
      {
        id: "transformer",
        number: "04",
        category: s("Verified GPU", "Verified GPU"),
        title: "WHAT WORKS ON TRANSFORMERS",
        teaser: s(
          `La base publica ya es fuerte: hasta ${formatRatio(bestRatio)} de compresion real de KV con match=true en la suite GPU.`,
          `The public baseline is already strong: up to ${formatRatio(bestRatio)} of real KV compression with match=true on the GPU suite.`
        ),
        action: s("See the suite", "See the suite"),
        tags: ["GPU", "match=true", "Qwen", "SmolLM"],
        summary: s(
          "Esta es la parte mas estable del proyecto: compresion real de KV en modelos Transformer, con JSONs reproducibles y resultados publicables.",
          "This is the most stable part of the project: real KV compression on Transformer models, with reproducible JSONs and publishable results."
        ),
        coverArt: chapterCover("verified GPU suite", formatRatio(bestRatio), "Transformer baseline"),
        content:
          `<div class="modal-visual">${simpleFigure("transformer", data)}</div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("Verified models", "Verified models"))}</h4><ul class="modal-list">${transformerList}</ul></div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("Claim we can make", "Claim we can make"))}</h4><p>${escapeHtml(
            s(
              `Podemos sostener hasta ${formatRatio(bestRatio)} de compresion de KV con match=true sobre la suite publica de Transformers.`,
              `We can support up to ${formatRatio(bestRatio)} of KV compression with match=true across the public Transformer suite.`
            )
          )}</p></div>`,
      },
      {
        id: "hybrid",
        number: "05",
        category: s("Hybrid frontier", "Hybrid frontier"),
        title: "WHY THE HYBRID CASE MATTERS",
        teaser: s(
          `Con Zamba2 aparece el dato clave: KV-only queda en ${formatRatio(kvRatio)}, state-only sube a ${formatRatio(stateRatio)} y combined llega a ${formatRatio(combinedRatio)}.`,
          `With Zamba2 the key fact appears: KV-only lands at ${formatRatio(kvRatio)}, state-only rises to ${formatRatio(stateRatio)}, and combined reaches ${formatRatio(combinedRatio)}.`
        ),
        action: s("Read the frontier", "Read the frontier"),
        tags: ["Zamba2", "Mamba", "hybrid"],
        summary: s(
          "Aca esta el giro mas importante del proyecto. El cuello ya no es solo KV. En los hibridos, recurrent state manda.",
          "This is the most important turn in the project. The bottleneck is no longer just KV. In hybrids, recurrent state dominates."
        ),
        coverArt: chapterCover("local Zamba2 frontier", formatRatio(combinedRatio), "Hybrid memory frontier"),
        content:
          `<div class="modal-inline-stats"><div><span>KV-only</span><strong>${escapeHtml(formatRatio(kvRatio))}</strong></div><div><span>State-only</span><strong>${escapeHtml(formatRatio(stateRatio))}</strong></div><div><span>Combined</span><strong>${escapeHtml(formatRatio(combinedRatio))}</strong></div></div>` +
          `<div class="modal-visual">${simpleFigure("hybrid", data)}</div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("The important claim", "The important claim"))}</h4><p>${escapeHtml(
            s(
              `El modo combinado llega a ${formatRatio(combinedRatio)} de reduccion total del runtime-cache y ${formatRatio(combinedSpeedup)} de speedup vs native. Esa es la semilla de una historia de hybrid memory.`,
              `The combined mode reaches ${formatRatio(combinedRatio)} total runtime-cache reduction and ${formatRatio(combinedSpeedup)} speedup versus native. That is the seed of a real hybrid memory story.`
            )
          )}</p></div>`,
      },
      {
        id: "prompts",
        number: "06",
        category: s("Prompt readout", "Prompt readout"),
        title: "WHAT THE PROMPTS SUGGEST",
        teaser: s(
          `En la mini suite local, code quedo en ${formatRatio(code)} de speedup promedio frente a ${formatRatio(daily)} para daily.`,
          `In the local mini suite, code landed at ${formatRatio(code)} average speedup versus ${formatRatio(daily)} for daily prompts.`
        ),
        action: s("Open readout", "Open readout"),
        tags: ["code", "daily", "agentic"],
        summary: s(
          "No es un benchmark universal. Si es una lectura exploratoria util para hablar de tareas de codigo o loops mas agentic.",
          "It is not a universal benchmark. It is a useful exploratory read for talking about code-heavy tasks or more agentic loops."
        ),
        coverArt: chapterCover("local prompt suite", `${formatRatio(code)} / ${formatRatio(daily)}`, "Prompt readout"),
        content:
          `<div class="modal-visual">${simpleFigure("prompts", data)}</div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("Prompt suite", "Prompt suite"))}</h4><ul class="modal-list">${promptList}</ul></div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("How to read it", "How to read it"))}</h4><p>${escapeHtml(
            s(
              "Lo contamos como una senal local: el grupo de prompts de codigo retuvo mejor speedup que el grupo daily bajo la misma compresion hibrida.",
              "We present it as a local signal: the code prompt family kept stronger speedup than the daily family under the same hybrid compression."
            )
          )}</p></div>`,
        },
        {
          id: "stress",
          number: "07",
          category: s("Stress lab", "Stress lab"),
          title: "WHAT WE CAN PROVE AFTER RESTORE",
          teaser: s(
            `Restore equivalence: hash=${String(restoreMission?.headline_metrics?.hash_match === true)}, tokens=${String(restoreMission?.headline_metrics?.generated_ids_match === true)}, delta=${formatNumber(restoreMission?.headline_metrics?.max_abs_logit_delta, 4)}.`,
            `Restore equivalence: hash=${String(restoreMission?.headline_metrics?.hash_match === true)}, tokens=${String(restoreMission?.headline_metrics?.generated_ids_match === true)}, delta=${formatNumber(restoreMission?.headline_metrics?.max_abs_logit_delta, 4)}.`
          ),
          action: s("Open stress lab", "Open stress lab"),
          tags: ["restore", "stress", "receipts", "TUI"],
          summary: s(
            "El hash prueba integridad bit-perfect del snapshot. La nueva prueba agrega una continuacion deterministica que tambien matchea antes y despues del restore.",
            "The hash proves bit-perfect snapshot integrity. The new probe adds a deterministic continuation that also matches before and after restore."
          ),
          coverArt: chapterCover("stress missions v1", `${formatNumber(stressMissions.length, 0)} missions`, "Restore equivalence"),
          content:
            `<div class="modal-inline-stats"><div><span>State hash</span><strong>${escapeHtml(String(stateMission?.headline_metrics?.hash_match === true))}</strong></div><div><span>Restore tokens</span><strong>${escapeHtml(String(restoreMission?.headline_metrics?.generated_ids_match === true))}</strong></div><div><span>Promoted blocks</span><strong>${escapeHtml(formatNumber(contextMission?.headline_metrics?.promoted_block_total, 0))}</strong></div></div>` +
            `<div class="modal-section"><h4>${escapeHtml(s("Precise claim", "Precise claim"))}</h4><p>${escapeHtml(
              s(
                "State Juggler prueba que el snapshot serializado/restaurado es identico. Restore Equivalence prueba, en un probe corto y deterministico, que la continuacion desde el estado restaurado tambien matchea.",
                "State Juggler proves the serialized/restored snapshot is identical. Restore Equivalence proves, on a short deterministic probe, that continuation from the restored state also matches."
              )
            )}</p></div>` +
            `<div class="modal-section"><h4>${escapeHtml(s("Caveat", "Caveat"))}</h4><p>${escapeHtml(
              s(
                "Esto no es todavia una evaluacion semantica amplia ni un benchmark largo. Es la base tecnica para poder hacer esas pruebas con integridad.",
                "This is not yet a broad semantic evaluation or long benchmark. It is the technical base that makes those tests integrity-checkable."
              )
            )}</p></div>`,
        },
        {
          id: "side",
          number: "08",
          category: s("Side lanes", "Side lanes"),
        title: "WHAT IS STILL BROKEN",
        teaser: s(
          "HXQ Zamba2 sigue bloqueado por logits no finitos. Gemma queda como exploracion paralela, no como claim central.",
          "HXQ Zamba2 remains blocked by non-finite logits. Gemma stays as a parallel exploration, not as a central claim."
        ),
        action: s("Check blocked lanes", "Check blocked lanes"),
        tags: ["HXQ", "Gemma", "blocked"],
        summary: s(
          "La historia mejora cuando no sobreclaimamos. Lo bloqueado sigue visible, pero no manda la lectura publica del proyecto.",
          "The story improves when we do not overclaim. Blocked lanes remain visible, but they do not drive the public reading of the project."
        ),
        coverArt: chapterCover("side lanes", hxqFinite ? "finite" : "non-finite", "HXQ + Gemma"),
        content:
          `<div class="modal-inline-stats"><div><span>HXQ Zamba2</span><strong>${escapeHtml(hxqFinite ? "finite" : "non-finite")}</strong></div><div><span>Gemma blocked</span><strong>${escapeHtml(formatNumber(data.blockedGemma, 0))}</strong></div><div><span>Gemma attempts</span><strong>${escapeHtml(formatNumber(data.gemmaModels.length, 0))}</strong></div></div>` +
          `<div class="modal-section"><h4>HXQ</h4><p>${escapeHtml(
            s(
              "El diagnostico reproducible sigue marcando logits_finite=false en EchoLabs33/zamba2-1.2b-hxq. Importa, pero hoy no sostiene un claim publico de calidad.",
              "The reproducible diagnostic still reports logits_finite=false for EchoLabs33/zamba2-1.2b-hxq. It matters, but today it does not support a public quality claim."
            )
          )}</p></div>` +
          `<div class="modal-section"><h4>Gemma</h4><p>${escapeHtml(
            s(
              "Gemma 3 quedo gated y Gemma 4 choco con limites de carga local. Es una lane abierta, no la tesis principal.",
              "Gemma 3 stayed gated and Gemma 4 hit local loading limits. It is an open lane, not the main thesis."
            )
          )}</p></div>`,
      },
        {
          id: "evidence",
          number: "08",
        category: s("Evidence", "Evidence"),
        title: "WHERE THE EVIDENCE LIVES",
        teaser: s(
          `Hoy hay ${formatNumber(data.manifest.length, 0)} artifacts visibles entre canonical evidence y lab notebook.`,
          `There are ${formatNumber(data.manifest.length, 0)} visible artifacts today across canonical evidence and the lab notebook.`
        ),
        action: s("Browse the ledger", "Browse the ledger"),
        tags: [s("canonical evidence", "canonical evidence"), s("lab notebook", "lab notebook"), "JSON"],
        summary: s(
          "El sitio no esta armado con claims vacios. La historia sale de JSONs reales bajo verification/.",
          "The site is not built from empty claims. The story comes from real JSON files under verification/."
        ),
        coverArt: chapterCover("artifact-backed public story", formatNumber(data.manifest.length, 0), "Artifact ledger"),
        content:
          `<div class="modal-visual">${simpleFigure("ledger", data)}</div>` +
          `<div class="modal-section"><h4>${escapeHtml(s("What is public", "What is public"))}</h4><ul class="modal-list"><li>${escapeHtml(s("Canonical evidence", "Canonical evidence"))}: ${escapeHtml(formatNumber(data.canonicalCount, 0))}</li><li>${escapeHtml(s("Lab notebook", "Lab notebook"))}: ${escapeHtml(formatNumber(data.labCount, 0))}</li><li>${escapeHtml(s("Abrir /research para leer reportes y navegar artifacts.", "Open /research to read reports and browse artifacts."))}</li></ul></div>`,
      },
    ];
  }

  function renderStaticCopy() {
    document.documentElement.lang = state.locale;
    byId("brand-subtitle").textContent = "Inference Session OS";
    byId("nav-index").textContent = "Index";
    byId("nav-research").textContent = "Research";
    byId("nav-app").textContent = "App";
    byId("hero-eyebrow").textContent = s("Una historia sobre sesiones de inferencia", "A story about inference sessions");
    byId("hero-subtitle").textContent = s(
      "De compresion KV a un sistema local para dormir, verificar y despertar modelos.",
      "From KV compression to a local system for sleeping, verifying, and waking models."
    );
    byId("hero-summary").textContent = s(
      "Helix empezo como compresion de memoria de inferencia. Ahora muta hacia un OS de sesiones: catalogo SQLite, snapshots .hlx, auditoria diferida, prefix reuse, checkpoints hibridos y clientes OpenAI-compatible.",
      "Helix started as inference-memory compression. It is now mutating toward a session OS: SQLite catalog, .hlx snapshots, deferred audit, prefix reuse, hybrid checkpoints, and OpenAI-compatible clients."
    );
    byId("hero-quote-title").textContent = s(
      "La frontera ya no es solo ahorrar memoria. Es administrar estados cognitivos.",
      "The frontier is no longer just saving memory. It is managing cognitive state."
    );
    byId("hero-quote-copy").textContent = s(
      "Cada agente/modelo conserva su propia memoria de trabajo comprimida. Helix la guarda, la verifica, la restaura y decide cuando conviene reutilizar un prefijo o despertar un checkpoint exacto.",
      "Each agent/model keeps its own compressed working memory. Helix saves it, verifies it, restores it, and decides when to reuse a prefix or wake an exact checkpoint."
    );
    byId("hero-link-index").textContent = s("Leer la historia", "Read the story");
    byId("hero-link-research").textContent = s("Abrir research dossier", "Open research dossier");
    byId("index-title").textContent = "INDEX OF CHAPTERS";
    byId("index-subtitle").textContent = "2025 - 2026";
    byId("footer-status-copy").textContent = s("Public static build online", "Public static build online");
    byId("footer-copy").textContent = "Helix Memory - Inference Session OS + Hybrid Memory Frontier";
    byId("modal-close").setAttribute("aria-label", isEs() ? "Cerrar" : "Close");

    document.querySelectorAll(".locale-btn").forEach((button) => {
      const active = button.dataset.locale === state.locale;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", active ? "true" : "false");
    });
  }

  function renderHero(data) {
    const prefixModel = (data.prefixReuse?.models || [])[0] || {};
    const metrics = [
      [s("Hybrid prefix", "Hybrid prefix"), formatRatio(data.hybridPrefix?.best_speedup)],
      [s("Hybrid cache", "Hybrid cache"), formatRatio(data.hybridLocal?.combined_hybrid_gain?.hybrid_total_runtime_cache_ratio_vs_native)],
      [s("Transformer prefix", "Transformer prefix"), formatRatio(prefixModel.claim_speedup_including_restore)],
      [s("Recall", "Recall"), data.memoryCatalog?.status === "completed" ? "ok" : "--"],
      [s("Layer slice", "Layer slice"), data.layerSlice?.unrelated_array_loaded === false ? "ok" : "--"],
      [s("API smoke", "API smoke"), data.openaiSmoke?.status === "completed" || data.memoryOpenai?.status === "completed" ? "ok" : "--"],
    ];
    byId("hero-metrics").innerHTML = metrics
      .map(([label, value]) => `<div class="metric-chip"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`)
      .join("");
    renderHeroVisual(data);
  }

  function renderChapterList() {
    byId("chapter-list").innerHTML = state.chapters
      .map(
        (chapter) => `
      <article class="index-row" data-chapter-id="${escapeHtml(chapter.id)}">
        <div class="index-meta">
          <span class="index-number">${escapeHtml(chapter.number)}</span>
          <span class="index-category">${escapeHtml(chapter.category)}</span>
        </div>
        <div class="index-main">
          <p class="index-tagline">${escapeHtml(chapter.action)}</p>
          <h3 class="index-title">${escapeHtml(chapter.title)}</h3>
          <p class="index-teaser">${escapeHtml(chapter.teaser)}</p>
        </div>
        <div class="index-action">
          <span class="index-action-label">${escapeHtml(s("Abrir", "Open"))}</span>
          <span class="action-orb">↗</span>
        </div>
      </article>
    `
      )
      .join("");

    document.querySelectorAll("[data-chapter-id]").forEach((node) => {
      node.addEventListener("click", () => openModal(node.getAttribute("data-chapter-id")));
    });
  }

  function openModal(chapterId) {
    const chapter = state.chapters.find((item) => item.id === chapterId);
    if (!chapter) return;
    state.activeChapterId = chapterId;
    byId("modal-kicker").textContent = `${s("Capitulo", "Chapter")} ${chapter.number}`;
    byId("modal-title").textContent = chapter.title;
    byId("modal-summary").textContent = chapter.summary;
    byId("modal-tags").innerHTML = chapter.tags.map((tag) => `<span class="modal-tag">${escapeHtml(tag)}</span>`).join("");
    byId("modal-cover-art").innerHTML = chapter.coverArt;
    byId("modal-content").innerHTML = chapter.content;
    byId("home-modal").hidden = false;
    document.body.style.overflow = "hidden";
  }

  function closeModal() {
    byId("home-modal").hidden = true;
    document.body.style.overflow = "";
  }

  function renderLoading() {
    byId("chapter-list").innerHTML =
      `<article class="index-row"><div class="index-meta"><span class="index-number">..</span><span class="index-category">Loading</span></div><div class="index-main"><h3 class="index-title">${escapeHtml(
        s("Cargando capitulos desde artifacts reales...", "Loading chapters from real artifacts...")
      )}</h3></div><div class="index-action"><span class="index-action-label">...</span></div></article>`;
  }

  function renderError(error) {
    byId("chapter-list").innerHTML =
      `<article class="index-row"><div class="index-meta"><span class="index-number">!!</span><span class="index-category">Error</span></div><div class="index-main"><h3 class="index-title">${escapeHtml(
        s(
          "No pude cargar la historia publica desde los artifacts del proyecto.",
          "I could not load the public story from the project artifacts."
        )
      )}</h3><p class="index-teaser">${escapeHtml(String(error.message || error))}</p></div><div class="index-action"><span class="index-action-label">-</span></div></article>`;
  }

  async function loadHome() {
    renderLoading();
    try {
      const manifest = await fetchManifestWithFallback();
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
        manifest,
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
      state.chapters = buildChapters(data);
      renderHero(data);
      renderChapterList();
      if (state.activeChapterId) openModal(state.activeChapterId);
    } catch (error) {
      renderError(error);
    }
  }

  function bindLocale() {
    document.querySelectorAll(".locale-btn").forEach((button) => {
      button.addEventListener("click", () => {
        state.locale = button.dataset.locale;
        window.localStorage.setItem("helix-research-locale", state.locale);
        renderStaticCopy();
        if (state.datasets) {
          const data = getData();
          state.chapters = buildChapters(data);
          renderHero(data);
          renderChapterList();
          if (state.activeChapterId) openModal(state.activeChapterId);
        }
      });
    });
  }

  function bindModal() {
    byId("modal-close").addEventListener("click", closeModal);
    byId("home-modal").addEventListener("click", (event) => {
      if (event.target === byId("home-modal")) closeModal();
    });
    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !byId("home-modal").hidden) closeModal();
    });
  }

  renderStaticCopy();
  bindLocale();
  bindModal();
  loadHome();
})();
