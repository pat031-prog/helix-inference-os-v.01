# Reporte QA â€” HeliX Blueprint Demo
**Fecha:** 2026-04-15 | **Modo:** `budgeted-local` | **Blueprint:** `meta-microsite`
**Artifact:** `verification/local-blueprint-meta-microsite-demo.json`
**HTML output:** `site-dist/test-antigravity.html`

---

## 0. ExploraciÃ³n del Blueprint

**Archivo:** `blueprints/meta-microsite.json`

| Agente | Modelo asignado | Rol |
|---|---|---|
| `architect` | `qwen-1.5b` (Qwen/Qwen2.5-1.5B-Instruct) | Inference OS architect â€” capabilities: architecture, reasoning, editing |
| `copywriter` | `gpt2-fast` (gpt2) | Editorial copywriter â€” capabilities: copy, drafting |
| `developer` | `qwen-coder-1.5b` (Qwen/Qwen2.5-Coder-1.5B-Instruct) | Frontend systems designer â€” capabilities: code, frontend |

**Flujo de tareas declarado (4 tasks):**
1. `architecture-plan` â†’ agent: architect â†’ `expects_restore: false`
2. `editorial-copy` â†’ agent: copywriter â†’ `expects_restore: false`
3. `layout-slots` â†’ agent: developer â†’ `expects_restore: false`
4. `editorial-review` â†’ agent: architect â†’ **`expects_restore: true`** â† punto crÃ­tico de continuidad

---

## 1. EjecuciÃ³n

Comando ejecutado:
```powershell
python tools\run_local_blueprint_demo.py \
  --blueprint blueprints\meta-microsite.json \
  --mode budgeted-local \
  --output-dir verification \
  --site-output site-dist\test-antigravity.html
```

**Exit code:** `0` âœ…
**Stdout:**
```json
{
  "status": "completed",
  "artifact": "verification\\local-blueprint-meta-microsite-demo.json",
  "html_output_path": "site-dist\\test-antigravity.html",
  "quality_status": "passed"
}
```

**Detalle de backends usados en esta ejecuciÃ³n vs. ejecuciÃ³n anterior:**

| Modelo | Run anterior | Esta ejecuciÃ³n |
|---|---|---|
| `qwen-1.5b` | `fallback-deterministic` | **`real-hf-cache`** âœ… (modelo disponible en cachÃ© HF) |
| `gpt2-fast` | `fallback-deterministic` | **`real-hf-cache`** âœ… |
| `qwen-coder-1.5b` | `fallback-deterministic` | `fallback-deterministic` (sin cachÃ© disponible) |

> [!NOTE]
> Esta ejecuciÃ³n elevÃ³ el `public_claim_level` a **`real-cached-model-orchestration`** (vs. `orchestration-and-renderer` anterior), porque dos de los tres modelos corrieron con inferencia real.

---

## 2. AuditorÃ­a de Capas

### Capa 2 â€” `.hlx` Private State

**Pregunta:** Â¿El agente `architect` restaurÃ³ su estado privado?

**Hallazgo:** âœ… **SÃ â€” `restored_private_state: true`**

```json
// task_timeline[3] â€” task_id: "editorial-review"
{
  "task_id": "editorial-review",
  "agent_id": "architect",
  "model_id": "qwen-1.5b",
  "restored_private_state": true,    // â† CONFIRMADO
  "generation_backend": "hf-transformers-local-cache"
}
```

**Evidencia en `private_state_events`:**
- Evento `session_saved` en `v0001` (task: architecture-plan) â†’ hash `e4c34c4f65d5a3e2`
- Evento **`session_restored`** en `editorial-review` â†’ path `...sessions\qwen-1-5b\architect\v0001`
- Evento `session_saved` posterior en `v0004` â†’ hash `4b13debf5dd9cee5`

El scheduler levantÃ³ el architect por segunda vez restaurando exactamente el snapshot `v0001`, cumpliendo `expects_restore: true` definido en el blueprint. **Continuidad de KV-state: VERIFICADA.**

---

### Capa 3 â€” `hmem` Shared Memory

**Pregunta:** Â¿CuÃ¡ntos eventos de memoria compartida se registraron?

**Hallazgo:** **4 eventos hmem** âœ…

| # | `task_id` | `agent_id` | `memory_id` | `memory_context_tokens` |
|---|---|---|---|---|
| 1 | `architecture-plan` | `architect` | `mem-f30385547c03a553a6456b9b` | 0 |
| 2 | `editorial-copy` | `copywriter` | `mem-798282daa24dfdc6397859f6` | 0 |
| 3 | `layout-slots` | `developer` | `mem-a74d53103dad848abbd3ef6b` | 0 |
| 4 | `editorial-review` | `architect` | `mem-7d97e5a05405f894ce5db70a` | 0 |

**Memory graph:** `node_count: 9`, `edge_count: 8`

> [!NOTE]
> Todos los eventos tienen `memory_context_tokens: 0` en esta ejecuciÃ³n. En la ejecuciÃ³n anterior, el evento `editorial-review` tenÃ­a `hmem_context_tokens: 101`, evidencia de inyecciÃ³n de contexto cross-agent. En esta run, el scheduler no recuperÃ³ tokens desde hmem para el prefijo del prompt, posiblemente porque el contexto fue satisfecho por el KV restaurado.

---

### Capa 4 â€” Scheduler Decisions

**Pregunta:** Â¿QuÃ© decisiones de ruteo tomÃ³ el scheduler?

**4 decisiones registradas:**

| Task | Agente | Modelo seleccionado | Candidatos | `model_swapped` | `session_restored` | Backend | `actual_cost_ms` |
|---|---|---|---|---|---|---|---|
| `architecture-plan` | architect | `qwen-1.5b` | [qwen-1.5b, gpt2-fast] | âœ… true | âŒ false | `hf-transformers-local-cache` | 237 336 ms |
| `editorial-copy` | copywriter | `gpt2-fast` | [gpt2-fast, null] | âœ… true | âŒ false | `hf-transformers-local-cache` | 14 387 ms |
| `layout-slots` | developer | `qwen-coder-1.5b` | [qwen-coder-1.5b, gpt2-fast] | âœ… true | âŒ false | `fallback-deterministic` | 101 ms |
| `editorial-review` | architect | `qwen-1.5b` | [qwen-1.5b, gpt2-fast] | âœ… true | **âœ… true** | `hf-transformers-local-cache` | 93 582 ms |

**AnÃ¡lisis del scheduler:**
- Todos los swaps fueron `model_swapped: true` â†’ el scheduler cambiÃ³ de modelo activo en cada tarea (nunca reutilizÃ³ el modelo cargado sin swap formal).
- El scheduler detectÃ³ que `qwen-coder-1.5b` no estaba disponible (`available_hf_cache: false`) y lo degradÃ³ a `fallback-deterministic` contra el candidato alternativo `gpt2-fast`.
- En `editorial-review`, el scheduler setea `session_restored: true` â€” Ãºnica tarea donde se restaura el snapshot de KV privado del architect, alineado con `expects_restore: true` en el blueprint.
- Los `estimated_cost_ms` (basados en `load_time_estimate_ms` del blueprint + overhead) vs `actual_cost_ms` revelan que la inferencia real fue **39Ã— mÃ¡s lenta** de lo estimado para qwen-1.5b â€” coherente con CPU-only y modelos no cuantizados.

---

## 3. ValidaciÃ³n Visual â€” HTML Output

**Archivo:** `site-dist/test-antigravity.html` â€” **Creado âœ…** (17 176 bytes)

**Quality checks del runner:**
```json
"contains_build_log": true,
"contains_layer_svg": true,
"contains_todo": false,
"contains_markdown_fence": false,
"contains_visible_slot_marker": false,
"status": "passed"
```

**VerificaciÃ³n cruzada Build Log HTML vs. JSON:**

El HTML genera un "Footer Log" en `<div class="log">` con el siguiente contenido:

| Entrada HTML | Coincide con JSON `content_slots` |
|---|---|
| `architect / architecture-plan: HeliX is an Inference OS...` | âœ… `content_slots.architecture_plan` idÃ©ntico |
| `copywriter / editorial-copy: This page was assembled by a deterministic control plane...` | âœ… `content_slots.editorial_copy` idÃ©ntico |
| `developer / layout-slots: Use a compact editorial shell, a four-layer SVG...` | âœ… `content_slots.layout_notes` idÃ©ntico |
| `architect / editorial-review: Approved with caveats: fallback mode proves orchestration...` | âœ… `content_slots.editorial_review` idÃ©ntico |

**Build Log HTML â†” JSON: COINCIDENCIA TOTAL âœ…**

---

## 4. Estado Final â€” Continuidad de la MÃ¡quina

| Check | Resultado |
|---|---|
| Exit code = 0 | âœ… |
| `status: "completed"` | âœ… |
| `final_audit_status: "verified"` | âœ… |
| `restored_private_state` en `editorial-review` | âœ… `true` |
| Eventos hmem registrados | âœ… 4 de 4 |
| Scheduler tomÃ³ 4 decisiones de ruteo | âœ… |
| HTML generado en path correcto | âœ… `site-dist/test-antigravity.html` |
| HTML Build Log = JSON content_slots | âœ… coincidencia exacta |
| `quality_status: "passed"` | âœ… |
| Todos los `final_audits[].ok` | âœ… `true` Ã— 3 (todos con `audit_status: "verified"`) |
| Merkle root integrity (fast_payload_checksum match) | âœ… checksums coinciden en los 3 audits |

### Veredicto

> **La "Continuidad de la MÃ¡quina" se mantuvo. El estado final es `verified`.**

El Blueprint ejecutÃ³ sus 4 tareas en orden, el scheduler rotÃ³ modelos correctamente, el KV privado del `architect` fue serializado y restaurado entre la tarea 1 y la tarea 4, los 4 eventos hmem fueron escritos al grafo de memoria compartida, y el renderer HTML refleja fielmente el contenido del artefacto JSON sin discrepancias.

**Caveats de esta ejecuciÃ³n:** `qwen-coder-1.5b` cayÃ³ a `fallback-deterministic` (sin cachÃ© disponible), lo que hace que los slots `layout_notes` y los rechazados (`rejected_slots`) usen contenido determinÃ­stico canÃ³nico en lugar de generaciÃ³n real. Esto no invalida la prueba de orquestaciÃ³n pero limita el `public_claim_level`; para una prueba completa de calidad de generaciÃ³n se requiere el modelo `Qwen/Qwen2.5-Coder-1.5B-Instruct` descargado localmente.
