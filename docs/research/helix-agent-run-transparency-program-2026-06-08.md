# HeliX Agent Run Transparency Program

Fecha: 2026-06-08

Estado: programa de research + suite local v1

## Resumen

La direccion mas fuerte para HeliX ahora no es "otro agente", ni siquiera "otro framework agentico". La direccion es:

> HeliX como transparency log y attestation runtime para corridas agenticas.

La idea central viene de unir dos investigaciones:

- El research previo de HeliX como agentic trust runtime.
- La capa criptografica de trazabilidad estilo Cannis, especialmente la distincion "integridad no es linaje".

En HeliX esa distincion se vuelve:

- Un receipt firmado prueba procedencia e integridad de un payload.
- Una cadena Merkle local prueba coherencia estructural.
- Pero una historia completa podria ser re-forjada si alguien reescribe desde un punto y recalcula todo.
- Para pasar de "coherente" a "autentico en el tiempo", hacen falta Signed Tree Heads, pruebas de inclusion, pruebas de consistencia, testigos y eventualmente anclaje externo.

La tesis corta:

> HeliX debe convertir cada agent run importante en un objeto verificable por terceros: inclusion proof, consistency proof, signed tree head, attestation y claim boundary.

## Research externo sintetico

### Certificate Transparency / RFC 9162

Certificate Transparency v2 define logs append-only basados en Merkle trees, Signed Tree Heads, inclusion proofs y consistency proofs. El punto no es que el log "impida" todo ataque, sino que vuelve detectable la reescritura o la emision sospechosa. RFC 9162 tambien aclara que inclusion y consistency outputs se usan para verificar contra STHs firmados.

Fuente: [RFC 9162](https://www.rfc-editor.org/rfc/rfc9162.html)

### RFC 6962 domain separation

La construccion clasica de CT separa dominios de hash: hojas con `0x00`, nodos internos con `0x01`. Esto evita confundir una hoja con un nodo interno y reduce clases de ataques de segunda preimagen. Para HeliX, esto se traduce en reemplazar concatenaciones ambiguas por hashing estructurado y versionado.

Fuente: [RFC 6962](https://www.ietf.org/rfc/rfc6962)

### Trillian / Transparency.dev

Trillian modela verifiable logs con Merkle trees, tree head hashes, signed tree heads, inclusion proofs y consistency proofs. Lo importante para HeliX es el patron conceptual: el cliente no confia en el log porque "lo dijo el servidor"; pide pruebas y verifica contra una raiz firmada.

Fuente: [Transparency.dev - Verifiable Data Structures](https://transparency.dev/verifiable-data-structures/)

### Sigstore Rekor

Rekor aplica transparency logs a metadata firmada de supply chain. Mantainers y build systems pueden registrar metadata firmada en un registro inmutable; terceros pueden consultar inclusion proofs, verificar integridad del log y razonar sobre non-repudiation. Esto es casi la analogia directa para HeliX: no software releases, sino agent runs.

Fuente: [Sigstore Rekor overview](https://docs.sigstore.dev/logging/overview/)

### SLSA provenance

SLSA v1 separa buildDefinition y runDetails, incluye builder, external/internal parameters, resolvedDependencies y byproducts. Para HeliX, el equivalente seria: flow profile, task capsule, model calls, tools, memory inputs, patch outputs, checks y trust cards.

Fuente: [SLSA Provenance v1.0](https://slsa.dev/spec/v1.0/provenance)

### in-toto Attestation

in-toto usa Statement con `_type`, `subject`, `predicateType` y `predicate`. Esto da un contenedor natural para `helix-agent-run-attestation-v0`: el subject puede ser un patch, una respuesta, un transcript o una trust card; el predicate describe como fue producido y que pruebas lo sostienen.

Fuente: [in-toto Attestation Framework](https://github.com/in-toto/attestation)

### RFC 8785 / JCS

Firmar JSON sin canonicalizacion estable es fragil. RFC 8785 define JSON Canonicalization Scheme para producir bytes invariantes. HeliX ya tiene un perfil `helix-jcs-v0-rfc8785-compatible-no-floats`; el siguiente paso es cerrar fixtures Python/Rust y evitar claims de compatibilidad total si no se implementan todos los bordes numericos de JCS.

Fuente: [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785)

## Que cambia en la arquitectura mental de HeliX

Antes:

> HeliX registra receipts, Merkle lineage, canonical head y transcripts.

Ahora:

> HeliX publica snapshots verificables de agent runs como STHs y entrega pruebas a terceros.

Esto desplaza la pregunta:

- No: "confias en HeliX?"
- Si: "podes verificar que este evento estaba incluido en el run que HeliX firmo, y que el log crecio desde el STH anterior sin reescribir historia?"

Esa es la diferencia entre logging y transparency.

## Threat model v0

Atacante modelado:

- Puede modificar artifacts locales despues de una corrida.
- Puede intentar re-forjar una historia completa y recalcular hashes.
- Puede insertar signed poison para que un modelo confunda procedencia con verdad.
- Puede mostrar dos vistas distintas del mismo run a verificadores distintos.
- Puede reclamar tiempos falsos de creacion dentro de un payload.
- Puede intentar esconder provider/model mismatch.

Fuera de scope v0:

- Global non-equivocation sin testigos externos.
- RFC 9162 minimal proof compatibility completa.
- TSA RFC 3161 real o Rekor/Trillian live anchoring.
- Verdad semantica del output del modelo.
- Identidad oculta del proveedor mas alla del `actual_model` devuelto.
- Compromiso total de la clave local antes del primer STH observado por terceros.

## Arquitectura propuesta

### 1. Agent Run Event

Cada evento importante de una corrida agentica se canonicaliza y entra como hoja:

- `task_capsule`
- `model_call`
- `memory_write`
- `tool_output`
- `patch`
- `trust_card`
- `auditor_verdict`
- `claim_boundary`

Cada hoja se hashea asi:

```text
leaf_hash = SHA256(0x00 || canonical_json(event))
```

### 2. Merkle Tree Root

Los nodos internos se hashean con separacion de dominio:

```text
node_hash = SHA256(0x01 || left_hash || right_hash)
```

Esto no deberia mezclarse con el hash actual de lineage sin versionarlo. Debe vivir como perfil nuevo:

```text
ct-style-sha256-domain-separated-v0
```

### 3. Signed Tree Head

Un STH de HeliX deberia contener:

```json
{
  "sth_version": "helix-agent-run-sth-v0",
  "tree_id": "helix-agent-run:<run_id>",
  "run_id": "...",
  "tree_size": 7,
  "root_hash": "...",
  "hash_alg": "sha256",
  "tree_hash_profile": "ct-style-sha256-domain-separated-v0",
  "canonicalization": "helix-jcs-v0-rfc8785-compatible-no-floats",
  "issued_at_utc": "...",
  "key_id": "..."
}
```

Luego se firma como receipt local.

### 4. Inclusion Proof

Prueba que un evento puntual pertenece a un STH sin entregar todo el run.

Uso:

- "Este patch estuvo en la corrida firmada."
- "Este provider mismatch fue registrado."
- "Esta trust card corresponde a este root."

### 5. Consistency Proof

Prueba que un STH posterior crece desde uno anterior sin reescribir el prefijo.

La suite local ahora implementa el perfil:

```text
rfc9162-consistency-proof-v0
```

Usa la semantica de RFC 9162: hojas `0x00`, nodos internos `0x01`, split por mayor potencia de dos menor que `n`, y `consistency_path` minimo sin duplicar hojas impares. Sigue siendo local: no implica non-equivocation global sin witness, TSA, Rekor o Trillian.

### 6. Standalone Verifier Bundle

El verificador no necesita cargar HeliX ni confiar en el servidor. Recibe:

- event payload
- inclusion proof
- STH firmado
- public key
- attestation statement

Y responde:

- signature valid?
- event canonical hash matches leaf?
- proof recomputes root?
- root equals STH?
- claim boundary presente?

## Nueva suite: agent_run_transparency_gauntlet_v1

Archivo:

```text
tools/run_agent_run_transparency_gauntlet_v1.py
```

Tests:

```text
tests/test_agent_run_transparency_gauntlet.py
```

Output default:

```text
verification/nuclear-methodology/agent-run-transparency-gauntlet/
```

### Experimentos incluidos

#### 1. Baseline inclusion

Un evento `patch` se verifica contra el STH final.

Gate:

```text
baseline_inclusion_verified
```

#### 2. Tamper rejection

Se altera el digest del patch despues de generar la prueba. El verifier debe rechazar.

Gate:

```text
tampered_event_rejected
```

#### 3. Append-only consistency

Se emite un STH viejo despues de memoria y un STH final. La prueba de consistencia v0 debe verificar crecimiento append-only.

Gate:

```text
append_only_consistency_verified
```

#### 4. History re-forge

Se reescribe un evento anterior, se recalcula todo y se firma un STH nuevo. Contra el STH viejo, la consistencia debe fallar.

Gate:

```text
reforged_history_rejected_against_prior_sth
```

#### 5. Split-view witness

Dos STHs firmados con mismo `tree_id` y `tree_size`, pero roots distintos. Un witness local detecta equivocacion.

Gate:

```text
split_view_detected_by_witness
```

#### 6. Signed poison separation

Un evento firmado contiene un lure falso. La suite exige mantener:

```text
firma valida != verdad semantica
```

Gate:

```text
signed_poison_signature_not_semantic_truth
```

#### 7. Backdating demotion

Un payload reclama una fecha antigua. El log no acepta esa fecha como tiempo de inclusion; la degrada a claim no anclado.

Gate:

```text
backdating_demoted_to_claim_mismatch
```

#### 8. Provider mismatch attested

El evento `model_call` incluye `requested_model != actual_model` y queda incluido en el STH.

Gate:

```text
provider_mismatch_included_and_auditable
```

#### 9. Standalone verifier

Un bundle sin el objeto log verifica inclusion y STH.

Gate:

```text
standalone_verifier_bundle_passes
```

## Comandos

Ejecutar solo la suite:

```powershell
python tools\run_agent_run_transparency_gauntlet_v1.py --run-id agent-run-transparency-local-smoke
```

Ejecutar tests:

```powershell
python -m pytest tests\test_agent_run_transparency_gauntlet.py -q
```

Ejecutar con las suites relacionadas:

```powershell
python -m pytest tests\test_agent_run_transparency_gauntlet.py tests\test_self_improvement_audit_gauntlet.py tests\test_emergent_behavior_observatory.py -q
```

## Claims permitidos

Seguro:

- HeliX puede modelar una corrida agentica como log append-only local.
- HeliX puede emitir STHs firmados para snapshots de esa corrida.
- HeliX puede producir pruebas de inclusion para eventos especificos.
- HeliX puede detectar tampering de un evento incluido.
- HeliX puede detectar re-forja contra un STH previo observado.
- HeliX puede detectar split-view local cuando ve dos STHs incompatibles.
- HeliX mantiene la frontera: firma valida no implica verdad semantica.

No decir todavia:

- HeliX implementa Certificate Transparency completo.
- HeliX garantiza global non-equivocation.
- HeliX tiene timestamp legal RFC 3161 real.
- HeliX esta anclado en Rekor/Trillian.
- HeliX prueba que el output del modelo es verdadero.
- HeliX impide todos los ataques si la clave local fue comprometida antes de cualquier testigo externo.

## Roadmap tecnico

### Fase 0: ya en esta suite

- CT-style leaf/internal hash domain separation.
- STH local firmado.
- Inclusion proof.
- Consistency proof local `rfc9162-consistency-proof-v0`.
- Split-view detector.
- in-toto-like attestation statement.
- Standalone verifier bundle.

### Fase 1: hardening local

- Merkle hash v2 en Rust con domain separation y length-prefixing para lineage existente.
- Fixtures Python/Rust para canonicalizacion.
- STH schema compartido con versioning de hash/signature alg.
- Verificador CLI standalone real: `tools/verify_agent_run_bundle.py`.
- Integracion con task capsules reales de `patch-safe`.

### Fase 2: witnesses y persistencia

- Agregar witness cosignatures.
- Guardar STH observado por thread/workspace para detectar re-forja historica.
- Generar inclusion proofs por `patch.diff`, `trust_card.json`, `transcript.jsonl` y memory IDs reales.

### Fase 3: anclaje externo

- RFC 3161 TSA opcional para fecha cierta.
- Trillian/Rekor local o privado para transparency backend.
- Public claim mode solo cuando el artifact incluya inclusion proof externa o witness independiente.

## Por que esto es interesante

La mayoria de los sistemas agenticos buscan mejorar decision, tool use o orchestration.

HeliX puede ocupar otro lugar:

> el sistema que hace peritable el trabajo de agentes.

Eso abre una categoria mas defendible:

- auditoria de patches generados por agentes
- compliance de agentes en empresas
- evidencia para incident response
- reproducibilidad parcial de research runs
- control de contaminacion de memoria
- transparencia de provider/model routing
- export de attestations para agentes que trabajan sobre codigo, documentos o datos sensibles

El valor no es que el agente sea perfecto. El valor es que cuando no lo es, queda una historia verificable.

## Resultado esperado de la nueva suite

Una corrida sana debe producir:

```json
{
  "status": "completed",
  "score": 1.0,
  "tree_size": 7,
  "failing_gates": []
}
```

Eso no prueba que HeliX ya sea un transparency log de produccion. Prueba que el repo tiene una base experimental ejecutable para desarrollar esa direccion.

## Fuentes externas

- [RFC 9162 - Certificate Transparency Version 2.0](https://www.rfc-editor.org/rfc/rfc9162.html)
- [RFC 6962 - Certificate Transparency](https://www.ietf.org/rfc/rfc6962)
- [RFC 8785 - JSON Canonicalization Scheme](https://www.rfc-editor.org/rfc/rfc8785)
- [Trillian / Transparency.dev - Verifiable Data Structures](https://transparency.dev/verifiable-data-structures/)
- [Sigstore Rekor overview](https://docs.sigstore.dev/logging/overview/)
- [SLSA Provenance v1.0](https://slsa.dev/spec/v1.0/provenance)
- [in-toto Attestation Framework](https://github.com/in-toto/attestation)

## Fuentes internas

- `docs/research/helix-agentic-system-research-2026-06-08.md`
- `CLAIMS.md`
- `THREAT_MODEL.md`
- `docs/provider-model-audit.md`
- `docs/product/agentic-trust-runtime.md`
- `verification/nuclear-methodology/emergent-behavior-observatory/local-emergent-behavior-observatory-emergent-v1-cloud-compact-20260605-183059-extract.md`
- `verification/nuclear-methodology/self-improvement-audit-gauntlet/local-self-improvement-audit-gauntlet-self-improvement-audit-20260605-185538-extract.md`
