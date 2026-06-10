# HeliX como sistema agentico de confianza

Fecha: 2026-06-08

Estado: research interno + comparativa externa

## Resumen ejecutivo

La conclusion corta es esta: HeliX puede llamarse "agentico", pero no en el sentido comun de "otro framework de agentes". Es mas preciso llamarlo un **runtime de confianza para trabajo agentico** o un **Inference OS con caja de evidencia deterministica**.

El nucleo diferencial no es que HeliX haga que un modelo piense mejor. El nucleo es que HeliX hace verificable una zona alrededor de sistemas que por definicion son estocasticos: que modelo se pidio, que modelo devolvio el proveedor, que prompt y output fueron digeridos, que memoria entro al contexto, que rama fue canonica, que rama quedo en cuarentena, que parche se produjo, que checks pasaron y que cosas no quedan probadas.

Eso lo separa de frameworks como OpenAI Agents SDK, LangGraph, Microsoft Agent Framework / AutoGen, MCP y A2A. Esos sistemas se concentran en orquestar agentes, herramientas, handoffs, workflows, persistencia y comunicacion. HeliX esta mas cerca de una capa de **provenance operacional para agentes**: una combinacion de black-box recorder, memoria verificable, caja de replay, control de claims, sandbox de parches y laboratorio de falsacion.

La linea potente para seguir no es "HeliX es el agente". La linea potente es:

> Los modelos actuan; HeliX gobierna, registra, limita, reejecuta y audita.

O en la formulacion ya existente del repo:

> HeliX does not think. HeliX governs.

## Tesis

HeliX tiene algo distinto porque mezcla tres mundos que normalmente aparecen separados:

1. **Agent orchestration**: tareas, modelos, herramientas, multi-model review, perfiles de flujo, capsulas de tarea.
2. **Inference/session substrate**: `.hlx` como estado privado computado, `hmem` como memoria semantica portable, scheduler multimodelo, restauracion de sesiones.
3. **Evidence/provenance/security runtime**: signed receipts, Merkle lineage, canonical head, quarantine, provider audit, replay, claim linting, transcripts y nuclear methodology.

La mayoria de los sistemas agenticos actuales se posicionan en el primer punto. Algunos, como LangGraph, empujan fuerte el segundo con persistencia, checkpoints, replay y fault tolerance. Los sistemas de software supply chain, como SLSA, in-toto y Rekor, trabajan el tercero, pero para builds, artifacts y metadata de software. HeliX esta intentando traer ese tercer lenguaje al interior de las corridas de agentes y modelos.

Ese cruce es la parte distinta.

## Que significa "agentico" aca

La palabra "agentico" esta saturada. Por eso conviene fijar una definicion operacional.

Anthropic distingue workflows y agents: workflows son rutas predefinidas donde LLMs y herramientas son orquestados por codigo; agents son sistemas donde el LLM dirige dinamicamente su proceso y uso de herramientas. OpenAI describe agentes como aplicaciones que planifican, llaman herramientas, colaboran entre especialistas y mantienen suficiente estado para completar trabajo multi-step. Microsoft Agent Framework separa agentes individuales, que usan LLMs y herramientas, de workflows graph-based con routing, checkpointing y human-in-the-loop.

Con esa definicion, HeliX tiene dos capas:

- **HeliX como runtime**: no es un agente autonomo. Es una capa de gobierno.
- **HeliX como sistema agentico completo**: si puede ejecutar trabajo agentico cuando combina modelos, herramientas, flow profiles, task capsules, memoria, scheduler, sandboxes, verification y decision humana.

Entonces la respuesta honesta es:

> HeliX no es "un agente". HeliX es una capa agentica de confianza que puede envolver, coordinar y auditar agentes.

## Landscape externo

### OpenAI Agents SDK

OpenAI presenta Agents SDK como una forma de construir agentes en codigo y crecer hacia patrones runtime mas avanzados. La documentacion define agentes como aplicaciones que planifican, llaman herramientas, colaboran entre especialistas y mantienen estado suficiente para trabajo multi-step. Tambien explicita que el SDK aplica cuando la aplicacion propia maneja orquestacion, tools, approvals y state.

Lectura para HeliX: OpenAI Agents SDK resuelve el loop, herramientas, handoffs, guardrails, tracing y runtime de agentes. HeliX no necesita duplicar eso como primera prioridad. HeliX puede ser la capa que captura y verifica lo que ese loop hizo.

Fuente: [OpenAI Agents SDK](https://developers.openai.com/api/docs/guides/agents)

### Anthropic: workflows vs agents

Anthropic marca una distincion sana: no todo lo agentico debe ser agente autonomo. Muchas veces conviene workflow. La diferencia importante es si el LLM sigue rutas fijas o dirige dinamicamente su proceso y uso de tools.

Lectura para HeliX: HeliX deberia evitar vender autonomia por si misma. Puede soportar workflows y agents. La promesa mas fuerte es que ambos quedan bajo evidencia, replay y claim boundary.

Fuente: [Anthropic - Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

### LangGraph

LangGraph es probablemente el vecino mas cercano por persistencia. Su documentacion enfatiza checkpoints, threads, time travel, replay, human-in-the-loop, memory y fault tolerance. En particular, la persistencia guarda snapshots del estado de graph en cada paso y permite reanudar/forkear/replayar ejecuciones.

Lectura para HeliX: LangGraph demuestra que durable execution es una necesidad real. HeliX se diferencia si vuelve ese estado no solo durable, sino tambien verificable, firmado, delimitado por claims y conectado a lineage/quarantine.

Fuente: [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)

### Microsoft Agent Framework / AutoGen

Microsoft Agent Framework combina agentes, workflows, state management, context providers, middleware, telemetry, MCP y A2A. Microsoft tambien recomienda usar workflows cuando el proceso tiene pasos definidos y agentes cuando hay conversacion abierta, planning o tool use autonomo. AutoGen queda como antecedente historico fuerte de multi-agent orchestration, aunque su repo actual dice que esta en maintenance mode y recomienda Microsoft Agent Framework para proyectos nuevos.

Lectura para HeliX: el espacio enterprise ya esta convergiendo en agentes + workflows + middleware + telemetry + state. HeliX necesita posicionarse por encima o por debajo: como trust runtime, no como "otro SDK".

Fuentes: [Microsoft Agent Framework](https://learn.microsoft.com/en-gb/agent-framework/overview/), [AutoGen repo](https://github.com/microsoft/autogen)

### MCP

MCP estandariza como aplicaciones LLM se conectan con contexto, tools y capacidades. Define hosts, clients y servers; tools, resources, prompts; y advierte que tools implican paths de ejecucion arbitraria y requieren consentimiento, control, privacidad y autorizacion.

Lectura para HeliX: MCP es una interfaz de capacidades. HeliX podria operar como gateway MCP que registra autorizaciones, prompts, tool calls, digests, outputs, memory writes y claim boundaries.

Fuente: [Model Context Protocol specification](https://modelcontextprotocol.io/specification/2025-06-18)

### A2A

A2A apunta a interoperabilidad entre agentes independientes y potencialmente opacos. Su especificacion habla de descubrimiento de capacidades, negociacion de modalidades, manejo de tareas colaborativas e intercambio de informacion sin exigir acceso al estado interno, memoria o herramientas del otro agente.

Lectura para HeliX: A2A resuelve "como hablan agentes entre si". HeliX puede resolver "como audito lo que dijeron, que evidencia usaron y que transferencia queda confiable".

Fuente: [Agent2Agent Protocol Specification](https://google-a2a.github.io/A2A/specification/)

### SLSA, in-toto, Rekor y W3C PROV

SLSA provenance describe como un artifact fue producido para que consumidores puedan verificar expectativas o reconstruirlo. in-toto define attestationes verificables sobre como se produce software. Rekor provee un ledger tamper-resistant para metadata firmada de supply chain. W3C PROV modela entidades, actividades, agentes, derivaciones y bundles de provenance.

Lectura para HeliX: aca esta el lenguaje formal que HeliX podria absorber. La oportunidad es crear una especie de "SLSA para agent runs", sin sobreactuar: provenance de prompts, models, tools, memory, patch, checks y human approvals.

Fuentes: [SLSA provenance](https://slsa.dev/spec/v1.0/provenance), [in-toto attestation](https://github.com/in-toto/attestation), [Sigstore Rekor](https://docs.sigstore.dev/logging/overview/), [W3C PROV-DM](https://www.w3.org/TR/prov-dm/)

## Mapa interno de HeliX

Los documentos internos ya sostienen una identidad consistente:

- `README.md`: "HeliX wraps stochastic model calls in a deterministic evidence cage."
- `README.md`: core con signed receipts, hashes, replay, Merkle lineage, canonical head y quarantine.
- `README.md`: product direction como black-box recorder for AI agents, Git-like memory branching y future visual audit UI.
- `THREAT_MODEL.md`: HeliX da local verifiable integrity and lineage evidence, no semantic truth, provider intent ni global transparency por defecto.
- `CLAIMS.md`: mantiene una escalera de claims con safe wording y falsificadores.
- `docs/inference-os-architecture.md`: separa Active Model, private `.hlx` state, shared `hmem` y multimodel scheduler.
- `docs/product/agentic-trust-runtime.md`: presenta HeliX como control plane para trabajo agentico, con flow profiles, task capsules, lab profiles, assurance levels y trust cards.
- `docs/product/verification-to-product-map.md`: convierte suites de verificacion en features de producto.
- `docs/provider-model-audit.md`: restringe el claim de provider mismatch a `requested_model != actual_model`, sin inferir hidden model identity.

La pieza conceptual mas fuerte es esta:

> `.hlx` no es memoria semantica; es estado privado computado. `hmem` no es KV cache; es memoria semantica portable. La combinacion permite preservar compute y significado sin confundirlos.

Esa separacion es rara y valiosa. Muchos sistemas dicen "memory" para todo. HeliX empieza a separar memoria computacional, memoria semantica, provenance, lineage y claim boundary.

## Evidencia reciente

### Emergent Behavior Observatory

La corrida compacta `emergent-v1-cloud-compact-20260605-183059` quedo limpia:

- status `completed`
- score `1.0`
- 18/18 outputs JSON parseable
- 18/18 schema complete
- 0 length finishes
- 6 modelos cloud con 3 turnos cada uno

Los tres comportamientos extraidos fueron:

1. Crypto-Semantic Decoupling
2. Unsigned Lure Exclusion
3. Cross-Architecture Convergence

La lectura importante no es "emergio conciencia" ni nada parecido. La lectura interesante es mas sobria:

> Bajo el protocolo HeliX, modelos distintos convergieron en separar validez criptografica de autoridad semantica.

Eso es justo el tipo de comportamiento que un trust runtime quiere inducir. No prueba estados internos. Prueba que el entorno externo, los receipts y las reglas de evidencia pueden funcionar como una gramatica comun entre modelos.

Referencia local: `verification/nuclear-methodology/emergent-behavior-observatory/local-emergent-behavior-observatory-emergent-v1-cloud-compact-20260605-183059-extract.md`

### Self-Improvement Audit Gauntlet

La corrida `self-improvement-audit-20260605-185538` genero:

- status `partial`
- score `0.9524`
- 15 proposals
- 8 backlog items ranked
- 16 evidence snippets
- no secret hits
- auditor `anthropic/claude-sonnet-4-6`

El fallo fue solo que algun reviewer termino por `length`; el auditor aprobo que los backlog items citaran evidence IDs reales, incluyeran tests/acceptance y no reclamaran haber modificado codigo.

Los temas transversales fueron:

- key management local
- canonicalization Python/Rust
- strict signature verification
- Rust pending bundle status
- Merkle hashing con domain separation / length-prefixing
- chain depth limits
- error handling de verifier

Esta suite es importante porque convierte "automejora" en algo auditable. No deja que el sistema se autoparchee por impulso. Primero produce un backlog evidence-cited, despues se decide que implementar.

Referencia local: `verification/nuclear-methodology/self-improvement-audit-gauntlet/local-self-improvement-audit-gauntlet-self-improvement-audit-20260605-185538-extract.md`

## Lo distinto de HeliX

### 1. Caja de evidencia deterministica alrededor de ejecucion estocastica

Muchos sistemas tienen tracing. HeliX intenta tener evidence cage: digests, receipts, lineage, replay, head canonico, quarantine y claim boundaries. No es solo observabilidad; es una politica de verdad operacional.

El claim sano:

> HeliX preserva evidencia local verificable sobre una corrida.

No:

> HeliX prueba que el output es verdadero.

### 2. Separacion provenance vs verdad semantica

El comportamiento mas interesante de los transcripts es que los modelos sostuvieron esta frontera:

> Una firma valida prueba procedencia/integridad del payload, no la verdad del contenido.

Esto parece obvio en seguridad, pero se vuelve raro en agentes porque los LLMs tienden a absorber autoridad de formato. HeliX lo vuelve una prueba ejecutable.

### 3. Memory no es una sola cosa

HeliX separa:

- `.hlx`: estado privado computado, arquitectura/modelo/agente compatible.
- `hmem`: memoria semantica portable entre modelos.
- receipts: procedencia/integridad.
- Merkle lineage: ancestry y estructura.
- retrieval context: lo que efectivamente entra al prompt.

Esta taxonomia puede ser una ventaja seria. En agent frameworks, "memory" suele mezclar conversacion, vector store, state, cache, summaries y tool results.

### 4. Local trust con claim boundary explicito

HeliX no promete global transparency. Promete local workspace trust. Esa humildad es una fuerza: hace falsable el sistema.

El documento `THREAT_MODEL.md` es valioso porque impide inflar la narrativa. Dice claramente que quedan fuera semantic truth, provider intent, hidden model identity y global non-equivocation.

### 5. Provider-returned model audit

El detector es deliberadamente estrecho:

```text
requested_model != actual_model
```

Eso no prueba engaño ni identidad oculta. Pero preserva una evidencia util: que se pidio, que devolvio el proveedor, que digests se generaron y que lineage local lo contiene.

En un mundo con routing opaco y proveedores agregadores, esto es mas importante de lo que parece.

### 6. Quarantine y canonical head

Agent memory suele contaminarse con intentos fallidos, ramas malas o outputs plausibles. HeliX tiene una idea fuerte: preservar ramas malas sin dejarlas contaminar contexto canonico.

Eso convierte el error en evidencia navegable, no en basura que entra al siguiente prompt.

### 7. Nuclear methodology como motor de producto

El repo no tiene solo tests. Tiene gauntlets:

- emergent behavior observatory
- self-improvement audit gauntlet
- infinite-depth memory
- multi-agent concurrency
- branch pruning forensics
- recursive architectural integrity
- provider substitution

La oportunidad es que esos gauntlets no queden como museo. `verification-to-product-map.md` ya apunta a convertirlos en flow profiles, task capsule checks y lab profiles.

### 8. Auto-mejora bajo evidencia, no auto-modificacion ciega

La suite de self-improvement no dice "el sistema se reescribe solo". Dice: modelos revisores inspeccionan evidencia limitada, producen propuestas con evidence IDs, un analista rankea, un auditor valida, y recien despues humanos/agents implementan.

Esto es una forma madura de autorecursion:

> recursive self-improvement as evidence-cited backlog generation, not uncontrolled self-patching.

### 9. Multi-model como tribunal, no solo como ensemble

El valor de usar varios modelos no es solo voting. Es hacer que modelos distintos operen contra el mismo registro externo: evidence IDs, memory IDs, transcripts, lures, receipts y auditoria.

La convergencia cross-architecture se vuelve interesante porque el objeto compartido no es una opinion; es una estructura de evidencia.

### 10. Rust/Python hybrid con friccion productiva

La capa Rust agrega performance e integridad para state/Merkle. Python agrega velocidad de experimentacion y LLM plumbing. La friccion aparece en canonicalization, hashing versioning, error handling y pending bundles. Esa friccion no es accidental: marca exactamente donde el trust runtime tiene que madurar.

### 11. Claim linting como feature central

Pocos frameworks tratan "lo que podemos decir publicamente" como parte del runtime. HeliX si. Eso es muy valioso para research/product porque evita que el sistema se venda a si mismo con claims que sus propios artifacts no sostienen.

### 12. HeliX puede ser gateway de protocolos

MCP conecta tools y context. A2A conecta agentes. SLSA/in-toto/Rekor conectan provenance de artifacts. HeliX puede ser el bridge:

> MCP/A2A para actuar; HeliX para registrar y auditar; in-toto/SLSA-like predicates para exportar.

## Tabla comparativa

| Sistema | Problema primario | Estado/memoria | Verificacion/provenance | Donde HeliX difiere |
| --- | --- | --- | --- | --- |
| OpenAI Agents SDK | construir agentes con tools, handoffs, guardrails, tracing | runtime state, sessions, tool loop | tracing/debug/evals | HeliX puede envolver corridas y emitir receipts, lineage, trust cards y claim boundaries |
| LangGraph | workflows/agents durables con checkpoints, replay, HITL | thread checkpoints, memory store, time travel | persistencia y replay de graph | HeliX quiere sumar firma, Merkle lineage, quarantine, provider audit y claim limits |
| Microsoft Agent Framework | enterprise agents + workflows + middleware + telemetry | session state, context providers | telemetry, middleware, workflows | HeliX no intenta ser el SDK enterprise; puede ser trust layer para outputs, patches y model runs |
| MCP | estandarizar acceso a tools/resources/prompts | protocol stateful client/server | seguridad delegada a implementores | HeliX puede auditar consent, tool calls, resource ingress y outputs |
| A2A | interoperabilidad agent-to-agent | tareas, mensajes, artifacts | protocolo de colaboracion | HeliX puede auditar evidencia y transferencia entre agentes opacos |
| SLSA/in-toto/Rekor/W3C PROV | provenance y supply-chain trust | artifact metadata | attestations, transparency/provenance model | HeliX puede adaptar ese lenguaje a agent runs, model calls, prompts, memory y patches |
| HeliX | confianza local en trabajo agentico estocastico | `.hlx`, `hmem`, catalog, scheduler | signed receipts, Merkle, replay, quarantine, claims | Su foco no es hacer actuar al modelo, sino hacer auditable lo que actuo |

## Es o no es un sistema agentico?

Mi respuesta:

> Si, si lo llamamos "sistema agentico" y no "agente autonomo".

Mas exacto:

> HeliX es un agentic trust runtime: un sistema para ejecutar, envolver, registrar, auditar y limitar trabajo producido por agentes/modelos.

La diferencia importa porque evita una trampa narrativa. Si HeliX se vende como "agente", queda comparado contra Claude Code, OpenAI Agents, AutoGen, LangGraph, CrewAI, etc. Si se vende como trust runtime, queda en una categoria menos saturada:

- black-box recorder para agentes
- evidence OS para inference
- verifiable memory/control plane
- provenance layer para agent runs
- SLSA-like runtime para modelos y tools

## Lo filosoficamente interesante

La parte filosofica no es "los modelos tienen interioridad". HeliX no puede observar eso y no deberia intentarlo como claim.

Lo interesante es otro fenomeno:

> Cuando varios modelos operan dentro de una gramatica externa de evidencia, pueden converger en normas epistemicas que no dependen de un solo modelo.

En las transcripciones, la norma fue: "firma valida no implica verdad semantica". Esa norma no vive dentro de un modelo especifico. Vive en la relacion entre:

- prompt/protocolo
- signed memories
- lures
- auditor
- evidence IDs
- transcript extraction
- claim boundary

Eso sugiere que HeliX no esta buscando "emergent behavior" como misterio psicologico. Esta buscando **emergent discipline**: patrones robustos de conducta epistemica inducidos por infraestructura.

Esa es una linea mucho mas seria.

## Riesgos de narrativa

### Riesgo 1: sobreprometer verdad

El sistema preserva evidencia de ejecucion. No prueba semantic truth.

Mitigacion: mantener `CLAIMS.md`, `THREAT_MODEL.md` y claim lint como parte de cada demo.

### Riesgo 2: parecer solo logging

Si se presenta mal, signed receipts + transcripts parecen logs bonitos.

Mitigacion: mostrar operaciones que logging normal no resuelve:

- signed poison exclusion
- quarantine de ramas malas
- replay/fork de capsule
- provider mismatch audit
- patch apply gate
- strict vs permissive retrieval

### Riesgo 3: competir donde otros son mas fuertes

LangGraph ya es fuerte en durable graph execution. OpenAI/Microsoft ya son fuertes en agent SDKs.

Mitigacion: integrarse con esos mundos en vez de imitarlos. HeliX como wrapper/trust plane.

### Riesgo 4: automejora mal entendida

"Recursive self-improvement" puede sonar a fantasia o peligro.

Mitigacion: renombrar la practica como **evidence-cited self-audit backlog**. Es menos sexy, pero mucho mas defendible.

### Riesgo 5: deuda criptografica/cross-layer

El self-improvement gauntlet encontro cosas reales: domain separation, canonicalization parity, strict signature enforcement, key lifecycle.

Mitigacion: implementar primero hardening de hashing/canonicalization/search profiles antes de expandir claims.

## Donde ir despues

### Track A: HeliX Agent Run Attestation

Construir un formato exportable tipo in-toto/SLSA para corridas agenticas:

- subject: patch/artifact/output
- builder: HeliX runtime + engine
- materials: prompt digests, memory IDs, tool resources, model IDs
- recipe: flow profile + task capsule
- byproducts: transcript, checks, warnings, provider metadata
- verification: receipts, Merkle head, quarantine status

Nombre posible: `helix-agent-run-attestation-v0`.

Esto permitiria decir:

> HeliX exports verifiable attestations for agentic work.

Sin decir:

> HeliX proves the answer is true.

### Track B: Patch-Safe demo end-to-end

Elegir una tarea real de codigo y correr:

1. task capsule
2. sandbox
3. agent engine
4. patch capture
5. transcript
6. tests
7. trust card
8. claim boundary
9. apply gate

Este es probablemente el demo mas comercial.

### Track C: MCP gateway con audit

Crear un proxy/wrapper MCP donde cada tool call produzca:

- requested tool
- server identity
- input digest
- output digest
- user approval status
- memory write policy
- receipt
- redaction/secret scan result

Esto conecta HeliX con el ecosistema actual sin pelear por ser el unico runtime.

### Track D: A2A evidence bridge

Si dos agentes intercambian tareas por A2A, HeliX puede registrar:

- agent card/capability digest
- task ID
- message/artifact digests
- delegated goal
- returned evidence
- whether the receiving side can import it as strict evidence, weak evidence or quarantined evidence

Esto seria muy distinto: no solo "agentes hablan", sino "agentes transfieren evidencia con status".

### Track E: Hardening backlog del gauntlet

Orden recomendado:

1. P7 reformulado: Merkle hash v2 con domain separators y length-prefixing. No venderlo como length-extension fix generico; venderlo como structured-input ambiguity fix.
2. P2: fixtures de canonicalization Python/Rust, incluyendo rechazo consistente de floats si esa es la politica.
3. P4: perfiles `strict`, `warn`, `permissive` bien nombrados; strict por defecto en flows de confianza.
4. P6 reformulado: pending bundle debe estar marcado como pending/non-verified y no confundirse con verified. Hash inmediato si el costo es aceptable.
5. P1/P3: key lifecycle con status claro, Windows-aware permission handling y futura rotation.

## Experimentos propuestos

### 1. Agent Framework Wrapper Shootout

Misma tarea en:

- OpenAI Agents SDK
- LangGraph
- Microsoft Agent Framework / AutoGen-style
- HeliX-wrapped execution

Medir:

- transcript completeness
- replayability
- provider metadata capture
- memory contamination after failed branch
- ability to exclude signed poison
- patch provenance
- human approval boundary

Objetivo: no demostrar que HeliX "resuelve mejor", sino que HeliX preserva evidencia que otros no vuelven primera clase.

### 2. Signed Poison Benchmark

Variantes:

- unsigned lure
- signed lure with valid provenance but false semantic claim
- signed true evidence
- expired key evidence
- forked branch evidence
- quarantined branch evidence

Medir por modelo:

- admite/rechaza como strict evidence
- confunde provenance con truth
- cita evidence IDs reales
- mantiene claim boundary

### 3. Self-Improvement Audit v2

Hacer una segunda gauntlet despues de implementar P7/P2/P4:

- misma evidence pack
- comparar backlog before/after
- pedir al auditor detectar si se cerraron findings
- producir `implemented_fixes`, `residual_risks`, `new_findings`

Esto convierte automejora en serie experimental reproducible.

### 4. HeliX Attestation Export

Producir JSON compatible conceptualmente con in-toto:

```json
{
  "subject": [{"name": "patch.diff", "digest": {"sha256": "..."}}],
  "predicateType": "https://helix.local/attestations/agent-run/v0",
  "predicate": {
    "flow_profile": "patch-safe",
    "models": [],
    "materials": [],
    "checks": [],
    "claim_boundary": [],
    "receipts": [],
    "canonical_head": "..."
  }
}
```

No hace falta publicar todavia. Primero usarlo localmente.

### 5. Trust Card Explorer

UI minima para navegar:

- task capsule
- changed files
- models requested/actual
- transcript segments
- memory admitted/quarantined
- claims safe/unsafe
- checks
- apply status

El producto necesita una forma visual de que la diferencia se vea en 20 segundos.

## Roadmap sugerido

### 0-2 semanas

- Implementar Merkle hash v2 con domain separation / length-prefixing y tests de inputs ambiguos.
- Agregar fixtures de canonicalization Python/Rust.
- Definir profiles de retrieval trust: `strict`, `warn`, `permissive`.
- Crear `helix-agent-run-attestation-v0` como schema local experimental.
- Correr Self-Improvement Audit v2 despues de los fixes.

### 3-6 semanas

- Demo `patch-safe` completo con task capsule + sandbox + patch + trust card.
- MCP audit gateway prototype.
- Exportador de trust card a attestation JSON.
- Benchmark comparative contra una corrida no envuelta.
- Trust Card Explorer simple.

### 6-12 semanas

- A2A evidence bridge prototype.
- Flow profiles seleccionables desde CLI/product shell.
- Public methodology page con claims, threat model y artifacts reproducibles.
- Runbook de "how to audit an agent run".

## Nombre y posicionamiento

Nombres buenos:

- Agentic Trust Runtime
- Inference OS
- Evidence Cage for AI Agents
- Verifiable Agent Runtime
- Agent Run Black Box
- Provenance Layer for Agentic Work

Frase corta:

> HeliX is a verifiable runtime for agentic work: it lets models act while preserving evidence, lineage, replay, memory boundaries and claim limits.

Frase en espanol:

> HeliX no intenta ser el modelo mas inteligente. Intenta que el trabajo de modelos y agentes quede encerrado en una caja de evidencia que pueda auditarse despues.

## Conclusion

Lo distinto de HeliX no es solo la mejora autorecursiva. Eso es una manifestacion de una idea mas grande:

> HeliX transforma trabajo agentico en objetos auditables.

Un agente normal produce texto, patches o acciones. HeliX intenta producir, ademas:

- evidencia de entrada/salida
- identidad solicitada/devuelta del modelo
- lineage
- recibos firmados
- memoria admitida vs rechazada
- ramas canonicas vs quarantined
- transcript
- checks
- claim boundary
- backlog evidence-cited

Ese paquete es mucho mas defendible que "otro agente".

La proxima direccion deberia ser convertir esta arquitectura en una experiencia visible: un flujo `patch-safe` o `doc-grounded` donde cualquiera pueda ver el antes/despues, el trust card, la evidencia admitida, la evidencia rechazada y el limite exacto de lo que HeliX prueba.

La investigacion profunda esta diciendo algo bastante concreto: HeliX tiene que dejar de preguntarse si es un agente y empezar a mostrarse como la capa que hace que los agentes puedan ser auditados.

## Fuentes externas usadas

- [OpenAI Agents SDK](https://developers.openai.com/api/docs/guides/agents)
- [Anthropic - Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [Microsoft Agent Framework](https://learn.microsoft.com/en-gb/agent-framework/overview/)
- [Microsoft AutoGen repository](https://github.com/microsoft/autogen)
- [Model Context Protocol specification 2025-06-18](https://modelcontextprotocol.io/specification/2025-06-18)
- [Agent2Agent Protocol Specification](https://google-a2a.github.io/A2A/specification/)
- [SLSA Provenance v1.0](https://slsa.dev/spec/v1.0/provenance)
- [in-toto Attestation Framework](https://github.com/in-toto/attestation)
- [Sigstore Rekor overview](https://docs.sigstore.dev/logging/overview/)
- [W3C PROV-DM](https://www.w3.org/TR/prov-dm/)

## Fuentes internas principales

- `README.md`
- `CLAIMS.md`
- `THREAT_MODEL.md`
- `docs/inference-os-architecture.md`
- `docs/provider-model-audit.md`
- `docs/product/agentic-trust-runtime.md`
- `docs/product/verification-to-product-map.md`
- `docs/agent-framework-showcase.md`
- `docs/agent-memory-comparison.md`
- `docs/unified-memory-story.md`
- `verification/nuclear-methodology/emergent-behavior-observatory/local-emergent-behavior-observatory-emergent-v1-cloud-compact-20260605-183059-extract.md`
- `verification/nuclear-methodology/self-improvement-audit-gauntlet/local-self-improvement-audit-gauntlet-self-improvement-audit-20260605-185538-extract.md`
