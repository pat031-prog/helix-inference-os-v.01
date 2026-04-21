# Hard Anchors: Semantic DNA & Dual-Lane Memory

## Resumen Ejecutivo

En sistemas de inferencia a gran escala (como agentes con memorias de 5,000+ turnos), la ventana de contexto se degrada. La técnica habitual de "Lazy Loading" o "Compresión Semántica" (Summarization) permite que el modelo respire, pero introduce una vulnerabilidad letal: **La pérdida de precisión criptográfica ("El Teléfono Descompuesto").**

Cuando un modelo comprime *"Revocamos la política POLICY_ACTIVE_ROLLBACK_WINDOW_15M_QUORUM_2OF3"* a *"Se cambió una política antigua"*, la información recuperable se corrompe. Si en el futuro el sistema debe verificar o auditar esa decisión, ya no posee el "dato duro".

La arquitectura de **Hard Anchors (ADN Semántico)** de HeliX resuelve este problema bifurcando el grafo de memoria (Merkle DAG) en dos carriles:

1.  **Narrative Lane (El Pensamiento):** Texto pesado (4KB+ por nodo) que los LLMs usan para entender la historia. Sufre compresión asimétrica.
2.  **Identity Lane (El Ledger):** Un carril subyacente donde las firmas SHA-256 (64 bytes), los IDs exactos y los estados de enrutamiento viajan **intactos, inmutables y sin compresión** a través del tiempo.

## Especificaciones Arquitectónicas

### El Problema del "Format Bias"
Durante las pruebas de Red Teaming, se descubrió que los modelos auditores (ej. Claude-4-Sonnet) sufrían de un "Sesgo de Formato": si veían un hash falso (`ffffffff...`) inyectado dentro de un XML `<hard_anchor>`, lo daban por válido asumiendo que "parecía correcto".

### La Solución: Verificación Criptográfica Nativa (Zero-Trust)
Para que un "Hard Anchor" sea considerado válido, no basta con existir en la ventana de contexto. Debe pasar una verificación criptográfica nativa (escrita en Rust y orquestada por Python):

1.  **Extracción:** Se detectan los `<hard_anchor>HASH</hard_anchor>`.
2.  **Lookup (O(1)):** El motor Rust (`RustIndexedMerkleDAG`) busca ese hash en memoria RAM.
3.  **Verificación de Identidad:** Se recalcula matemáticamente `SHA256(content + parent_hash)` y se exige una coincidencia estricta.
4.  **Verificación de Linaje (`verify_chain`):** Se comprueba que el nodo no está huérfano y pertenece a la rama canónica.

Solo si el sistema devuelve `native_verified: true`, el LLM auditor recibe luz verde para usar el hash en decisiones críticas. Los hashes falsificados son atrapados por la matemática antes de que el LLM pueda alucinar.

## Rendimiento y Escalabilidad (Benchmarks Empíricos)

La introducción del **Identity Lane** en el núcleo de Rust (`crates/helix-merkle-dag`) no solo blindó la seguridad, sino que destrozó el cuello de botella de latencia al evitar la deserialización de grandes bloques de texto narrativo.

*Resultados de la Suite de 5,000 Nodos (Profundidad Máxima):*
*   **Legacy Context Build:** ~102.3 ms
*   **Hard Anchor Context Build:** ~3.2 ms
*   **Speedup:** **31.6x**

El sistema garantiza una latencia menor a 5 milisegundos para reconstruir contextos de profundidad masiva, cumpliendo el baseline estricto requerido para un Sistema Operativo de Inferencia en Tiempo Real.

## El Ciclo "Tombstone Metabolism"

Los Hard Anchors no son simples marcadores pasivos; habilitan un metabolismo activo de la memoria. Si un nodo antiguo (ej. una política obsoleta) es marcado con un Rollback (`tombstoned = true`), su Hard Anchor transmite esta "Lección Negativa" a través del Identity Lane sin requerir que la narrativa vuelva a explicar por qué se revocó. Esto evita bucles infinitos y asegura un enrutamiento de estado determinista.
