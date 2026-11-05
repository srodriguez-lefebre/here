# Estado de here

Actualizado: 2026-11-05. Leé CLAUDE.md primero. Historial y rulings:
`docs/development/DELIVERY.md`.

## Objetivo vigente

Implementar y entregar M2–M5 completos por instrucción explícita del usuario
(ruling 20, reemplaza ruling 15). Conservar definiciones de terminado y los tres
niveles de evidencia. M5 exige distribución Windows 1.0.0 y aceptación real.

## Código y secuencia

- Base de trabajo: `878e7d4`, v0.2.0, PR #19 mergeado. `main` incorporó PR #20
  en `763a891`, integrado en esta rama con los contratos del ruling 19.
- Grupo actual: `codex/m2-catalog`, worktree
  `C:/Users/savar/.codex/worktrees/m2-catalog/here`.
- Orden: M2 A (`M2_CATALOG`, 1–3), M2 B (`M2_LIFECYCLE`, 4–5),
  M3 A (`M3_SEARCH`, 1–2), M3 B (`M3_LIBRARY`, 3–4),
  M4 A (`M4_GROUNDED`, 1–3), M4 B (`M4_PRODUCT`, 4–5),
  M5 A (`M5_SETTINGS`, 1–2), M5 B (`M5_PORTABILITY`, 3),
  M5 C (`M5_RELEASE`, 4–5). Versiones: 0.3.0, 0.4.0, 0.5.0, 1.0.0.
- Solo solapamientos de CLAUDE §11; migraciones, pyproject y versión en serie.
- Cada tarea: RED por subagente → GREEN → review independiente → docs/evidencia
  por otro subagente. Cada grupo: PR con evidencia/review → Copilot recibido y
  respondido sobre head actual → CI Windows verde → merge commit con
  `--admin --match-head-commit` → ledger. Ningún gate se omite.

## Tarea en curso

`docs/development/M2_CATALOG_PLAN.md`, tarea 3: ingestión, identidad y revisiones
con candidato en el worktree de Green y siete hallazgos de review independiente
abiertos. La suite estable del candidato pasó 883 tests/9 skips en 598,56 s,
pero no aprueba los contratos cuestionados. Las tareas 1 y 2 tienen implementación
y review independiente aceptadas; los contratos de tarea 2 ya están integrados.
SPEC: `docs/development/M2_CATALOG_SPEC.md`. Hito: `docs/MILESTONE_2_PLAN.md`.
Tarea 1: implementación local y review independiente aceptadas, incluida la corrección de
`Busy` en lecturas bloqueadas. Foco: 34 pasan/1 skip en 4,70 s. Corrida final
`M2A-T1-FINAL-20261105`: 778 pasan/7 skips en 173,68 s (CPython 3.13.15, Qt
6.11.2, offscreen, key sintética y wheel no editable aislado). Ambos smokes de
wheel pasaron; los siete skips requieren privilegio de symlink. Código de tarea
en `9a80060`; integración del grupo y CI de Windows sobre el head del PR siguen
pendientes.

Tarea 2: fuente `037fc03`, integrada como `1cdaadb`; review independiente de SPEC
y calidad PASS sin hallazgos. Foco: 310 pasan/1 skip en 24,60 s; captura/recuperación
con kills reales: 68 pasan en 25,64 s. Corrida `M2A-T2-FINAL-20261105`: 783 pasan/
8 skips en 160,71 s. Sus ocho skips son seis de symlink y dos smokes de wheel sin
entorno no editable configurado; no sustituye la corrida de wheel de tarea 1 ni CI.
Tarea 3: RED de servicio/artefactos/controlador 31 fallos/18 pasan en 2,50 s,
confirmado por root en 4,25 s. RED conjunto con CLI: 39 fallos/18 pasan en 7,33 s.
Focos GREEN incluyen 85 casos de propiedad/procesamiento/recovery/CLI/controlador
en 79,17 s, cinco de retry legado/procesamiento en 14,10 s y dos de UTC en 1,83 s.
Son corridas de alcances y fuentes intermedias, no una suite final ni review aceptada.
El snapshot de 27 archivos coincide exactamente con el candidato local fijado por
los commits `efb73a9` (fixture) y `6c422f0` (Task3), todavía sin integrar en root.
Es FIX_BASE para corregir, no aceptación. Sus guards de roles/estados y limpieza
desconocida están en el snapshot verificado; la observación anterior de guards
ausentes pertenece a fuente supersedida. La review pide siete correcciones, incluida
la asociación explícita revisión/audio autorizada por ruling 31.
Worktree `C:/Users/savar/.codex/worktrees/m2-ingestion/here`, con ownership separado
de los archivos compartidos. Preparación de M2 B y fixtures
sintéticos disponibles. No hay cierre de M2 A ni de M2, ni certificación nueva de
hardware, proveedor o instalación limpia.

## Decisiones abiertas y bloqueos

- Sin decisiones de producto abiertas; defaults de los PLAN se registran al usarse.
- PR #20 (`docs/ruling-19-smoke-key`) mergeado con `763a891` el 2026-11-05,
  20:24:29 UTC. La autorización de credencial fue reiterada en la instrucción
  actual; M2 no la necesita. Head `2f25a0d`: corrección
  de M5 revisada independientemente; observación `4485511464` respondida con
  `4485574698` y resuelta; Copilot `5791519593` COMMENTED sin problemas activos.
  CI Windows `42102225168` y package `42102224714` pasaron sobre ese head.
- Máquina limpia disponible a pedido: solicitar con protocolo cuando corresponda.
- Prueba de dos horas: **omitida por ruling 5**, nunca aprobada.
- No tocar ruleset, permisos ni grabaciones privadas.

## Próxima acción

Completar ronda 1 de correcciones de tarea 3: RED independiente, fixes y re-review
de siete hallazgos (asociación revisión/audio, identidad CLI legada, retry idéntico,
resultado postcommit, ownership de productores/recuperación, marcador malformado y
categorías de error CLI). Después, suite estable y review del grupo;
después abrir PR del grupo con suite y evidencia. El PR del grupo A aún no está
abierto; ningún gate de merge del grupo se considera cumplido.
Antes de implementar M2 B, ajustar sus contratos al A final y entregar el PR de
documentación de SPEC/PLAN exigido por ruling 14. La versión 0.3.0 llega solo al
cierre de B y del hito, no al completar A.
En una reanudación consultar el ledger local
`.superpowers/sdd/M2_CATALOG_PLAN/progress.md` del worktree y git log antes de
redistribuir tareas.

## Recuperación de la orquestación actual

- Implementador de tarea 3: subagente `ingestion_green`, worktree `m2-ingestion`,
  base `1cdaadb`; no tocar sus fuentes mientras escribe.
- RED adicional: `catalog_red`, contratos de propiedad, privacidad, reconstrucción,
  retry fallido/legado, UTC y limpieza desconocida. Tests solo sintéticos.
- Rulings 23–30 fijan locks, marcadores locales, warning separado, roles permitidos,
  recibos, snapshots y staging registrado. Ruling 31 agrega el contrato pendiente
  de migración 2 `revision_audio_assets`/recibo v2/show/rebuild; no modifica migración 1.
  No hay nueva dependencia ni red.
- Reviewer independiente de tarea 3: `ingestion_review`; informe y snapshot exacto
  en `task-3-review-report.md` y `task-3-candidate-files.json`. Docs: `segments_red`.
- RED de ronda 1: cuatro archivos nuevos `tests/test_memory_review1_*.py` en
  `m2-preparation`; implementación en `m2-ingestion`. Preparación de SPEC/PLAN
  de M2 B: `preparation`, worktree `m2-segments`, sin implementación todavía.
- Fixture de memoria: `tests/memory_fixtures.py` en `m2-preparation` y `m2-ingestion`.
  ZIPs hostiles sintéticos de M5: `tests/portability_fixtures.py` en `m2-segments`.
  Memoria está incluida en el candidato Green; los ZIPs siguen como preparación,
  sin implementación de importador. Conservar las copias hasta integrar sus tareas.
- Reportes y logs de los roles: `.superpowers/sdd/M2_CATALOG_PLAN/` del worktree
  principal `m2-catalog`; el ledger `progress.md` registra los commits y handoffs.

## Verificación que queda abierta

La fuente de tarea 3 tiene suite local verde, pero no tiene review aceptada.
El full diagnóstico con fuente cambiante/otros checks dio 34 fallos/822 pasan/
9 skips en 363,35 s; no se presenta como corrida final ni se omiten sus fallos.
Captura aislada falló también en la base pre-tarea 3
(15 fallos/18 pasan/87,08 s) y en el candidato (17/16/87,56 s). El diagnóstico
`capture-diagnostic-report.md` midió un timeout durante bootstrap y reprodujo la
interferencia de un lector con replace; no se atribuyen todas las fallas a carga.
El arnés corregido de `tests/test_isolated_capture.py` pasó cuatro casos en
19,71 s y los 33 aislados en 126,10 s, sin cambiar deadlines del producto
(ruling 29). `capture_diagnostics` aceptó SPEC/calidad de ese fixture; no es
review de tarea 3. La primera suite estable posterior dio 6 fallos/877 pasan/9 skips
en 486,12 s: la instrumentación de limpieza contaba lecturas ajenas al fallo
inyectado. El autor RED la acotó al delegado de limpieza, conservando todas las
aserciones; sus diez casos pasaron en 16,76 s (Green: 17,10 s). La segunda suite
estable dio 883 pasan/9 skips en 598,56 s; logs/XML separados en el worktree Green.
Sus skips son siete de symlink y dos de wheel sin entorno no editable configurado.
La revisión estática del snapshot exacto requiere siete correcciones aunque esa
suite pase. Antes de dar el grupo por listo faltan regresiones/fixes y re-review,
suite del código corregido, review del grupo y gates exactos de GitHub.
No ejecutar tests/builds/mypy
en paralelo mientras se corre la suite de deadlines.
