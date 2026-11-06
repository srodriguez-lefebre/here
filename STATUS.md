# Estado de here

Actualizado: 2026-11-05. Leé CLAUDE.md primero. Historial y rulings:
`docs/development/DELIVERY.md`.

## Objetivo vigente

Implementar y entregar M2–M5 completos por instrucción explícita del usuario
(ruling 20). Primero cerrar M2 A con todos sus gates. El ruling 35 ordena después
los dos frentes de proyecto M2 B y M3 A; todavía no se creó ninguno. Se mantienen
las definiciones de terminado y los tres niveles de evidencia. Versión vigente
0.2.0; 0.3.0 solo al cerrar B/M2. M5 exige distribución Windows 1.0.0 y aceptación real.

## Código y secuencia

- Base histórica: `878e7d4`, v0.2.0; PR #20/ruling 19 mergeado en `763a891` e
  integrado. Grupo actual: `codex/m2-catalog`, worktree
  `C:/Users/savar/.codex/worktrees/m2-catalog/here`.
- Código integrado actual: `8087e252112fd2e6de1efeff1d0f3f531d4be77d`, equivalente
  a Green `4a3cb7c` tras rondas 2/3. Src tree
  `948efa342c9569ca751cffb4718d82f3dd872de6`, tests tree
  `30c62e6f7e9e0a818e125a6c7762db91335b47f1`; source/config/scripts/workflows
  sin diferencias con Green. Integración no equivale a aceptación ni merge de A.
- Los nueve grupos se conservan: M2 A (`M2_CATALOG`, 1–3), M2 B
  (`M2_LIFECYCLE`, 4–5), M3 A (`M3_SEARCH`, 1–2), M3 B (`M3_LIBRARY`, 3–4),
  M4 A (`M4_GROUNDED`, 1–3), M4 B (`M4_PRODUCT`, 4–5), M5 A
  (`M5_SETTINGS`, 1–2), M5 B (`M5_PORTABILITY`, 3), M5 C (`M5_RELEASE`, 4–5).
  Versiones al cerrar hitos: 0.3.0, 0.4.0, 0.5.0, 1.0.0.
- Root es el único orquestador de contratos, leases por archivo, integración y
  merges. Un escritor por archivo; migraciones, pyproject/uv.lock y versión en
  serie. Suites de deadlines, benchmarks y builds pesados se ejecutan en ventanas
  exclusivas, sin otros tests/builds/mypy pesados.
- Cada tarea: subagente RED → GREEN → review independiente → docs/evidencia por
  otro rol. Cada grupo: PR con evidencia/review → Copilot recibido y respondido
  sobre head actual, nueva review tras fixes sustantivos → CI Windows verde en ese
  head → merge commit con `--admin --match-head-commit` → ledger. Ningún gate se omite.

## Tarea en curso y resultados reales

`docs/development/M2_CATALOG_PLAN.md`, tarea 3. SPEC `M2_CATALOG_SPEC.md`; hito
`docs/MILESTONE_2_PLAN.md`. Tarea 3 y grupo A siguen sin aprobar.

- Tarea 1: review independiente aceptada después del fix Busy. Foco 34 pasan/
  1 skip/4,70 s; full `M2A-T1-FINAL-20261105` 778 pasan/7 skips/173,68 s,
  ambos smokes de wheel pasan. Historial de fuente `9a80060`.
- Tarea 2: review SPEC/calidad PASS. Fuente `037fc03`, integración `1cdaadb`;
  foco 310/1/24,60 s, capture recovery 68/25,64 s, full 783/8/160,71 s.
  Sus dos skips de wheel no sustituyen la evidencia de wheel de otras corridas.
- Candidato inicial Task3 `6c422f0`: full estable 883 pasan/9 skips/598,56 s.
  Ronda 1 marcó F1–F7 ADDRESSED, pero abrió NB1. El full histórico `1f1bc88`
  dio **5 fallos/920 pasan/7 skips/553,39 s**, logs/XML
  `.superpowers/m2a-integrated-final-1.*`; ambos smokes wheel pasaron. Sus tres
  snapshots con locks nuevos, stub sf.info y retry audio-only tienen corrección
  y cobertura posterior; el resultado fallido se conserva, sin relajar assertions.
- Ronda 2: NB1 y las regresiones de integración ADDRESSED por review independiente;
  61 pasan/68,06 s y foco controller/procesamiento/UI 69/23,02 s. Un timeout hijo
  intermedio de 30 s permanece registrado: caso idéntico pasó aislado 5,02 s y en
  foco final (call 2,72 s), sin causa probada ni cambio de timeout.
- Ronda 3: NB2 ADDRESSED, sin breakage nuevo en su diff; review
  `task-3-fix-3-review-report.md`. Guard UUID antes de construir/probar journal
  cuando falta metadata; IDs opacos del manifiesto existente siguen compatibles.
  Focos 3/2,40 s y 27/20,86 s. Ruff/format y mypy comparativo 79/79 sin nuevos.
- R36/R37 implementados: marker v2 con token/source_kind, receipt v3 con prueba
  exacta y restauración de operaciones, replay tras crash real con DB presente o
  reconstruida; handoff privado consumido una vez por controller. Reserva audio-only
  canónica con origen/revisión NULL, cero revisiones históricas y fechas captura
  NULL durables. DB sigue schema 2, receipt v2 compatible sin inventar prueba.
- **Full integrado 2: 940 pasan/7 skips/cero fallos en 325,09 s**, fuente
  `8087e25`, iniciado solo a 2026-11-06T00:01:58Z. Log/XML
  `.superpowers/m2a-integrated-final-2.*`: 947 casos, cero failures/errors,
  tiempo interno 325,029 s; ambos smokes wheel pasan, skips siete symlinks.
  Full verde no equivale a aceptación de tarea/grupo mientras falte review global.
- Review de todo A por `catalog_group_review` en curso, base `763a891`..`8087e25`.
  Tres Important provisionales: metadata completed puede persistir antes del primer
  memory_pending marker (kill deja sin replay e interrupted); receipt JSON array
  puede abortar rebuild de vecinos por `.get`/AttributeError; cancelación aceptada
  durante persistencia del fallo de captura deja UI/manifest cancelled y catálogo
  failed con conflicto/marker atascado. GR1/GR2/GR3 Important/P2 están OPEN en
  `group-review-report.md`, source verdict changes requested. Review global sigue
  provisional hasta suplemento docs comprometido; sin aceptación. Corregir juntos
  sin reabrir NB1/NB2 ya addressed.
- Wheel/sdist nuevos CPython 3.13.15 y probe off-repo `-I` no editable de `8087e25`:
  65 módulos byte-idénticos, schema 2/FTS5/FK/integrity correctos. Hashes en
  `.superpowers/m2a-wheel-catalog-probe-8087e25.json`; probes anteriores históricos.
  Sin hardware/proveedor ni certificación nueva. No procesos, protocolos, deadlines,
  dependencias o versión nuevos. Meta final M5/1.0.0 intacta.

## Próxima acción obligatoria

1. Completar la review de grupo; conservar el full 2 verde y sus límites, y los
   hallazgos/veredictos reales. Rondas acotadas NB1/NB2 ya addressed, sin reiniciarlas.
2. Atender los tres hallazgos de grupo cuando se formalicen, con RED/GREEN/re-review,
   focos y nueva suite estable si la fuente cambia. No aceptar tarea/grupo solo
   por focos y reviews de diffs; falta veredicto de todo A y verificación agregada.
3. PR de A, Copilot sobre head exacto recibido/respondido y CI Windows verde;
   merge guardado y ledger. Todavía no hay PR ni CI de A aprobados.
4. Después del merge A y de los PRs de docs de contratos necesarios, root crea
   dos chats de proyecto separados para M2 B y M3 A, con briefs completos y leases.
   No usar `8087e25` como base de merge futura: la SHA real aún es desconocida.

## Dos frentes futuros — ruling 35, todavía pendientes

Proyecto descubierto: `local-e307e59e09194b354122137b16ef6d77`, host `local`, Git
habilitado, raíz original `C:/Users/savar/Documents/proyects/here`.

| Frente | Chat ID | Rama | Worktree | Base merge A | Responsable ejecutor |
|---|---|---|---|---|---|
| M2 B · ciclo de vida/borrado | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente de asignar por root |
| M3 A · búsqueda/acciones | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente de asignar por root |

Ningún chat fue creado. Antes de iniciar, cada brief fija objetivo, SPEC/PLAN
aceptados, SHA base real, archivos exclusivos/compartidos, dependencias, límites y
DoD verificable. Cada chat ejecuta su grupo completo con subagentes RED/GREEN/
review/docs; root conserva la orquestación, contratos, integración y merges.
No reiniciar ni duplicar trabajo ya realizado. Registrar IDs, ramas, worktrees y
responsables reales aquí cuando existan, sin inventar identificadores.

Los cuatro SPEC/PLAN de B y M3 A están preparados en `m2-segments`, todavía sin PR;
ver `lifecycle-plan-preparation-report.md` y `m3-plan-preparation-report.md` en el
ledger local. Reconciliarlos con el A finalmente mergeado y entregar los PRs docs
necesarios antes de despachar. M3 A puede avanzar búsqueda desde A aceptado, pero
sus acciones de título/reindex esperan los contratos aceptados de B que requieren;
no duplicar mutaciones ni contar fakes como esa integración. El texto de agenda del
plan de hito M3 debe alinearse en ese PR, sin cambiar sus criterios de aceptación.
Tras M2 B puede adelantarse M5 B con el gancho de claims pendiente hasta M4 B.
El resto conserva sus dependencias; no se inicia ningún frente en este checkpoint.

## Recuperación, decisiones y límites

- Implementador Green: `ingestion_green`, `m2-ingestion`; reviewer independiente
  `ingestion_review`; autor RED mantiene archivos propios del próximo fix; docs
  `segments_red` en root. Nadie toca source de otro writer.
- Rulings 31–34 registran asociación revision/audio/schema2, legacy confiable y
  lease nativa `.capture_owners/` retenida por scope del trabajo hasta cierre real
  y persistencia. La review de ronda 1 está en `task-3-fix-1-review-report.md`.
- Rulings 36/37 fijan el contrato de marker v2/recibo v3 con token y
  prueba durable del replay, handoff privado de una sola publicación y reserva
  audio-only sin padre ficticio ni fechas de captura inventadas. No se agrega
  migración. Implementados e integrados en `8087e25`, aún sin aceptación de tarea/grupo.
- Todas las corridas anteriores, incluidos fulls fallidos y diagnóstico de captura,
  conservan alcance y límites en ACCEPTANCE/DELIVERY/evidence JSON. El arnés revisado
  no cambió deadlines del producto; H1 bootstrap se observó en un caso y H2 sharing
  se reprodujo sin probar el solapamiento del fallo original.
- Fixtures sintéticos de memoria/búsqueda/playback/ZIPs permanecen en sus worktrees
  bajo ownership; no son benchmark, importador ni certificación de seek.
- PR #20/ruling19 está mergeado en `763a891`: head revisado `2f25a0d`, Copilot
  `5791519593` COMMENTED, inline `4485511464` respondido `4485574698`/resuelto,
  CI Windows `42102225168` y package `42102224714` pasaron. M2 no usa proveedor real.
- Máquina limpia: pedir con protocolo cuando una fila la necesite. Dos horas:
  **omitida por ruling 5**, nunca aprobada. No cambiar ruleset/permisos ni usar datos privados.
- Versiones y calidad/privacidad/aceptación/merges conservan sus requisitos.
  No hay cierre M2 A/M2, hosted CI de grupo ni certificación nueva real.
- Para retomar: leer este STATUS, el ledger local
  `.superpowers/sdd/M2_CATALOG_PLAN/progress.md`, los reportes y git log; continuar
  los hallazgos de grupo y las verificaciones agregadas antes de crear los dos chats; no repetir
  el trabajo ya resuelto de NB1 y de los cinco fallos históricos.
