# Estado de here

Actualizado: 2026-11-06. Leé CLAUDE.md primero; después este archivo y
`docs/MASTER_PLAN.md`. Historial y rulings: `docs/development/DELIVERY.md`.

## Objetivo y checkpoint vigente

Implementar y entregar M2–M5 completos bajo ruling 20, sin cambiar definiciones
de terminado. **M2 A cerrado y mergeado** por PR [#21](https://github.com/srodriguez-lefebre/here/pull/21).
Nivel: implementado y verificado automáticamente con datos sintéticos; no aporta
certificación nueva de hardware, proveedor o instalación limpia. M2 sigue abierto:
ciclo de vida/borrado y versión 0.3.0 corresponden a B/cierre del hito.
Versión vigente 0.2.0; M5 exige distribución Windows 1.0.0 y aceptación real.

Merge A: `53712c5beedafc2d40e2abf10c13362a1c3c9f78`, 2026-11-06T03:22:06Z.
Head revisado/publicado: `8a498e207812592ff84a0acc5ba4e9f392e02373`.
Fuente final `c59ae331ebf124400ef3e476e5f0fcc029e83999`, src tree
`5c66169c0a93a8f0606a011bd9570bbdd9137903`, tests tree
`fb958d021eaf7d149c00a3a1be174e36fbef2a1c`, iguales al merge.

Los cinco pasos de §7 se cumplieron en orden: PR no borrador con evidencia y
review independiente; Copilot solicitado sobre 8a y recibido como review
`5795325438` COMMENTED a03:20:34Z, sin issues abiertos y C5 resuelto; cinco
observaciones respondidas en sus hilos; CI exacto verde; merge commit con
`--merge --admin --match-head-commit 8a498e207812592ff84a0acc5ba4e9f392e02373`;
ledger en DELIVERY. C5 reply `4488660276`, hilo `PRRT_kwDORsQx4M60W6_-` resuelto.
Suite Windows `42155415514`: **980 pasan/2 skips wheel/307,46 s**.
Package Windows `42155415501`: success. Los dos smokes wheel omitidos en CI
pasan por separado en el wheel fresco exacto: distribución4/7,51 s.

## Tarea en curso: dos frentes separados, inicialización verificada

El PR de contratos [#22](https://github.com/srodriguez-lefebre/here/pull/22) se
mergeó en `73d3cb384bd422de6899d8c7b761f3f3159ea3f1` a2026-11-06T03:59:54Z.
Head revisado `96f76dbbcb5ab4ba92866d9acf73c2673e140ee8`; review independiente
APPROVED, PC1–PC3/R45 ADDRESSED. Copilot5795547739 recibido03:57:00Z sin issues
abiertos; observación BM25 respondida4488851997/hilo resuelto. CI Windows exacto
42159238848 pasó980/2wheel skips/304,66s y package42159238847 success. Merge
con `--merge --admin --match-head-commit 96f76dbbcb5ab4ba92866d9acf73c2673e140ee8`.
Esto acepta documentación; no implementa ni certifica búsqueda/ciclo de vida.

Root creó los dos chats pedidos en el mismo proyecto, cada uno con worktree y
rama propios. Git verificado por root: ambos clean en la base de contratos73d3cb3,
con el merge A53712c5 como ancestro. Inicialización solamente; antes del GO no
hay código/tests/docs nuevos ni suites/builds/proveedor ejecutados por los frentes.

Proyecto `local-e307e59e09194b354122137b16ef6d77`, host `local`; raíz original
`C:/Users/savar/Documents/proyects/here` preservada. Root orquesta desde
`C:/Users/savar/.codex/worktrees/m2-catalog/here`, rama
`codex/m2b-m3a-orchestration`, con base73d3cb3.

| Frente / responsable ejecutor | Chat ID | Rama | Worktree | Base |
|---|---|---|---|---|
| M2·B: ciclo de vida y borrado | 01a1af04-b158-75a2-8933-98783880f3e8 | codex/m2-lifecycle | C:/Users/savar/.codex/worktrees/faab/here | 73d3cb384bd422de6899d8c7b761f3f3159ea3f1 |
| M3·A: búsqueda y acciones | 01a1af04-e779-7363-be61-c840651b0b18 | codex/m3-search | C:/Users/savar/.codex/worktrees/2710/here | 73d3cb384bd422de6899d8c7b761f3f3159ea3f1 |

Briefs completos de cada frente en `.superpowers/m2b-front-brief.md` y
`m3a-front-brief.md`: objetivo, SPEC/PLAN, bases reales, archivos/leases,
dependencias, límites y DoD. Metadatos completos y GO entregados a ambos chats;
cada ejecutor usa subagentes RED/GREEN/review independiente/docs por tarea.
Root conserva contratos, integración, global docs y todos los merges. No reiniciar
trabajo previo ni duplicar fixtures. Monitorear los dos chats con wait_threads;
pedidos de leases/handoffs se registran en su progreso/reporte local.

Leases actuales: B posee memory repository/database/service/artifacts, controller/
recovery y CLI; artifacts se concedió para estados version2 con compatibilidad1.
M3 posee search.py/tests y UI enumerada; models.py DEVUELTO A B para R48 después
de congelar DTOs M3 en5c9197a/integración42d147a, blob4680f29d.
M3 puede integrar solo ese commit y errores canónicos0452945, no el service B4
parcial. B todavía no tiene writer UI. Processing continúa sin concesión.
Cualquier handoff adicional se concede explícitamente. Migraciones/pyproject/uv.lock/versiones
siguen seriales y requieren permiso de root. Global docs son root-only.

Runtime ACTUAL: B4-R1 RED acotado concedido después del full/release B y de M3.
M3 escribe UI sin tests/lint/mypy/build hasta nueva concesión. B GREEN R1 espera
RED real; no repetir full previo. Sin build/benchmark todavía. Cada coordinador libera la ventana
con evidencia real; root la asigna al otro
frente. Nunca dos tests/builds/mypy pesados en el host. Leer/escribir archivos
autorizados y preparación documental sí puede seguir en paralelo.

RED observados, sin aceptación nueva de tareas:

- B4:111 casos,47 fallan/64 pasan/0 errores/0 skips/76,66s; XML interno75,352s.
  59 controles A/M1 y5 controles de marker pasan; fallan APIs/efectos B4 ausentes.
  Prueba inicial43F/63P se conserva; corrección de un supuesto de formato JSON,
  sin cambiar intención/autoridad, previa a implementar. Test-only9d311a9.
- M3 Task1 público:111 fallan por módulo ausente/1 benchmark omitido/66 casos
  privados diferidos/6,91s, fuente test-only3d6a6ac; XML112/111F/0E/1S.
  UI:2 fallan/8 pasan/3,31s, fuente3becc79;16 casos de widgets pendientes de DTOs.
  No presentar preparaciones/deselecciones como tests aprobados.

Checkpoints posteriores: B4 selección111/111 GREEN76,83s, foco afectado294 pasan/
4 deselected/162,35s y mypy79→79 sin delta (previos al suplemento R48).
R48 RED11F/39deselected/5,26s, commit1f4d22d; GREEN122P/78,72s y probe aislado
verificados. Handoff e0d083ae2aebb0ef1b1d428f6b3beba42ac9030f,125 líneas en tres
archivos, autorizado a M3 preservando sus DTOs. API real ya disponible técnicamente,
sin aceptación global B4. M3 core e4d2469:177P/1benchmarkskip/26,79s, los66 controles
estructurales ejecutados; review APPROVED CORE con B5/performance/full pendientes.
UI post-DTO:2F/8P/26 errores por LibraryView ausente en el baseline; asserts de
widgets todavía no alcanzados, no verificación de R48 ni GREEN UI.

B4 FULL1 en9e10c9cb1e0d385feea8cd73c8912b9ebd8a3929 pasó1036/9skips/391,45s
(1045 casos,0F/E;2wheel ausente+7privilegios symlink). REVIEW necesita fixes por
B4-R1 Important: snapshot postcommit de transición/evento no queda journalizado
si falla el write, próximo reconcile no lo repara. R49 y una sola tanda RED/fix/
re-review por los mismos roles; Task4 y B5 siguen abiertos pese al full verde.
Fuente full congelada src284c24aa/tests01db3dd; logs/review en ledger B4.

Leér reportes de progreso canónicos para pedidos de lease y resultados (además
de wait_threads): `faab/here/.superpowers/sdd/M2_LIFECYCLE_PLAN/progress.md` y
`2710/here/.superpowers/sdd/M3_SEARCH_PLAN/progress.md`, bajo la raíz de worktrees.
R46 fija los defaults de DTOs/errores compartidos; R47 define empty-search→MeetingPage
con filtros intactos. Antes de UI GREEN completar RED real con los DTOs. El marker
slow se registrará por root en el próximo cambio serial de metadatos/versionado.

M3 consulta el catálogo A; título real espera contrato B4 aceptado y reindex/
supresión real espera B5. R45 exige generación de búsqueda coherente, rechazo
estructurado antes de usar ranks viejos y reinicio por requests existentes;
el listado fecha/ID conserva continuidad ante inserts. No contar fakes como
integración. Root facilita los contratos/DTOs compartidos con un solo writer.

## Secuencia y concurrencia autorizada

Los nueve grupos: M2 A `M2_CATALOG` tareas1–3 (**mergeado**); M2 B
`M2_LIFECYCLE`4–5; M3 A `M3_SEARCH`1–2; M3 B `M3_LIBRARY`3–4;
M4 A `M4_GROUNDED`1–3; M4 B `M4_PRODUCT`4–5; M5 A `M5_SETTINGS`1–2;
M5 B `M5_PORTABILITY`3; M5 C `M5_RELEASE`4–5. Versiones al cerrar hitos:
0.3.0, 0.4.0, 0.5.0 y1.0.0, en serie y solo con sus criterios verificados.

M2 B y M3 A tienen los gates de contratos cumplidos y avanzan tras sus GO. Después
de M2 B puede adelantarse M5 B en lo permitido por §11, con claims pendientes
hasta M4 B. El resto conserva las dependencias de los planes. Preparación del
PLAN siguiente, fixtures sintéticos y respuestas a Copilot pueden solaparse.

Un escritor por archivo, incluso entre worktrees. Migraciones, pyproject/uv.lock y
versiones nunca en paralelo. Suites de tiempos/deadlines, benchmarks, tests y
builds pesados usan lease exclusivo del host; sin otro test/build/mypy pesado.
Cada tarea: RED independiente → GREEN → review por otro autor → docs/evidencia.
Cada PR conserva los cinco pasos de §7 y Copilot recibido/respondido sobre el head
actual, nueva review tras cambios sustantivos, CI exacto y merge guardado. Sin
excepción por silencio, sin tocar ruleset/permisos.

## Evidencia y recuperación sin repetir trabajo

Fuente y aceptación A: `docs/development/M2_CATALOG_SPEC.md`, `_PLAN.md` y
`M2_CATALOG_EVIDENCE.json`; matriz/corridas en `ACCEPTANCE.md`, ledger en DELIVERY.
La implementación ya resuelve F1–F7, NB1/NB2, GR1–GR3 y C1–C5; no reabrir sus
reviews ni repetir suites históricas sin un cambio/fallo nuevo que lo justifique.
Reports locales bajo `.superpowers/sdd/M2_CATALOG_PLAN/`:
`progress.md`, `task-1-fix-review-report.md`, `task-2-review-report.md`,
`task-3-fix-1-review-report.md`, `task-3-fix-2-review-report.md`,
`task-3-fix-3-review-report.md`, `group-review-report.md`,
`group-fix-review-report.md`, `capture-fixture-forwarding-report.md`,
`copilot-1-review-report.md`, `copilot-2-review-report.md`,
`copilot-3-red-report.md`, `copilot-3-implementation-report.md`,
`copilot-3-review-report.md`, `copilot-3-docs-report.md` y `m2a-merge-docs-report.md`.
CI actual: `.superpowers/m2a-ci-42155415514.log`; wheel exacto:
`.superpowers/m2a-wheel-catalog-probe-c59ae33.json` y
`.superpowers/m2a-copilot-3-wheel-smokes.log`/`.xml`.
Los fulls fallidos1/3, diagnósticos, timeout no explicado y pases históricos
conservan alcance en DELIVERY/ACCEPTANCE/evidence y sus logs; no se relabelan.

Preparación futura ya existente: `next-contracts-preparation-report.md`,
`parallel-contracts-review-report.md`, `lifecycle-plan-preparation-report.md`,
`m3-plan-preparation-report.md`; briefs `.superpowers/m2b-front-brief.md` y
`m3a-front-brief.md`, ya completos y entregados a los chats.
Fixtures de memoria/búsqueda/playback/ZIPs siguen preservados en sus worktrees;
su generación no certifica benchmark, seek o importador.

M3 B preparación SPEC/PLAN lista en `m2-segments`, blobs11c2cf4/0e0a44f.
Preflight independiente READY FOR PREPARATION, cero hallazgos,11 cláusulas/14
bloques completos de tareas preservados. Reporte `m3-library-preparation-review-report.md`;
bind final a Citation/offsets R46, empty return R47 y adaptadores reales B4/B5/UI
todavía pendiente. B puede arrancar sobre A revisado con PR abierto, tras contratos
necesarios, y rebasear tras merge A: no agregar un bloqueo de merge más estricto.

Fixture real de actualización 0.2.0 ya descargado una vez desde el workflow M1
42102224714/artifact13450512413: `.superpowers/m5-upgrade-baseline-0.2.0/`.
Se verificaron seis checksums y 730 entradas de inventario; reporte
`.superpowers/m5-upgrade-baseline-preparation-report.md`. No se ejecutó ni instaló:
es preparación verificada del fixture, no certificación de actualización.

No se leyó el `.env` real ni audio/biblioteca privados. Smokes opt-in futuros usan
solo HERE_PROVIDER_SMOKE_KEY y texto/audio sintéticos según ruling19, sin imprimir
ni persistir la credencial. Máquina Windows limpia: pedir al necesitar la fila,
con protocolo escrito. Prueba de dos horas: **omitida por ruling 5**, nunca aprobada.
No hay bloqueo humano actual. Próxima acción: recoger RED R1 B4, ceder UI M3 si
está lista mientras B escribe el fix; después focos/re-review y full corregido.
Reanudar desde estos IDs/reportes sin recrear los chats ni repetir el catálogo A.
