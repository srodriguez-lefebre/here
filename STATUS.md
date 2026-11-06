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

## Tarea en curso: dos frentes separados en ejecución

El PR de contratos [#22](https://github.com/srodriguez-lefebre/here/pull/22) se
mergeó en `73d3cb384bd422de6899d8c7b761f3f3159ea3f1` a2026-11-06T03:59:54Z.
Head revisado `96f76dbbcb5ab4ba92866d9acf73c2673e140ee8`; review independiente
APPROVED, PC1–PC3/R45 ADDRESSED. Copilot5795547739 recibido03:57:00Z sin issues
abiertos; observación BM25 respondida4488851997/hilo resuelto. CI Windows exacto
42159238848 pasó980/2wheel skips/304,66s y package42159238847 success. Merge
con `--merge --admin --match-head-commit 96f76dbbcb5ab4ba92866d9acf73c2673e140ee8`.
Esto acepta documentación; no implementa ni certifica búsqueda/ciclo de vida.

Root creó los dos chats pedidos en el mismo proyecto, cada uno con worktree y
rama propios. Ambos se inicializaron clean en contratos73d3cb3 con A53712c5 como
ancestro; hoy contienen el trabajo detallado debajo. No recrearlos ni reiniciarlos.

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

Leases actuales: B posee models/repository/database/service, artifacts solo para
estado2/compatibilidad1, recovery y CLI. DTOs R46/errores canónicos y B4 completos
ya se integraron en M3; esto no concede un escritor libre de core al chat M3.
M3 posee search.py/tests y UI enumerada. B no tiene writer UI; la futura línea
app.py R52 espera API B verificada y RED consumidor, en la misma instancia/customroot.
Controller tiene ahora un único writer separado b5_tail_green del chat B,
controller.py solamente en el WT nativo566; GREEN principal tiene revocado ese
archivo en todos los worktrees. Processing continúa sin concesión.
Cualquier handoff adicional se concede explícitamente. Migraciones/pyproject/uv.lock/versiones
siguen seriales y requieren permiso de root. Global docs son root-only.

Runtime ACTUAL: B tiene grant EXCLUSIVO secuencial para tail2 y después H1/H2 GREEN9
cuando publique freeze, nunca simultáneos. RED H1/H2 terminó7F/2P/0E/S/6,63s y
liberó procesos; tests-only225144dc4dba807d093e1302ad2dfcc2ee076d66. M3 liberó
después de preview2 y no retiene procesos. Tail controller-only y GREEN principal
escriben archivos disjuntos; review/docs y preparación continúan en paralelo.
No full/build/uv/mypy/benchmark concurrente. Native RED34 observado/test-committed.
READERS7 terminó7P/5,10s, incluido primer uso con raíz vacía. M3 verificó44P/14,42s
con título/Conflict/FTS B4 real, source75be486/srcd1bddf68, test-format ed8e5aa.
Mypy79→78, cero diagnósticos nuevos; primer79→87 retenido. Foco legado33:30P/3F/
10,44s por mensajes vacíos; fix5db071a y foco7P/1,92s/Ruff/formato/scoped review
aprobados. No trasladar44/mypy al nuevo sourceE3 como nueva corrida.
La UI42P/review R1–R3 conservan su alcance previo; ningún grupo está cerrado.
La validación corta de integridad del fixture M4 también terminó y liberó host:
exit0/0,326s; sin here/proveedor/pytest. Faltan full/build/benchmark y mypy final
M3 sobre fuente integrada posterior a E3; el mypy79→78 previo corresponde a D1.
Cada coordinador libera inmediatamente al terminar los comandos, antes de reportes
o commits; root rota el host con esa confirmación. Escritura autorizada, docs y
reviews de lectura siguen en paralelo, nunca dos tests/builds/mypy pesados.

Última verificación B4: fuente4e1b2cead486563516b69e09fb45543ee85b53aa,
src4ca55bbe7e88ff59f79dee47e8740fef2253718e y testsbf4f471a87ad84df8ac41f8da375f8fd1025f42e.
Review R1 ADDRESSED/SPEC/quality aprobados sin hallazgos; mypy79→79 sin delta.
FULL2 pasó1043P/9S/390,03s, XML1052/0F/0E/9S/interno389,952s en
`task-4-full-2.*`; HEAD/src/tests clean y sin mutaciones. Dos skips wheel sin
entorno y siete privilegios symlink. Task4 ACEPTADA en local tras docs distintos
96e8aed255a442acb14bbdff5c32877ec201c628, source/tests invariantes. Root autorizó a
M3 consumir SOLO ese frozen B4, preservando DTOs/error canónico y sin lease libre
de core. Edición real/FTS M3 ya tiene44P; integración B5 guard/reindex pendiente.
B5 core y adapter nativo tienen RED real; implementación final aún no aceptada.

B5 RED100 observado92F/3P/5S/47,39s, XML0E. Dos pases son controles válidos; el
tercero era falso positivo de --yes por comando ausente. Un fallo era fixture sin
speaker. Tres reparaciones test-only075e280d28805b87012c059a74b5d9f95b54aeb7,
1654 líneas/3 archivos, se verificaron luego en M3 test-mergef3f06bc con B4 real
intacto:3F/0E/3,44s, fallos correctos de B5 ausente; no repetir100 ni sumar runs.
Primer GREEN B5:94P/1F/5S/82,78s sobre fuente candidata dirty documentada, no4ca55
como si fuera ese código. Fallo real de lectura que migraba bajo SQL Busy; fix
serviceSHA29ACF3565A6E554C944194DB85673E9CDC6B378830375897B0C60CD10695A845,
patchE6978215EE65CC29E268212D109BE6ED591A2B5416ABF0690343A035C384A18C.
Caso único1P/2,89s/XML0F0E0S, source/tests congelados y host liberado. No llamarlo
100 verde. READERS7 pasó7P/5,10s y foco guard/reindex34 pasó33P/1symlink skip/
16,57s; Ruff/formato propios pasan, mypy fresco79→79 sin delta. No full/interop B5.
Candidato técnico c9f64c3a65bf640196934e9c515ca158e0759ebf, cinco archivos fuente:
review independiente CHANGES REQUESTED, H1/H2 Important/P2 abiertos. H1 probe FTS
enmascara OperationalError de storage/Busy como FTS ausente; H2 registry I/O dentro
de BEGIN IMMEDIATE en edición/status. READ BEGIN R50 sí es intencional. Mismos
RED/GREEN/reviewer continúan fix acotado; no handoff c9 a M3 hasta re-review verde.
Reporte task-5-handoff-review-report.md. RED H1/H2 real7F/2P/6,63s, sin errores/skips,
test-commit225144d; storage/Busy y cinco lecturas bajo WRITE alcanzaron los asserts.
Los dos controles pasan. GREEN principal corrige ese alcance con controller revocado.

RED nativo R52 tiene su worktree separado verificado:
`C:/Users/savar/.codex/worktrees/m2-native-owner-red/here`, rama
`codex/m2-native-owner-red`, tests-only5667553213202655227d55f0e5fb6437e078018a.
Corrida real34:27F/4P/3S/0E/20,75s; tests SHA4EA0AC6AD6160B679189ED2A2CD3CACD8AAD82C87B7E75E8B728890CEFA12EFA.
Incluye tail existente que libera ownership canónico antes del cleanup nativo y
dos guards destructivos reales; APIs ausentes/controles se clasifican aparte en
task-5-native-red-report.md. No son34 fallos ni prueba de cleanup por API ausente.
RED terminó/congeló archivo y pasó a H1/H2 en faab. Nuevo GREEN b5_tail_green
solo controller.py en ese WT; principal conserva servicio/contexto/CLI/recovery
en faab, sin controller. Tail/source R52 aún no aprobados; cero writer UI B.

M3 checkpoint primario docs d7009c724da4fb5eca01d8e28fcec56f6529bf3d, source
e3ede58c32f5d848988b94dd8c4a5319512546c3, tests35efb958f45c9b107f3639fe018c6f0df42b3b79.
Preview1 funcional pasa, visual no aprobada por glifos cuadrados. Probe sin here
aisló QFontDatabase offscreen vacío. Preview2 con Segoe UI existente registrado
solo en arnés pasa función y texto legible en capturas533x452/1100x800; root vio
ambas. Conserva scroll/clipping de encabezado y combos, sin aprobación general
de layout ni cambio de producto/config/font global. Scoped reviewer aprobó solo
esa evidencia; docs distintos ya integrados en00cf0ea97bc97a6f8b02d242e26d179f407f8003,
sourceE3/tests35 sin cambios. Tres corridas Qt separadas,
no pytest/benchmark/DB/audio/red/proveedor. P95 real1000/100000 con guard+heartbeat,
full/mypy/review integrada y gates hospedados de M3 siguen pendientes.

Historia de corridas, no estado vigente ni resultados sumables:

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
UI post-DTO histórico:2F/8P/26 errores por LibraryView ausente en el baseline.
Después focos13F/25P,9F/29P,9F/29P conservaron causas/harness; el foco35P/3F
alcanzó assertions R48 reales. Correcciones UI y foco final38/38P/12,57s tras
formato/Ruff verifican roster, speakers scoped, targets pasivos y teclado, con DB
sintética/customroot real para dos casos. No edición real B4/guard B5/perf/full aún.
R1 compatibilidad129:128P/1F/86,28s y callers40P/30,10s. El único fallo era conteo
global del journal: RED preservó publicación exacta, agregó persist_state complete
y ausencia global de prepared; caso único corregido1P/3,15s, testf244018. No sumar
estos pases como full129; full corregido pendiente sobre fuente4e1b2ce.

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

M5 B preparación SPEC/PLAN congelada en `m2-segments`, blobs7904905/c77a3c6;
8 cláusulas/14 bloques preservados y review independiente estática READY FOR
PREPARATION, cero hallazgos, reporte `m5-portability-preparation-review-report.md`.
Core permite avanzar tras merge real M2 B, PR docs/handoff B5 y GO;
claims/answers esperan M4 B y adapters sus contratos reales. Sin código/runtime
portabilidad; reporte `m5-portability-contract-preparation-report.md` en ese worktree.

M3 B preparación SPEC/PLAN lista en `m2-segments`, blobs03207dd/9c177e5 tras deltas
aditivos R46/R47/R50 y76 líneas R51/R52. Preflight y scoped reviews independientes READY
FOR PREPARATION, cero hallazgos,11 cláusulas/14 bloques completos preservados.
Reportes `m3-library-preparation-review-report.md` y
`m3-library-binding-delta-review-report.md` y `m3-library-r51-r52-binding-review-report.md`;
Citation/offsets/Unicode, empty MeetingPage/InvalidInput.reason, guard READ,
ACK/leases/fase durable/contexto explícito fijados; adaptadores
reales B5/UI y PR docs final todavía pendientes. B puede arrancar sobre A revisado con PR abierto, tras contratos
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
Contrato B5/M3 R50 fijado tras review independiente estática, reporte
`b5-m3-guard-contract-review.md`: B posee lock y única read transaction;
snapshot de UUIDs y aliases exactos antes de LIMIT, fingerprint incluye IDs
resueltos; caller abre/cierra. Parser único, sin nuevas migraciones o p95 relajado.
No es implementación/aceptación B5; ambos frentes reciben API antes de integrarla.
R51 ya fija DTO DeletionResult/ReindexReport, categorías/fase durable, ACK fuera
del lock/SQL y revalidación, purger privado en transacción y roles de temporales.
Reporte `b5-result-hook-contract-review.md`; la interoperabilidad real se prueba
en M3 integrado con fuente B5 exacta, sin duplicar búsqueda en B. Source B5 aún
no aceptado; las nueve fronteras R42 y todos los gates siguen obligatorios.
R52 fija contexto nativo explícito y lifetime canónico; review de callpaths reales
`b5-native-owner-contract-review.md`. No registry durable de roots, ni autoridad
por origins. RED suplementario observado; código adapter espera GREEN/review, sin cambiar deadlines
ni ubicación/retención/privacidad. Posible línea GUI es del writer M3 y espera API
B verificada/RED consumidor; ninguna segunda instancia por defecto.

Mapa de cierre M2 listo: `m2-milestone-closure-readiness.md` coteja16 criterios y
DoD, evidencia A y faltantes B/0.3, nueve fronteras R42 y protocolo build/smoke.
No exige máquina limpia antes de M5 ni convierte full B4 con R1 abierto en cierre.
Fixture sintético M4 preparado por catalog_group_fixes, único archivo nuevo
`m2-preparation/here/tests/grounded_fixtures.py`, blob7d74c8221c219779099d5a3d4654951692f2997e,
SHA256318E31615073D5DDC9B73937DA10A1DA7BA291A59F74497C1983533F6094D265.
4 reuniones/5 revisiones/13 pasajes/8 anchors, estándar y declarativo. Validación
independiente de integridad exit0/0,326s, dos construcciones frescas, IDs/referencias/
8 spans Unicode y hash inalterados; sin here/proveedor/pytest. Dos exit1 previos
del arnés (whitelist tuple y stdout cp1252) conservados en
`m4-synthetic-corpus-validation-report.md`; ninguna aceptación semántica M4.
Futuro RED inspecciona/consume al recibirlo UNA vez por root.
Reporte `m4-synthetic-corpus-preparation-report.md` en ese worktree; corpus previos intactos.

Preparación documental M4 A en paralelo: responsable catalog_group_fixes,
`C:/Users/savar/.codex/worktrees/m4-grounded-preparation/here`, rama
`codex/m4-grounded-preparation`, base73d3cb384bd422de6899d8c7b761f3f3159ea3f1.
Brief explícito entregado; solo M4_GROUNDED_SPEC/PLAN y reporte ignored, sin
código/fixtures/runtime/GO M4. Conserva dependencia cierre M3. El actual SearchQuery
no provee selección meeting_ids ni OR del retrieval M4: handoff interno pendiente,
sin asumir MemoryService.search, SQL directo de intelligence o filtro post-LIMIT.
Root resolverá ese contrato; no es bloqueo humano ni aceptación de un adapter.

No hay bloqueo humano actual. Próxima acción: recoger RED H1/H2, corregir/revisar
ese alcance y handoff verificado a M3, mientras segundo GREEN termina controller.
Reanudar desde estos IDs/reportes sin recrear los chats ni repetir el catálogo A.
