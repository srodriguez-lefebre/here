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

## Tarea en curso: contratos para los dos frentes

Ruling 35: después de A, completar el PR de documentación necesario antes de crear
los chats separados M2 B y M3 A. **Todavía no se creó ninguno ni empezó su código.**
Rama actual de mantenimiento: `codex/m2b-m3a-contracts`; base real A merge
`53712c5beedafc2d40e2abf10c13362a1c3c9f78`; worktree raíz de orquestación
`C:/Users/savar/.codex/worktrees/m2-catalog/here`. PR de contratos
[#22](https://github.com/srodriguez-lefebre/here/pull/22), head
`1fd228fecc255198cfa00f8969d35c609c9de9e6`: review independiente final APPROVED,
PC1–PC3 ADDRESSED. Copilot5795454802 recibido sobre ese head a03:41:49Z abre
observación4488791517: BM25 cambia con el corpus y puede invalidar ranks de cursor.
R45 precisa generación de búsqueda reproducible y rechazo/reinicio estructurado,
con listado fecha/ID intacto. El diff documental corregido espera review acotada,
publicación/respuesta y nueva review Copilot. CI1fd pasó980/2wheel skips/294,91 s
(tests42157559503 y package42157559427 success); será histórico para el head nuevo,
que requiere sus propios gates. Merge docs pendiente; ningún chat creado.
No confundir aprobación documental con implementación de B/M3.

Documentos a reconciliar/revisar: `docs/development/M2_LIFECYCLE_SPEC.md` y
`M2_LIFECYCLE_PLAN.md`; `M3_SEARCH_SPEC.md` y `M3_SEARCH_PLAN.md`;
agenda de `docs/MILESTONE_3_PLAN.md`. Heredan el A aceptado y R39–R44:
NULL/cursor y orden estable, generación de búsqueda R45, parcial opt-in sin otro indexador,
barrera durable de borrado/supresión y warnings fail-closed de recuperación.
M3 A puede avanzar búsqueda independiente desde A; título/reindex y guards
compartidos esperan el handoff aceptado de B y lease de root. Fakes no cierran
esa integración. No se inventan contratos de B como APIs existentes.

Root prepara un brief por frente con objetivo, SPEC/PLAN, commit base real,
archivos exclusivos/compartidos, dependencias, límites y DoD. La base de trabajo
se fija tras el merge del PR de contratos, conservando este merge A como ancestro.

Proyecto: `local-e307e59e09194b354122137b16ef6d77`, host `local`, Git habilitado,
raíz original `C:/Users/savar/Documents/proyects/here` (checkout del usuario intacto).

| Frente | Chat ID | Rama | Worktree | Base A merge | Responsable |
|---|---|---|---|---|---|
| M2 B · ciclo de vida/borrado | Pendiente | Pendiente | Pendiente | 53712c5beedafc2d40e2abf10c13362a1c3c9f78 | Pendiente de asignar por root |
| M3 A · búsqueda/acciones | Pendiente | Pendiente | Pendiente | 53712c5beedafc2d40e2abf10c13362a1c3c9f78 | Pendiente de asignar por root |

Registrar IDs, ramas, worktrees, base de contratos y responsables reales al crear
los chats. Cada chat ejecuta su grupo completo con subagentes RED/GREEN/review/
docs; root conserva contratos, leases, integración y merges. No reiniciar tareas
ni duplicar fixtures o reviews ya aceptados.

## Secuencia y concurrencia autorizada

Los nueve grupos: M2 A `M2_CATALOG` tareas1–3 (**mergeado**); M2 B
`M2_LIFECYCLE`4–5; M3 A `M3_SEARCH`1–2; M3 B `M3_LIBRARY`3–4;
M4 A `M4_GROUNDED`1–3; M4 B `M4_PRODUCT`4–5; M5 A `M5_SETTINGS`1–2;
M5 B `M5_PORTABILITY`3; M5 C `M5_RELEASE`4–5. Versiones al cerrar hitos:
0.3.0, 0.4.0, 0.5.0 y1.0.0, en serie y solo con sus criterios verificados.

M2 B y M3 A podrán avanzar simultáneamente tras los gates de contratos. Después
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
`m3a-front-brief.md`, pendientes de completar con bases y asignaciones reales.
Fixtures de memoria/búsqueda/playback/ZIPs siguen preservados en sus worktrees;
su generación no certifica benchmark, seek o importador.

Fixture real de actualización 0.2.0 ya descargado una vez desde el workflow M1
42102224714/artifact13450512413: `.superpowers/m5-upgrade-baseline-0.2.0/`.
Se verificaron seis checksums y 730 entradas de inventario; reporte
`.superpowers/m5-upgrade-baseline-preparation-report.md`. No se ejecutó ni instaló:
es preparación verificada del fixture, no certificación de actualización.

No se leyó el `.env` real ni audio/biblioteca privados. Smokes opt-in futuros usan
solo HERE_PROVIDER_SMOKE_KEY y texto/audio sintéticos según ruling19, sin imprimir
ni persistir la credencial. Máquina Windows limpia: pedir al necesitar la fila,
con protocolo escrito. Prueba de dos horas: **omitida por ruling 5**, nunca aprobada.
No hay bloqueo humano actual. Próxima acción: cerrar la review del diff R45,
publicar/responder su hilo y recibir nuevo Copilot/CI exacto; mergear el PR de
contratos con sus gates, crear los dos chats con briefs y registrar su estado real.
