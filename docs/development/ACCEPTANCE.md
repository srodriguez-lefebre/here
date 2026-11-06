# Product acceptance evidence

Every row is open until the named run and inspectable result are recorded. Automated
fakes prove software contracts, not physical hardware or provider accuracy. Test data
must be synthetic; credentials and private conversations never enter evidence files.

| Milestone | Acceptance requirement | Evidence required | Current state |
|---|---|---|---|
| M1 | Both default Windows sources captured at 44.1/48 kHz | Actual device/format and WAV frame/duration report | Short actual two-source capture passed; long run omitted by user |
| M1 | Two-hour meeting without unbounded growth or unexplained gaps | Frames/known markers, peak memory, disk, queue and final artifacts | Omitted by explicit user instruction; not run |
| M1 | Pause/resume audio timeline, destructive cancel and saved exit | Qt and Windows synthetic marker runs | Short native pause/resume and paused save-exit passed; synthetic restart/exit barriers passed; no repeated installed capture claim; long run omitted by user |
| M1 | Provider/timestamps and diarization | Synthetic real-provider run and segment output inspected | Single/multiple-source TTS smokes passed; long transcription/diarization omitted by explicit user instruction |
| M1 | Failure/crash/network/disconnection/disk-pressure recovery | Fault-injection plus bounded hardware run | PR12 faults, writer/publication process kills and helper-owner death passed; native recoverable saved failure passed; no repeated installed fault matrix claim; long run omitted by user |
| M1 | Installed application opens, records and uninstalls | Fresh Windows installation and bundled-runtime smoke | Incorporated-source `1740b2d` local Windows 11 frozen GUI/CLI, Start launch, active-update refusal, reinstall/uninstall and data preservation passed; recording retains earlier source evidence; distribution peer review/clean hosted package CI/merge pending |
| M2 | Identity/relations/revisions and lifecycle durable | SQLite constraints, restart, migration and crash boundary tests | A implementado y verificado automáticamente; PR21 merge53712c5 tras CI980/2/307,46 s y reviews. Ciclo de vida/borrado B pendiente; M2 abierto, sin certificación nueva |
| M2 · criterio 2 | Grabación, archivo y retry comparten persistencia de aplicación | CLI delega a SessionProcessor, artefactos y fallos compatibles | Consolidación verificada en focos de tarea 2; integración de catálogo implementada en 7bfd0cb con foco final 26; review técnica y CI final A980/2/307,46 s aceptados; B/M2 pendientes |
| M2 · criterios 6–7 | Segmentos nullable y pausa sin desplazar el WAV | Cinco recorridos con proveedor fake, procedencia y seek de marcas en WAV sintético | Verificados automáticamente en focos de tarea 2 y catálogo implementado en tarea 3; full4 y review técnica final aceptados; sin certificación de captura/proveedor |
| M2 · criterio 8 | Migración transaccional, backup administrado y rechazo de esquema futuro | FK/integrity, rollback de DDL y snapshot SQLite API con WAL comprometido | Verificado automáticamente en local: M2A-T1-LOCK-GREEN-20261105 y M2A-T1-FINAL-20261105, más CI final A980/2/307,46 s |
| M2 · criterio 14 | Busy tipado y acotado, sin trabajo de proveedor/archivos en transacción | Segunda conexión real con bloqueo de escritura y EXCLUSIVE sobre lecturas | Base verificada en tarea1 y publicación implementada en tarea 3, con focos de recuperación Busy; review técnica y CI final A980/2/307,46 s aceptados; B/M2 pendientes |
| M2 · criterio 15 | Suite completa en CI de Windows sin credencial real ni display | Corrida hospedada sobre el head exacto del PR M2 A | Verificado automáticamente para A por M2A-CI-FINAL-8a498e2-20261106:980 pasan/2 skips wheel/307,46 s, run42155415514; B/cierre M2 exige su propia CI final |
| M2 | Deletion clears managed memory without resurrection | DB/FTS/files/answers/backup inspection after restart | Open |
| M3 | Listing/filter/search and original moment navigation | End-to-end Qt selection and seek plus real WAV marker playback | Open |
| M3 | Large library remains responsive | 1,000 meetings/100,000 segments benchmark and event-loop check | Open |
| M4 | Multi-meeting answers/extractions with valid evidence | Source-resolution and semantic synthetic corpus, provider smoke | Open |
| M4 | Uncertainty/injection/cancellation/deletion remain honest | No-evidence, invalid-citation, concurrent delete/redaction tests | Open |
| M5 | Offline first run, settings, tray/startup and safe exit | Qt, process concurrency and Windows settings tests | Open |
| M5 | Portable data including optional audio and citations | Export/import on a fresh root, hashes and hostile ZIP tests | Open |
| M5 | Install/update/uninstall preserve user memory | N to N+1 installation and separate uninstall test | Open |
| All | PR groups reviewed and merged | PR URLs, reviewed heads, comment dispositions, CI and merge SHAs | M2–M5 en ejecución por ruling 20; cada uno de los nueve grupos requiere review independiente, Copilot recibido/respondido y CI Windows verde sobre el head exacto antes del merge; M2 A mergeado por PR21/53712c5 con todos los gates; los ocho grupos restantes pendientes |

## Current execution scope

**M2 A cerrado y mergeado**, implementado y verificado automáticamente, sin nueva
certificación real. PR[#21](https://github.com/srodriguez-lefebre/here/pull/21),
head8a498e207812592ff84a0acc5ba4e9f392e02373, merge53712c5beedafc2d40e2abf10c13362a1c3c9f78
el2026-11-06T03:22:06Z. M2 sigue abierto hasta B y sus criterios/version0.3.0.

| Corrida nombrada | Resultado observado | Alcance y límites |
|---|---|---|
| M2A-CI-FINAL-8a498e2-20261106 | 980 pasan/2 skips en307,46 s, run42155415514 | Suite completa Windows offscreen/key sintética/raíces temporales sobre8a exacto; skips son ambos wheel sin HERE_WHEEL_PYTHON, no pases |
| M2A-PACKAGE-FINAL-8a498e2-20261106 | Success, run42155415501 | Workflow Windows de bundle/installer sobre8a; no certifica instalación limpia ni upgrade real |
| M2A-COPILOT3-WHEEL-c59ae33 | Distribución4 pasan/7,51 s, incluidos CLI y GUI instalados | Fuente exacta byte-igual al merge; ejecución local distinta de CI, no aumenta el conteo980 |

Review independiente final sourcec59ae33/docs3fdbd05 APPROVED. Copilot5795325438
COMMENTED recibido2026-11-06T03:20:34Z sobre8a, sin issues abiertos/C5 resuelto;
las cinco observaciones tienen respuesta y sus hilos están resueltos. Root ejecutó
merge commit con --admin --match-head-commit tras CI verde exacto. Ledger completo
en DELIVERY; log real `.superpowers/m2a-ci-42155415514.log`. Esta documentación
inspecciona resultados existentes y no afirma una nueva ejecución de tests.

C5 actual implementado c59ae331ebf124400ef3e476e5f0fcc029e83999 desdee8bd181:
src5c66169c0a93a8f0606a011bd9570bbdd9137903 y testsfb958d021eaf7d149c00a3a1be174e36fbef2a1c.
R44 conserva metadata legada visible con aviso fijo no reintentable cuando falla
resolver ownership, sin leer material sensible ni inferir identidad; dueño real y
paths inseguros siguen bloqueados y materialize falla cerrado. Review final de
fuente/docs APPROVED; el CI exacto y merge A constan arriba. Las siguientes corridas
locales conservan su fuente y límites; sin certificación nueva ni cierre de M2.

| Corrida nombrada | Resultado observado | Alcance y límites |
|---|---|---|
| M2A-COPILOT3-RED-20261106 | 2 fallos/3 controles pasan en10,98 s; ambigüedad adicional1 fallo/2,95 s | Seis casos distintos sobre5b, tres RED/tres PASS en dos corridas; corrupción SQLite real None/opaco y origen ambiguo FK-válido. No fue un único run de seis |
| M2A-COPILOT3-GREEN-20261106 | 6 pasan/4,24 s | Metadata/bytes intactos, cero probes protegidos, stale materialize y guard real del adapter; sin catálogo y lock OS mapeado real como controles |
| M2A-COPILOT3-AFFECTED-20261106 | 169 pasan/114,05 s, cero skips | Diecisiete archivos de recuperación/ownership/controller/UI/capture faults; no suite completa |
| M2A-COPILOT3-WHEEL-c59ae33 | Probe65 fuentes iguales, no editable/schema2/FTS5; distribución4 pasan/7,51 s | XML4/0F0E0S/interno7,456; CLI3,996/GUI2,757/mutex0,007/path0,559. FK/integrity correctos informados por root; no instalación limpia |

Ruff/format2 limpios; mypy canónico5b/current79/79 en65 fuentes/20 archivos, sin
mensajes nuevos; la base global no está limpia. Probe de2026-11-06T03:08:14.816511Z:
`.superpowers/m2a-wheel-catalog-probe-c59ae33.json`; logs/XML
`.superpowers/m2a-copilot-3-wheel-smokes.log`/`.xml`. Wheel SHA256
`2c49dd9d896ce4b46d932bddc2f58cfb09380c7a22fe6d0d6a1f2e2f2c096de5`;
sdist `952b4706e290bc8e9134e61def5dadc09b341edf311ca136c46923ba8b7ee9a4`.
Docs leyó logs/probe y parseó XML; no ejecutó runtime. R43/R44 mantienen la suite
completa obligatoria en Windows CI del nuevo head antes del merge, sin full local7
duplicado. CI5b42151367541 pasa974/2 wheel skips/349,23 s y package42151367491 pasa:
históricos para C5. Copilot5795057762 sobre5b confirmó C4 y abrió C5/4488473222;
C5 fue respondido/publicado y recibió nueva review/CI exacto antes del merge,
como consta arriba. Los estados pendientes anteriores conservan alcance histórico.

Checkpoint histórico C4 56af525 desde 0a28842, src a17ec107c212d62e1984bb1cede094b579db5137 y
tests c6a29f96af9ffc1e0c488ca73a430ef9a99b86b6. RED 6 fallan/2 controles pasan/
6,69 s; GREEN 8/8,18 s, afectado 124/90,52 s sin skips. Ruff/format 2 limpios y
mypy D23/current 79/79 sin nuevos. Probe fresco no editable 65 archivos iguales,
schema 2/FTS5 y FK/integrity informados por root. Distribución4 pasan 7,44 s, XML 4/
0F/0E/0S/interno 7,314 s; wheel CLI 3,928/GUI 2,748, mutex 0,005 y path 0,570 pasan.
Wheel SHA256 `cdc99d7b0b76bdce2d8c606edb7adabcd335984be4fc0bce000690648a5bc37a`;
sdist `df30e3ffc9aa4e33d4b5b995f982d1c52097541ff1633b0035ee00074d686eb3`.
Logs `.superpowers/m2a-copilot-2-wheel-smokes.*`, probe 56af525 de 02:26:03.485121Z.
Review final APPROVED con docs58176ce; C4 reply4488439090/hilo resuelto y confirmado
por Copilot5795057762; suite5b completa pasa974/2/349,23 s. R43 conserva el gate de suite completa: Windows CI sobre
el head exacto nuevo antes del merge; no duplicar con local full6. Full5CDF y
CI_D23 (966 pasan/2 wheel skips/331,02 s,42147566672; package42147566683 pasa)
son históricos. Copilot 5794829768 sobre D23 resolvió C1–C3 y abrió C4, después
resuelto como se registra arriba. Sin cierre ni certificación.

Checkpoint histórico PR #21: fuente c3cdf08 integra C1/C2/C3 bajo R40, desde
61fb8e2, con src b465d3198840f96d5d15a85eeb45e4312e4c4dec y tests
293bc5038bae501dee07631704899e5f385e9025. RED independiente 5 fallos/3 controles
pasan/8 casos/26,91 s; GREEN 8 pasan 9,77 s; afectado 325 pasan/1 symlink skip/
141,36 s. Ruff/format 6 limpios, mypy comparativo339/current 79/79 sin nuevos.
Full 5 pasa 961/7/371,80 s; review acotada final pendiente. C1–C3 provisionalmente ADDRESSED por inspección de fuente, sin veredicto final aún.
La review/full4/CI previos conservan su fuente; no prueban el código nuevo.

Probe fresco c3cdf08 de 2026-11-06T01:47:42.753122Z: 65 módulos iguales, no editable,
schema 2/FTS5 y FK/integrity informados por root. Wheel SHA256
`acbc092001eda20bcf980bf9d74d2c53c01698322641e9f8a9d313c34efec051`, sdist
`f7d80d8612aa6e2de13662324c16f670baa5022a2ad4dfcd976df1e6847b33ab`.
Full 5 comenzó solo a 2026-11-06T01:47:43.289345Z, logs/XML
`.superpowers/m2a-integrated-final-5.*`: **961 pasan/7 skips/cero fallos en 371,80 s**.
Lectura independiente de log/XML: 968 casos/0F/0E/7S/interno 371,710 s. Ambos smokes
wheel CLI 3,747/GUI 2,978 s pasan sin failure/skip; root confirma exit0. Siete skips
son privilegios symlink, no aprobaciones. CI Windows339 run 42142394863:958 pasan/2 skips wheel/250,79 s;
package 42142395026 pasó sobre 339 solamente. Faltan respuestas a Copilot 5794475912,
nueva solicitud/recepción en head corregido, CI exacto y merge guardado.

La instrucción explícita del 2026-11-05 (ruling 20) autoriza implementar y entregar
M2–M5 completos en nueve grupos, empezando por M2 A. No altera las definiciones de
terminado ni convierte verificación automática local en certificación de hardware,
proveedor o instalación limpia. La prueba de dos horas permanece omitida por
ruling 5, nunca aprobada. Las exclusiones históricas de M1 conservan su procedencia;
ninguna observación corta certifica una ejecución larga.

Los checkpoints locales de tareas 1/2 no cierran M2. La implementación de tarea 3
aporta ingestión, FTS, revisiones inmutables e integración de productores. Su
primer candidato pasó la suite estable y recibió siete correcciones. Reviews
acotadas de rondas 1–3 marcan F1–F7/NB1/NB2 addressed; full histórico 8087e25 pasa
940/7. La review global única terminó changes requested por GR1–GR3. La tanda
R38 está integrada en 7bfd0cb, con foco final 26/34,46 s y probe wheel nuevo;
full 3 fallido por wrappers permanece histórico. Corrección test-only ef70f78
pasa foco13/24,82 s y full4:953/7/358,94 s; review final técnica APPROVED sobre
docs8845877, GR1–GR3 ADDRESSED. B/CI/merge del grupo siguen abiertos. Los smokes reales de proveedor
autorizados por ruling 19/20 usan solo audio y texto sintéticos; no se ejecutaron
en estos checkpoints. No se leyó el
`.env` real, la biblioteca del usuario ni grabaciones privadas. La matriz y las
corridas históricas de M1 mantienen sus resultados y límites originales.

## Corridas locales de M2 A — tarea 1, 2026-11-05

Entorno observado en los logs: Windows, CPython 3.13.15, pytest 9.1.1,
PySide6/Qt 6.11.2, `QT_QPA_PLATFORM=offscreen`, `OPENAI_API_KEY` sintética,
DBs y raíces temporales. Implementador y reviewer independiente son distintos;
esta documentación registra su evidencia, no una nueva ejecución del autor.
Fuente corregida de tarea 1: `9a80060` (`codex/m2-catalog`). El grupo aún no tiene
CI ni merge propios; el resultado local no se atribuye a un head futuro integrado.

| Corrida nombrada | Resultado observado | Alcance y límites |
|---|---|---|
| M2A-T1-RED-20261105 | Autor RED: 27 fallos en 0,64 s; confirmación del implementador: 27 fallos en 3,32 s | `here.memory` ausente; fallos de feature, sin errores de colección |
| M2A-T1-INITIAL-GREEN-20261105 | Foco: 26 pasan/1 skip en 2,30 s; suite completa: 768 pasan/9 skips en 174,94 s | Antes de la corrección de lecturas Busy; dos smokes de wheel omitidos y siete skips por symlinks |
| M2A-T1-LOCK-RED-20261105 | 8 fallos/9 deseleccionados en 2,51 s; foco completo: 8 fallos/26 pasan/1 skip en 18,45 s | EXCLUSIVE real en segunda conexión con journal DELETE; se filtraba `OperationalError` en migración y siete lectores |
| M2A-T1-LOCK-GREEN-20261105 | 34 pasan/1 skip en 4,70 s | Busy tipado, retorno acotado, recuperación al liberar lock e integridad; symlink omitido por privilegio |
| M2A-T1-FINAL-20261105 | 778 pasan/7 skips en 173,68 s | Suite completa corregida; `HERE_WHEEL_PYTHON` apunta a entorno aislado no editable y ambos smokes de wheel pasan; seis skips de symlink preexistentes más uno nuevo de memory |

Comandos de prueba usados por los autores, con las variables anteriores:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_memory_database.py tests/test_memory_repository.py -q --tb=line
.venv/Scripts/python.exe -m pytest tests/test_memory_repository.py -k exclusive_lock -q --tb=short
.venv/Scripts/python.exe -m pytest tests/test_memory_database.py tests/test_memory_repository.py -q -rs
.venv/Scripts/python.exe -m pytest -vv
```

Los reportes y logs inspeccionables viven en el ledger local del worktree del
grupo: `.superpowers/sdd/M2_CATALOG_PLAN/task-1-red-report.md`,
`task-1-root-red.log`, `task-1-full.log`, `task-1-fix-red-report.md`,
`task-1-fix-green.log` y `task-1-final-full.log`. La review inicial
`task-1-review-report.md` pidió corregir Busy; `task-1-fix-review-report.md` acepta
el arreglo y no deja hallazgos accionables de tarea 1. Lint y formato de los
archivos de tarea 1 pasan. El chequeo mypy normal encuentra el problema de layout
plano conocido. Un chequeo posterior con copias intactas de los cuatro módulos de
`memory` y el validador de rutas en un paquete temporal canónico devolvió
`Success: no issues found in 4 source files`, sin ignorar imports faltantes:
`mypy --follow-imports=silent --no-incremental -p here.memory`, con `MYPYPATH`
limitado a ese temporal. Log `task-1-mypy-layout-checked.log`; root retiró después
solo ese temporal propio verificado. Es un chequeo de tipos acotado a memory,
sin cambios de dependencias/producción y sin presentar el mypy global como limpio.

Esta evidencia demuestra contratos locales de DB, backups, esquema, DTOs y
repositorio. No demuestra todavía publicación de archivos tras process-kill,
ingestión completa, borrado de backups ni funcionamiento sin dependencia del
catálogo durante una captura. No hay certificación nueva ni cierre de M2.

## Corridas locales de M2 A — tarea 2, 2026-11-05

Fuente exacta revisada: `037fc036ef43467136bcf7b8e95c6f8395d1aa5e`, integrada
sin cambios a sus archivos como `1cdaadb`. La review independiente de SPEC y
calidad pasó sin hallazgos (`task-2-review-report.md`); el reviewer no repitió la
suite. La evidencia corresponde a la rama de tarea 2 sobre su base `878e7d4`, no
a una futura suite conjunta de las tres tareas del grupo.

| Corrida nombrada | Resultado observado | Alcance y límites |
|---|---|---|
| M2A-T2-RED-20261105 | 82 fallos/153 pasan/1 skip en 21,18 s | Faltaban segmentos de texto solo, procedencia temporal y raw separado; fallos de feature, sin errores de colección |
| M2A-T2-RAW-RED-20261105 | 2 fallos/112 deseleccionados en 2,12 s | Fallo de replace de metadata dejaba raw nuevo huérfano o reemplazaba bytes anteriores; ambos casos pasan en el foco final |
| M2A-T2-FOCUS-20261105 | 310 pasan/1 skip en 24,60 s | Cinco recorridos reales con proveedor fake, texto fuente/limpio, tiempos inválidos/unilaterales, rollback raw y seek de segundo fragmento sobre WAV tras pausa virtual de 300 s; skip de symlink |
| M2A-T2-FIRST-FULL-20261105 | 8 fallos/775 pasan/8 skips en 171,12 s | Ocho wrappers de faults rechazaron el nuevo keyword `raw_text`; se corrigió su forwarding sin cambiar fases de kill ni assertions |
| M2A-T2-CAPTURE-RECOVERY-20261105 | 68 pasan en 25,64 s | Incluye los cuatro kills reales y cuatro fronteras de crash sintético antes falladas; ningún skip |
| M2A-T2-FINAL-20261105 | 783 pasan/8 skips en 160,71 s | Windows, CPython 3.13.15, Qt offscreen, key sintética; seis skips de symlink y dos smokes no editables sin `HERE_WHEEL_PYTHON`; `pwsh` presente |

Comandos del implementador, con `QT_QPA_PLATFORM=offscreen` y clave sintética:

```powershell
uv run --no-sync pytest -q tests/test_transcription_service.py tests/test_application_processing.py tests/test_output_session_writer.py tests/test_segments.py tests/test_cli.py tests/test_desktop_cli_recovery.py --tb=short --show-capture=no
uv run --no-sync pytest -q --tb=short --show-capture=no
```

La evidencia inspeccionable del ledger local incluye `task-2-red-output.txt`,
`task-2-green-output.txt`, `task-2-full-first-output.txt`, `task-2-crash-output.txt`,
`task-2-full-output.txt`, `task-2-implementation-report.md` y
`task-2-review-report.md`. Ruff y formato pasan en los 13 archivos tocados.
Comparación mypy sobre los siete archivos de producción: 60 diagnósticos de base,
58 actuales, sin mensajes nuevos; no se presenta como una base mypy limpia.
`M2_CATALOG_EVIDENCE.json` conserva resultados y límites de ambas tareas.

`raw_transcript.txt` conserva el texto anterior al cleanup; el transcript de
presentación puede diferir. Los cinco recorridos lo publican junto a segmentos por
`SessionProcessor`, y la CLI ya no escribe artefactos directamente. Texto solo
retiene segmento sin tiempo/hablante inferidos; `timing_source` es opcional y
nullable en documentos históricos. La procedencia nueva es `provider` si queda
algún límite válido y `absent` si no queda ninguno; `imported_explicit` se conserva
en el contrato de metadata. El rollback por errores I/O restaura bytes raw, pero
no es una transacción de filesystem frente a muerte de proceso o escritores
concurrentes. Publicación de revisiones de catálogo y reconciliación son tarea 3.
No hay CI hospedado de M2 A, certificación nueva, cierre de grupo ni cierre de M2.

## Tarea 3 — candidato con suite local verde y review solicitando cambios

Fuente candidata en `m2-ingestion`, base `1cdaadb`: fixture
`efb73a92ea89bc53318ce1a811eaf61129a76fbf` y Task3
`6c422f0b8671d2f736645abb6835d351783cbcc7`. Los 27 hashes coinciden con
`task-3-candidate-files.json`, verificado de forma independiente por este autor
de docs sin ejecutar código. Commits locales solo para fijar FIX_BASE, todavía
sin integrar en root ni aprobar tarea/grupo. Los focos históricos siguientes
pertenecen a estados de fuente distintos y no se suman. La suite estable posterior
pasó, pero review solicita siete cambios; aún no hay CI/Copilot del grupo.

| RED o corrección | Resultado observado y alcance |
|---|---|
| Servicio/artefactos/controlador | 31 fallos/18 pasan en 2,50 s; root confirma en 4,25 s; faltaba `here.memory.service`, sin error de colección |
| CLI e integración inicial | Ocho fallos CLI en 5,94 s; conjunto root 39 fallos/18 pasan en 7,33 s |
| Propiedad/recovery y volumen | Siete fallos por feature ausente en 1,44 s; el caso de volumen remoto mapeado falló en 2,00 s por WAL en lugar de DELETE; conjunto ocho fallos en 1,21 s |
| Rebuild/archivo fallido | Dos fallos en 5,99 s en preparación (servicio ausente y archivo fallido sin catálogo); el implementador reprodujo dos brechas concretas en Green en 6,41 s |
| Privacidad | Dos intentos de acceso a credencial sintética detenidos antes de leer/probar, 2,58 s contra Green; implementador reproduce dos fallos en 3,37 s. No se leyó un secreto real |
| Warning UI | Tres fallos en 5,65 s por ausencia del campo aprobado `memory_warning`; UI real, sin error de colección/teardown |
| Procesamiento/owner y retry fallido | Cuatro fallos concretos en 9,78 s; implementador reproduce cuatro en 9,26 s: faltaba estado/propiedad real durante proveedor e intención durable de desenlace fallido |
| Retry legado | Un fallo concreto en 7,58 s: el retry explícito duplicaba UUID de origen v1 sin ID, después de preservar audio y metadata compatible |
| Fechas UTC | Dos fallos concretos en 2,46 s: instantes correctos todavía guardaban offset -03:00; no era error de instante ni de bytes fuente |
| Compatibilidad de identidad nueva | El full diagnóstico encontró expectativa vieja `meeting_id=None` en un nuevo archivo externo; el autor cambió solo ese caso a UUID canónico estable y conservó identidad exacta de sesiones existentes; tres variantes pasan en 10,99 s |
| Limpieza desconocida | Un fallo concreto en 2,43 s: reconcile borraba un archivo no reconocido dentro de staging UUID. Registro de plan/limpieza diferida incluidos luego en foco 122/108,36 s y suite del candidato |
| Roles de receipt/pending/inventario y estados | Tres fallos concretos en 5,00 s y uno de `states/.env` en 3,08 s, detenidos por guards sintéticos antes de leer; roles y nombres generados corregidos antes del snapshot revisado |

| Foco GREEN intermedio | Resultado observado |
|---|---|
| Servicio/artefactos/controlador inicial | 49 pasan en 28,07 s |
| CLI y procesamiento | 50 pasan en 36,99 s |
| Propiedad/recovery y política de almacenamiento | Ocho pasan en 22,78 s |
| Rebuild/servicio/artefactos/propiedad | 34 pasan en 53,42 s |
| Warning UI y controlador | 27 pasan en 9,61 s |
| Privacidad/preflight y capture recovery | 70 pasan en 86,61 s |
| Ocho archivos de memoria combinados | 48 pasan en 51,99 s |
| Procesamiento/propiedad/recovery/CLI/controlador | 85 pasan en 79,17 s |
| Retry legado y procesamiento | Cinco pasan en 14,10 s |
| Fechas UTC | Dos pasan en 1,83 s |
| Foco estable tras registro de staging | 122 pasan en 108,36 s |
| Guards de roles/estado y recovery | 46 pasan en 66,62 s; foco anterior relacionado 31 pasan en 34,78 s |

Los cinco cortes de publicación ejecutan hijos reales con `os._exit(73)` en las
fronteras copy/rename/activation/commit/completion. Propiedad usa procesos/locks
reales y muerte del dueño; Busy usa otra conexión SQLite real y los paths incluyen
junction temporal real. No son certificación de micrófono, proveedor o instalación.
Los reportes `task-3-*-red-report.md`, `task-3-identity-compatibility-report.md` y
el draft `task-3-implementation-report.md` conservan IDs, comandos y límites de
cada corrida. El autor de docs leyó esa evidencia, sin ejecutar tests nuevos.

### Fallos diagnósticos de captura y arnés sintético

| Corrida | Resultado y límite |
|---|---|
| Full diagnóstico inicial | 34 fallos/822 pasan/9 skips en 363,35 s, con fuente cambiando y otros checks superpuestos. Catorce fallos expusieron catálogo inicializado antes del preflight rechazado; luego se corrigieron sin quitar assertions. No es el full final |
| Aislado diagnóstico posterior | 21 fallos/12 pasan en 75,37 s; registro de cada ID/traceback en `task3-isolated-diagnostic.txt` |
| Candidato quieto a 21:06:32 UTC | 17 fallos/16 pasan en 87,56 s, mayormente lease de apertura y un journal replace WinError 5; no se atribuyen todos a carga |
| Base con intérprete y paquete propios | 15 fallos/18 pasan en 87,08 s; demuestra presencia anterior de fallos del arnés, sin probar una causa global |
| Comparación aislada de un caso previa | Base 7,41 s y candidato 7,68 s pasan, pero el hijo aún resolvía el paquete Green instalado: no era aislamiento completo de fuentes. Un `[read]` había fallado una vez |
| Foco del arnés corregido | Cuatro pasan/29 deseleccionados en 19,71 s, sin otros workloads pesados |
| Aislado completo del arnés corregido | 33 pasan en 126,10 s, sin skips y sin otros workloads pesados |

H1 se observó en un lanzamiento fake sano: el lease original de 1,5 s expiró en
imports, antes de request, journal o dispositivo. H2 reprodujo el mecanismo de
sharing de Windows en un journal desechable: un lector abierto causó WinError 5 y
al cerrarlo el mismo writer funcionó. No se trazó ese solapamiento en el fallo
original, por lo que H2 no se presenta como su causa demostrada.

El fixture espera una confirmación real de bootstrap con presupuesto sintético
acotado de 10 s antes de devolver el helper. No emite un `ready` de protocolo ni
cambia el timeout de apertura/lectura/cierre/watchdog del producto. El progreso de
los dos tests de backpressure se observa tras checkpoints reales mediante marcas
independientes; el journal final se lee después del reap del hijo. No cambió ningún
deadline ni escenario de stall del producto. Review independiente del fixture por
`capture_diagnostics`: SPEC/calidad PASS sin hallazgos, SHA256
`665181979ffd5b0d354c473de9a6e1f12fbc3209f42561e1af5f99b0dad67ca1`.
El reviewer no repitió suites; su aceptación cubre solo ese fixture, no tarea 3.

Antes de la última enmienda de staging, Ruff/format pasaron sobre 24 archivos y
el mypy con layout temporal canónico comparó 79 diagnósticos de base/79 actuales
(62/64 módulos) sin errores nuevos en archivos tocados. El raw mypy plano reportó
232 errores/43 archivos; no se presenta una base limpia. Después de las enmiendas
de staging/roles, Ruff/format/diff pasaron sobre 27 archivos y mypy canónico quedó
79/79, sin mensajes nuevos en archivos tocados. Los guards de roles antes de
hash/read y de nombre de estado antes de lectura están en el candidato revisado;
la observación anterior de ausencia de esos guards quedó superada.

### Suite estable y review inicial — checkpoint histórico, superado por ronda 1

| Corrida nombrada | Resultado y alcance |
|---|---|
| M2A-T3-STABLE-FULL1-20261105 | Seis fallos/877 pasan/9 skips en 486,12 s, 892 casos en XML (seis failures, cero errors, nueve skipped). Fallan solo `test_postcommit_audio_cleanup_fault_preserves_committed_outcome`, read/missing_metadata/invalid_json × completed/failed, por contador de dos lecturas en lugar de una |
| M2A-T3-CLEANUP-AUTHOR-20261105 | Diez pasan en 16,76 s: el autor RED acota inyección/contador al delegado real `_cleanup_replaced_audio`, preservando exact-one, corrupción JSON real, resultado, causa original y bytes/rutas |
| M2A-T3-CLEANUP-GREEN-20261105 | Los diez casos transferidos pasan en 17,10 s; ahora se ejecutan también sus assertions de bytes posteriores al contador |
| M2A-T3-CANDIDATE-FULL-20261105 | 883 pasan/9 skips en 598,56 s, 892 casos en XML (cero failures/errors, nueve skipped), ejecutado solo con fuente congelada. No equivale a aprobar siete contratos estáticos abiertos |

Logs y XML permanecen separados en el worktree Green:
`.superpowers/task3-full-final.txt`/`.xml` y
`.superpowers/task3-full-final-2.txt`/`.xml`. Las duraciones anteriores son las
resumidas por pytest; el atributo interno de tiempo JUnit es 485,253 s y 592,473 s,
respectivamente. Los nueve skips son siete de privilegio de symlink y dos smokes
del wheel no editable sin `HERE_WHEEL_PYTHON`; no se afirma que CI vaya a omitirlos
ni que ese wheel haya sido certificado en esta corrida.

La corrección de instrumentación está documentada en
`task-3-cleanup-instrumentation-report.md`: publicación de catálogo relee metadata
legítimamente después del cleanup; la inyección global antigua contaba también ese
consumidor independiente. No se cambió código productivo, deadline ni assertion
para arreglar el contador. El primer full fallido y sus límites se conservan.

La review estática inicial `task-3-review-report.md` contra los 27 hashes
exactos solicitó cambios de SPEC y calidad; el reviewer no ejecutó tests. La tabla
conserva el estado de ese checkpoint; la re-review vigente está abajo:

| ID | Prioridad | Contrato abierto en la review inicial |
|---|---|---|
| F1 | P1 | Relación explícita e inmutable revisión/audio, incluyendo A→B→A con assets reutilizados; ruling 31 autoriza migración 2/recibo v2/show/rebuild, todavía pendientes |
| F2 | P2 | `here trans` sobre sesión v1 importada con ID None no debe inventar otra identidad ni reescribir la opcional; el test GUI de legacy no cubría esa ruta CLI |
| F3 | P2 | Retry explícito con evidencia idéntica debe completar estado/revisión; no quedarse processing y luego interrupted |
| F4 | P2 | Fallo de housekeeping posterior al commit no puede reportar rechazo precommit cuando la revisión nueva ya está activa |
| F5 | P2 | Adquirir ownership antes de mutación/proveedor y excluir/revalidar dueño vivo en discovery/materialize antes de recuperar o limpiar |
| F6 | P2 | Marcadores JSON array/null o revisión no-string se rechazan sin abortar recovery de vecinos válidos |
| F7 | P2 | Categorías tipadas separan entrada inválida (2), conflicto (3) e I/O/integridad (4), sin inferirlas del texto de warnings |

F5 incluye la ruta concreta segunda instancia → discovery → materialize →
persistencia/cleanup. La discovery por sí sola no cambia la fila de catálogo; no
se observó ni se afirma borrado real de audio. Ese alcance pertenece a propiedad
de M2 A, no se difiere al borrado/edición de B. Correcciones: RED independiente,
GREEN, re-review y suite de fuente corregida; luego review del grupo y gates de
GitHub. En ese checkpoint los commits del candidato no estaban integrados en root;
el resultado de la integración posterior se registra abajo.

| Nivel | Estado del checkpoint histórico inicial |
|---|---|
| 1 · Implementado | Existe candidato local `6c422f0` con suite pasando; la tarea está sin aprobar por siete contratos abiertos |
| 2 · Verificado automáticamente | Suite local 883/9 y focos con datos sintéticos; siete omisiones de symlink explícitas y wheel opt-in ausente; hosted CI del grupo pendiente |
| 3 · Certificado | Sin nueva certificación de hardware, proveedor, instalación limpia o wheel del candidato |

M2 A y M2 siguen abiertos. Versión vigente 0.2.0; 0.3.0 solo al cerrar B/M2.
El PLAN de B se ajusta en su PR de documentación antes de implementar, conservando
contratos y los gates estrictos del grupo A.

### Histórico de ronda 1 — NB1 y cinco fallos en ese checkpoint

Fuente Green revisada `c28472b5bd651f4d7f7fe2f40449f5a23e325f67`, fixes
`16263c4` + `c28472b`, integrada en root
`1f1bc88513840b84c388a61d9c077907cf2fba2e`. Los árboles Git src
`780529d42936c979740443c51f1022fd2bb99a5e` y tests
`d226358ec038837dba6e40790c77eac74e74b538` son idénticos entre ambos, confirmado
por lectura Git de este autor. Source/config/scripts/workflows no tienen diff.
Los 40 nuevos casos RED pasan juntos en 69,70 s; Ruff/format de 20 archivos
limpios y mypy 79/79/sin mensajes nuevos son evidencia del implementador.

`task-3-fix-1-review-report.md` es re-review estática independiente: originales
F1–F7 **ADDRESSED**, pero **Important NB1/P2 sigue abierto** y bloquea aceptación.
Controller + SessionProcessor pueden publicar el mismo retry comprometido otra
vez; el marcador retenido tras commit tampoco distingue replay de un intento
nuevo, queda con parent obsoleto y warning/conflicto. RED de ronda 2 en
`tests/test_memory_publication_replay.py` está en preparación/ejecución, sin
resultado entregado en este checkpoint. No se acepta parent arbitrario obsoleto,
ni se omite la revisión nueva de un retry idéntico ni se borra intención incierta sin
prueba durable para resolverlo.

`M2A-INTEGRATED-FULL1-20261105`: **5 fallos, 920 pasan, 7 skips en 553,39 s**.
Log/XML `.superpowers/m2a-integrated-final-1.*`; XML 932 casos, cinco failures,
cero errors, siete skipped (tiempo interno JUnit 552,876 s). Ambos smokes de wheel
pasaron. Los siete skips requieren privilegio de symlink; no se promete cero skips
en CI ni se presenta esta suite fallida como verde.

| Fallo observado | Estado |
|---|---|
| `test_failed_manifest_copies_with_same_uuid_are_both_discoverable` | Snapshot incluye nuevos `.capture_owners`/`.owners`; diagnóstico/fix pendiente |
| `test_retry_copied_session_preserves_independent_capture[canonical-interrupted]` | Snapshot incluye lease nuevo; diagnóstico/fix pendiente |
| `test_retry_copied_session_preserves_independent_capture[canonical-failed]` | Snapshot incluye lease nuevo; diagnóstico/fix pendiente |
| `test_trans_command_completes_failed_session_audio` | Stub sf.info sin subtype; exit 1 observado, diagnóstico/fix pendiente |
| `test_legacy_audio_only_session_still_retries` | Retry legacy solo-audio exige session.json ausente; compatibilidad pendiente de diagnóstico/fix |

No se omiten casos, relajan assertions ni declaran como simple ruido. La review de
NB1, estos cinco fallos, la suite corregida y la review del grupo son gates antes
del PR/Copilot/CI/merge. La re-review de originales addressed no aprueba tarea/grupo.

### Wheel y source histórico de 1f1bc88

Wheel y sdist frescos construidos con CPython 3.13.15. Probe
`M2A-INTEGRATED-WHEEL-20261105`, off-repo con `-I`, entorno no editable:
65 archivos Python byte-idénticos al source, versión 0.2.0, schema 2,
FTS5 disponible y `foreign_key_check`/`integrity_check` correctos.
Evidencia `.superpowers/m2a-wheel-catalog-probe.json` y script correspondiente;
este autor los inspeccionó sin ejecutar imports, tests o builds.

| Artefacto | SHA-256 |
|---|---|
| `here-0.2.0-py3-none-any.whl` | `bb7a49a9f5b5c6a6b25f48f39b036badc489bee3870c69b0024af8f0d61dd7bd` |
| `here-0.2.0.tar.gz` | `8ebd3a3a44d48df68d94dcb8748adf8af52dfde334465bc2bcecfa0d95513358` |

Es verificación automática local de distribución/importación/catálogo, sin audio
hardware ni proveedor y sin certificar instalación limpia o M2. No convierte en
éxito la suite integrada fallida.

| Nivel en el checkpoint histórico de ronda 1 | Resultado |
|---|---|
| 1 · Implementación | Código de ronda 1 integrado y originales addressed; tarea completa no aprobada por NB1 y cinco fallos |
| 2 · Verificación automática | Foco 40 pasan, wheel/source verificables y full con 920 pasan/5 fallos/7 skips; hosted CI pendiente |
| 3 · Certificación | Sin nueva certificación real de hardware, proveedor o instalador/upgrade limpio |

Ruling 35 requiere cerrar A con todos los gates antes de crear dos chats de
proyecto M2 B/M3 A. Bases de merge, IDs, ramas/worktrees y responsables futuros
siguen pendientes en STATUS; no se creó ni inició ningún frente. Los PRs docs de
contratos necesarios preceden al despacho, y los criterios/privacidad/calidad de
cada hito no cambian.

### Checkpoint histórico 8087e25 — rondas 2/3 y full integrado 2

Root `8087e252112fd2e6de1efeff1d0f3f531d4be77d` equivale a Green
`4a3cb7ced77a73046098e20cb34aeb00af0e6f61`; src tree
`948efa342c9569ca751cffb4718d82f3dd872de6`, tests tree
`30c62e6f7e9e0a818e125a6c7762db91335b47f1`. Lectura Git independiente confirma
árboles iguales y diff source/config/scripts/workflows vacío.

R36/R37 implementados: marker 2 con token/source_kind; receipt 3 con prueba exacta
(token/padre/origen/huella/revisión comprometida) y restauración de operaciones
completas durante rebuild, manteniendo receipt 2 compatible sin inventar prueba.
El controller consume una vez el handoff privado real; durable replay independiente
cubre crash real tras commit, con evidencia igual/cambiada y DB presente/reconstruida.
Reserva audio-only canónica con origen revision NULL/cero revisiones históricas y
source_kind durable mantiene fechas captura NULL, también tras retry/rebuild.
Schema DB 2, procesos/protocolos/deadlines/dependencias/versión sin cambios.

`task-3-fix-2-review-report.md` marca NB1 y regresiones de integración ADDRESSED.
Foco congelado: 61 pasan/68,06 s; controller/procesamiento/UI: 69/23,02 s; Ruff/format de 10 archivos
limpios, mypy 79/79/sin nuevos. Los snapshots no se relajaron: inspección metadata-only
no crea locks, materialización con journal adquiere/retiene lease antes de mutar.
El stub CLI fue reemplazado por WAV PCM16 real por su autor independiente, sin
fallback artificial en producción. Full 1 fallido permanece histórico arriba.

El foco intermedio de replay dio 16 pasan/1 falla en 315,60 s por timeout de hijo de
30 s antes de prueba/replay. Mismo caso sin cambios pasó aislado en 5,02 s y en foco
final de 61 casos (call 2,72 s). Causa no probada; no cambió timeout ni assertion, ni se
presenta ese fallo como prueba de un mecanismo identificado.

`task-3-fix-3-review-report.md`: NB2 ADDRESSED, sin nuevo Critical/Important en su
diff. Guard UUID en rama sin metadata precede construir/sondear journal; manifiesto
existente conserva identidad opcional/opaca. RED: 2 fallan/1 control pasa/2,45 s;
GREEN: 3/2,40 s y foco de 27/20,86 s. Ruff/format de 2 archivos limpios, mypy 79/79/sin nuevos.
Reviews estáticas independientes inspeccionan evidencia, sin rerun del reviewer.

**M2A-INTEGRATED-FULL2-8087e25: 940 pasan, 7 skips, cero fallos en 325,09 s**.
Inició solo a 2026-11-06T00:01:58Z sobre fuente 8087e25, con wheel no editable nuevo.
Log/XML `.superpowers/m2a-integrated-final-2.txt`/`.xml`: 947 casos, cero failures/
errors, siete skipped, tiempo interno JUnit 325,029 s. Ambos smokes wheel CLI/GUI
pasan; siete skips requieren privilegio symlink. Lectura independiente de XML y
summary confirma resultados; no certifica hardware/proveedor ni promete cero skips CI.

Wheel/sdist CPython 3.13.15 nuevos, probe
`.superpowers/m2a-wheel-catalog-probe-8087e25.json`: 65 archivos Python byte-idénticos,
off-repo `-I`/no editable, schema 2/FTS5/foreign_key_check/integrity_check correctos.
Los probes de 1f1 y 2a3 permanecen históricos con sus hashes propios.

| Artefacto de 8087e25 | SHA-256 |
|---|---|
| here-0.2.0-py3-none-any.whl | `4f37ca61d50df87e588ecdd1c2e9221690fd3aa9472dc33081c4116600147ad8` |
| here-0.2.0.tar.gz | `b5110b1dd0c78c485e52a738e147f21a5a64e6dfaa5aaf515ae68b8d10de5271` |

Review de todo A por `catalog_group_review` en curso contra 763a891..8087e25.
Tres Important provisionales: metadata completed puede comprometerse antes del primer
memory_pending marker, dejando un kill sin replay y estado interrupted; rebuild
puede llamar `.get` sobre receipt JSON array y abortar vecinos válidos; cancelación
aceptada durante persistencia del fallo de captura deja UI/manifest cancelled,
catálogo failed y conflicto/marker atascado. `group-review-report.md` registra
GR1/GR2/GR3 Important/P2 **OPEN**, source verdict **changes requested**. El estado
global sigue provisional hasta inspeccionar el suplemento docs comprometido; sin
fix de esos tres caminos. El full verde no los resuelve: faltan
RED/GREEN/re-review y verificación posterior, aceptación del
grupo y todos los gates de PR/Copilot/WindowsCI/merge.

| Nivel en ese checkpoint histórico | Resultado y límite |
|---|---|
| 1 · Implementado | Fuente 8087e25 y fixes acotados con suite pasando; tarea/grupo aún sin aceptar por review global pendiente |
| 2 · Verificado automáticamente | Focos 61/69/3/27, full 940/7 y wheel exacto; evidencia local sintética, hosted CI pendiente |
| 3 · Certificado | Sin nueva certificación hardware/proveedor/instalador o upgrade limpio |

R35 conserva prioridad: cerrar/mergear A y PRs docs de contratos antes de crear los
dos chats de proyecto B/M3 A. IDs/ramas/worktrees/base mergeada/responsables pendientes;
ninguno creado. Versión 0.2.0; meta final 1.0.0 y requisitos de calidad/privacidad intactos.

### Checkpoint histórico 7bfd0cb — tanda única de fixes GR1–GR3

Root `7bfd0cbf477a42aacc44b83b3d2da1e438801756` integra Green
`ec55b9564d3af01247d3367675a340a361f6b6ef`. Lectura Git independiente confirma
src tree `2b4036f7deb905efb0a8836f378d267a066564b4` y tests tree
`a4310416bd10dc922b49e010d8495a98bfdfd585` iguales. La review global única de
763a891..8087e25 y suplemento e00dd08 terminó **changes requested** para GR1–GR3;
GD1/GD2 Minor se corrigieron en7be93af. La misma review acotada final de tanda,
fixture y docs8845877 es **APPROVED**, GR1–GR3/GD1/GD2 **ADDRESSED**, sin nuevo
Critical/Important ni defecto fuente accionable. Tarea3 aceptada técnicamente,
grupo/hito aún pendientes de gates y alcance restantes. Reporte `group-fix-review-report.md`.

R38 implementado: marker 3 con token/padre/source kind/intent y hashes exactos del
manifiesto prospectivo y artefactos con roles validados se persiste antes del
replace final de metadata. Evidencia final ausente/distinta o manifest pending
se conserva con aviso, sin adoptar resultado anterior ni llamar al proveedor;
el aviso pending aparece en `RecoveryCandidate.error_summary`. El controller real
difiere la primera publicación de fallo de captura hasta sellar cancelación
aceptada, conservando una identidad/revisión parcial y audio/causa originales.
Receipts malformados se retienen y avisan por meeting sin abortar vecinos válidos.
Readers marker 1/2, receipts 2/3, schema 2, privacidad, conflictos y fechas NULL de
audio-only se conservan; no cambia proceso/deadline/dependencia/versión.

| Corrida local | Resultado real | Límite |
|---|---|---|
| RED independiente 8087e25, cuatro logs secuenciales | 11 fallan/1 control pasa, 12 casos; 27,48 s sumados | Control13 posterior autorado sin ejecución baseline |
| Primer GREEN, 12 casos | 12 pasan/21,67 s | Antes del control adicional |
| Compatibilidad afectada, 21 archivos | 404 pasan/1 symlink skip/159,03 s | Antes únicamente del último fix de aviso pending-v3; no ejecución post-fix |
| Control extra sobre candidato antes del fix de aviso | 1 falla/5 deselected/3,93 s | Falta aviso visible; R1/interrupted/bytes/marker preservados |
| Foco final de grupo y recovery | 26 pasan/34,46 s | Todos los 13 casos de grupo más malformed-marker/legacy ownership/NB2; no suite completa |

Logs `.superpowers/group-fix-focus.txt`, `group-fix-compatibility.txt`,
`group-fix-seal-warning-red.txt`, `group-fix-final-focus.txt`; reportes retenidos
`group-red-report.md` y `group-fix-implementation-report.md` en el ledger local.
Lectura de summary RED/GREEN confirma resultados; docs no ejecutó tests/imports.
Ruff/format de nueve archivos pasan; mypy canónico baseline 8087/current: 79 errores/
20 archivos/65 fuentes en ambos, mensajes nuevos `{}`. Es no regresión comparativa,
no una base mypy global limpia.

Wheel/sdist nuevos CPython3.13.15 y probe off-repo `-I` no editable en
`.superpowers/m2a-wheel-catalog-probe-7bfd0cb.json`, observado
2026-11-06T00:43:49.472905Z: 65 archivos Python byte-idénticos, schema 2/FTS5;
root informa foreign_key_check vacío/integrity_check=ok. Lectura del probe confirma
head, hashes, formato no editable y ausencia de hardware/proveedor.

| Artefacto de 7bfd0cb | SHA-256 |
|---|---|
| here-0.2.0-py3-none-any.whl | `e2f9c3ff09a5d5b41fcec9c35c1fab149294e9901f276e042bd2ce0bd12b0827` |
| here-0.2.0.tar.gz | `499d9249202451e95dafb3718dd78f2004bc825659f5810659f401a5707e92ee` |

**Full 3: 8 fallan/945 pasan/7 skips en 364,23 s**, iniciado solo a
2026-11-06T00:44:04.5472024Z con wheel nuevo; logs/XML
`.superpowers/m2a-integrated-final-3.txt`/`.xml`. Lectura independiente confirma
XML 960 casos/8 failures/0 errors/7 skipped/interno 362,276 s; los dos casos de
wheel cli_main/gui_main no tienen failure/skip y pasan (3,593 s/2,747 s).

Los ocho fallos son TypeError por `before_metadata_commit` no aceptado por dos
wrappers de `tests/test_capture_recovery.py`, en el hijo real y `publish_at`
simulado. Afectan fases reales pending/final_before/final_after/remove y simuladas
pending_before/pending_after/final_before/final_after. El autor RED corrige solo
forwarding del keyword opcional en ambos wrappers conservando assertions y cortes
de crash. Corrección autor `c3a3ccbcf2928502cbf0d0881fdfee3d7f86d84a` integrada en
root `ef70f78cfc9c8f2910e77b2b2818ce2e6b4d71ae`, solo tests; foco 13 (6 reales/
7 simulados) pasa en24,82 s, leído en log real. Autor compara 114 assertions
idénticas por AST; lectura del diff confirma wrappers y cortes intactos. SHA del
archivo `5E542C069A811CC3F209B55815EC52B5D953ED5F977699AF18D7C86B9582E134`;
reportes `capture-fixture-forwarding-report.md`/`capture-fixture-forwarding-focus.txt`.
Source tree sigue `2b4036f7deb905efb0a8836f378d267a066564b4`; tests tree nuevo
`30c88848b3cf42c9ce4443328e3c8e0100e2af44`. Full 4 empezó solo a
2026-11-06T00:54:42.9822634Z, log/XML `.superpowers/m2a-integrated-final-4.*`;
**953 pasan/7 skips/cero fallos en 358,94 s**. Lectura independiente de log/XML:
960 casos/0 failures/0 errors/7 skipped/interno 358,621 s; ambos wheel CLI/GUI
pasan sin failure/skip (3,018 s/2,634 s). Root confirma exit 0. Los siete skips son
los mismos privilegios symlink, nunca tests aprobados. No cambia producto; el wheel/probe 7bfd sigue
siendo exacto para fuente sin cambios. El full anterior940/7 no prueba este head.

| Nivel vigente | Resultado y límite |
|---|---|
| 1 · Implementado | Tanda R38 fuente7bfd y fixtureef70 con full4 pasando; tarea3 aceptada por review técnica final, GR1–GR3 ADDRESSED; grupo/merge pendientes |
| 2 · Verificado automáticamente | RED/GREEN/foco 26, fixture 13, static sin nuevos errores, wheel/probe exacto y full 4:953/7/358,94 s con ambos smokes pasando; full 3 fallido conservado; CI pendiente; datos sintéticos |
| 3 · Certificado | Sin nueva certificación hardware/proveedor/instalador/upgrade limpio; dos horas omitida por ruling 5 |

R35 exige merge A y PRs docs de contratos antes de chats B/M3 A; siguen sin crear.
Versión 0.2.0, criterios, calidad/privacidad y meta 1.0.0 intactos. Copilot sobre head
actual, CI Windows exacto y merge guardado pendientes. Todas las suites fallidas
anteriores y el timeout de causa no probada conservan fuentes y límites originales.

## Runs

Sanitized, versioned observations from the real synthetic-provider runs are in
[`M1_PROVIDER_EVIDENCE.json`](M1_PROVIDER_EVIDENCE.json). Only authored fixture
content, source formats, provider segments and audio hashes are included; no local
session paths, credentials or private recordings are published.

- 2026-10-03: original main `2bdb6ef`, 163 tests passed (Python environment in original
  checkout), and fresh managed worktree baseline 163 passed (Python 3.14.7, 42.19 s).
- Windows default device enumeration: microphone `Microphone (3- Arctis Nova 7P)`,
  44100 Hz, two channels; loopback `Headphones (3- Arctis Nova 7P) [Loopback]`,
  48000 Hz, two channels. Enumeration only; no claim of captured content.
- 2026-10-03 01:44 America/Montevideo: configured real provider, model
  `gpt-4o-transcribe-diarize`, accepted a generated Windows TTS fixture (~11 seconds).
  Production SessionProcessor preserved audio, four correct source utterances, four
  provider timestamp ranges and speaker A with request/chunk scope in `segments.json`.
  Evidence is local ignored `.superpowers/provider-smoke/result.json` plus session
  `20261003_014354`. No private recording was uploaded; no credential was displayed or
  written to evidence. This verifies single-request provider integration, not speaker
  identity across requests or full long-duration acceptance.
- 2026-10-03 02:05 America/Montevideo: alternating Windows David/Zira voices in two
  synthetic stereo sources at 44,100/48,000 Hz (26.159 seconds) passed the actual
  mixing, normalization, provider and artifact path. Seven provider segments retain
  timestamps and alternating A/B speakers in request scope; source format/frame
  provenance and normalized audio are preserved. Local evidence:
  `.superpowers/provider-multisource/result.json`, session `20261003_020516`.
  A spurious quote in the first utterance is preserved as original provider output.
  The first returned start precedes its synthetic source offset by 300 ms; later
  utterance starts are within 40 ms. This measures provider estimates, not an exact
  acoustic alignment guarantee, physical capture or long-meeting certification.
- Capture correctness suite at review-fix commit `4860095`: 211 passed, including
  ten regressions for the two Copilot findings. Changed-file Ruff and diff checks pass.
  Hosted Windows/Ubuntu tests and wheel/source builds passed at CI correction head
  `545174e`, run `37099618688`.
- Hosted review-fix CI at `240858e`, run `37103530702`: 266 tests passed in Windows
  (8.35 s) and Ubuntu (7.71 s), including the locally skipped symlink test; wheel and
  source builds passed. This is evidence for that immutable head, not later commits.
- 2026-10-03 05:37 UTC, code `5ec4e92`: short actual default-device capture with an
  authored TTS fixture on loopback, pause/resume and stop. Wall duration 8.281 s,
  pause 2.000 s, stop latency 0.107 s. Microphone: 271,360 frames at 44,100 Hz,
  stereo (6.153 s); system loopback: 295,936 frames at 48,000 Hz, stereo (6.165 s).
  Persisted, returned and live frame counts matched for both sources; nonzero loopback
  signal observed. No provider call; probe-owned raw audio was removed. The original
  probe used raw integer samples for peak, so no calibrated amplitude/clipping claim
  is made. This verifies a short device/control path, not long capture, journal recovery
  or installed-runtime behavior.
- Corrective code `7ed996a`: one full local offscreen suite, 385 passed/6 skipped in
  10.62 s. New tests reproduce external-entry recovery routes, partial output writes
  and reversed timing across aliases/objects/typed serialization. Real hardlinks and
  junctions passed; six symlink cases require local Windows privileges. Hosted checks
  for the final PR head are recorded in the PR and evaluated before merge.

- First M1 capture group merged 2026-10-03 07:52:15 UTC: PR #12, exact reviewed
  head `3162a74`, merge `ca07aed`. Current-head CI `37107198007` passed 469 tests
  on Windows (14.24 s, no skips), 467 on Ubuntu (7.97 s, two Windows-only junction
  skips), and wheel/source builds on both. All six locally skipped symlink cases
  ran successfully on both hosts. Local final-code suite: 463 passed/6 skipped in
  12.98 s; independent review accepted the two latest corrections. Final Copilot
  review `5399598693` confirms the fixes and records the legacy private CLI helper
  limitation assigned to desktop Task 1; no full-M1 completion is claimed. Details:
  [`M1_CAPTURE_EVIDENCE.json`](M1_CAPTURE_EVIDENCE.json). Short physical observations
  retain their original code provenance in
  [`M1_NATIVE_EVIDENCE.json`](M1_NATIVE_EVIDENCE.json).

## Measurement decisions

Desktop source `76d317a` passed 558 tests with six existing local symlink-privilege
skips (53.72 s), separate Task 1/2 independent reviews and a bounded native Qt check.
The native check observed actual 44.1/48 kHz default-device names, asynchronous
three-second quiet microphone and authored-tone loopback diagnostics, credential-free
local recovery and a paused save-exit with intentionally injected local transcription
failure. Audio remained locally retryable until probe-owned cleanup; no provider call
was issued. The maximum Qt heartbeat gap was 49.6 ms. This is source-runtime evidence,
not frozen or long-provider acceptance; details and limits are in
[`M1_DESKTOP_EVIDENCE.json`](M1_DESKTOP_EVIDENCE.json).

Whole-branch corrective source `490b77c` passed 569 tests with six existing local
symlink-privilege skips (56.92 s) and scoped independent acceptance. A native Qt
authored-WAV probe confirmed recovery when the normalized file is already absent,
retained choices after missing-key failure, and an explicit retry after editing the
effective configuration in the same window. It rebuilt 240 frames while preserving
identity/selected filename/raw bytes; no physical capture or provider was used.


PR [#13](https://github.com/srodriguez-lefebre/here/pull/13) is open. Initial hosted
checks at `ef7010e` failed: Ubuntu exposed eager PortAudio initialization in pure
subprocess imports and a Windows-only path assumption in a test; Windows reported a
native Qt teardown exception whose cause remains unconfirmed. Commits `cc947e0` and
`98e19b1` defer unrelated Linux hardware loading, make platform tests explicit, and
address Copilot review `5400263899`/inline `4172753832`: retain the original child
capture error once instead of obscuring it with a generic parent error. Real child
failure/timeout/death regressions passed. The final committed-source suite on actual
Python 3.13.15/Qt 6.11.2 passed 581 tests with six existing local privilege skips
(56.67 s); scoped independent review accepted the corrections. Bounded Qt diagnostics
did not reproduce the native exception; verbose hosted diagnostics were added without
a speculative product patch. A new current-head Copilot review and hosted checks
still precede merge. Installer and long-duration M1 acceptance remain open.


A second hosted run at `62089a4` exposed a post-stop read-watchdog race and repeated
the native Qt teardown failure. Copilot review `5400371073` also identified the
watchdog issue in its body. Commits `0b29897` and `63030f0` use the dedicated closing
deadline after stop, deliver background results only after actual worker-thread exit,
and retain/drain test desktops before widget destruction. Three deterministic failing
regressions verify the closing cause and normal/closed job ownership; six corrected
cases pass. Final source `63030f0`, Python 3.13.15/Qt 6.11.2: 585 passed, six existing
local privilege skips (56.93 s), scoped independent acceptance. A native Windows 11
Qt authored-WAV recovery/retry/exit check also passed with actual cleanup, no provider
or hardware, and 41.8 ms maximum heartbeat gap. These contract corrections do not
prove the original access-violation cause; new-head hosted checks and Copilot review
remain merge gates. M1 installer and long-duration acceptance are still open.


Hosted CI at `bd90996` passed 591 Windows tests (50.64 s, no skips), 588 Ubuntu
tests (33.28 s, three Windows-only skips), and wheel/source builds on both. Copilot
review `5400431791` then identified a high-severity postcommit cleanup defect and a
medium empty-directory-variable defect; both were accepted. Commits `f141189` and
`e3b8917` stage obsolete files before metadata-last publication, restore on handled
precommit failure, retain failed-rollback evidence, and keep owned cleanup failures
from falsifying committed success; empty LOCALAPPDATA/XDG uses the home fallback.
Thirteen RED cases became fifteen GREEN cases. Final committed-source suite:
600 passed, six existing local privilege skips (57.67 s), with scoped independent
acceptance. New-head hosted checks and Copilot review remain merge gates. Retained
owned backups and handled-I/O rollback are explicit limits; no atomic process-kill
transaction or full-M1 completion is claimed.


Hosted checkpoint `9299d59` passed 606 Windows tests (56.09 s, no skips),
603 Ubuntu tests (33.85 s, three Windows-only skips), and wheel/source builds.
Copilot review `5400522452` then found two medium body issues. Both are fixed:
`7b5a124` scopes recovery identity to its validated directory and journal
reservation; `3100d3e` makes only completed capture-file cleanup best effort,
preserving completed artifacts and persisted events even when residual WAV/path
validation or deletion fails. Direct discard/cancellation, precommit/provider/
publication failures and actual live/native closure remain strict. Seventeen RED
failures became twenty-four GREEN cases at `3100d3e` (624 passed, six existing
local privilege skips, 63.45 s). Scoped review exposed a copied-session retry
association defect: `e6e93a0` associates only the exact journal reservation,
leaving foreign bytes intact and invalid journal/path guards strict. Two RED
public-flow failures became nine GREEN cases. Final committed-source suite:
633 passed, 6 existing local privilege skips
(64.53 s); scoped corrective review accepted.
Residual capture evidence can remain after cleanup faults. Current-head hosted
checks and Copilot are still required before merge; full M1 remains open.


Copilot body review `5400663671` identified loss of original child-device errors
in a running controller: persistence and the UI selected the IPC wrapper even
though the journal retained the original cause. The accepted correction at
`9b21f57` derives capture errors from current validated journal events and
uses the persisted capture metadata in the desktop, preserving type, message
and occurrence time. Restart recovery uses the same persistence selection;
no-journal/parent-only failures retain a valid fallback. Meaningful regressions:
7 RED failures became
18 GREEN cases. Final committed-source suite:
654 passed, 6 existing local privilege
skips (68.06 s), with scoped independent acceptance. Previous
`35aed69` hosted CI passed 639 Windows/636 Ubuntu tests and both builds; new-head
Copilot and hosted checks are still required. Raw recovery and actual worker
ownership remain intact; no full-M1 or installed-runtime completion is claimed.


Copilot body review `5400816114` identified retryable audio candidates blocked by
a malformed optional meeting ID being used as a capture-journal locator.
Correction `0ca9a3a` checks canonical UUID form before journal lookup; absent,
malformed or noncanonical optional IDs associate no journal. Selected audio and
metadata remain validated, with their original identity values preserved. Canonical
journal document/path/media checks and exact reservation ownership remain strict.
This M1 compatibility rule grants no journal authority and does not certify M2
canonical catalog identity. Regressions: 15 RED
failures became 32 GREEN cases. Final source
suite: 677 passed, 6 existing local
privilege skips (71.7 s); scoped independent review accepted.
New-head hosted checks and Copilot remain merge gates; full M1 remains open.


Copilot body review `5400937500` identified CLI failure exits before actual worker
completion and pause requests hiding stalled device readers. Correction `662baba`:
CLI failure and interrupt exits cancel only active work and independently await unfinished actual worker completion in finally, so cancellation races cannot skip the fence and FAILED is not destructively cancelled. Original failures retain exit 1; original KeyboardInterrupt retains exit 130 with ordinary cleanup failure logged, without claiming a failed wait completed or catching BaseException.
Each actual child reader advances one monotonic counter only after active IO/write or paused availability/drain operations complete. Existing bounded ticks carry those counters separately from audio/live frames; parent deadlines renew only for strictly increased counters of that source. Missing, stale or lower values preserve history; sender activity and parent pause cannot mask a stalled reader. The separate post-stop closing deadline is unchanged.
Reader health originates from each actual child reader; parent pause state and
sender activity cannot certify device progress. Terminal failure does not grant
destructive cancellation. Real-gated CLI regressions reproduced four premature exits and two interrupt cleanup failures; final CLI/controller coverage passed 53 cases. Nine intended real-child watchdog failures became 16 passing pause cases, including either stalled source, blocked paused drain/availability, stale/missing/rollback ticks, healthy pauses, resume and paused stop with actual child/worker closure.
Final source suite: 705 passed, 6
existing local privilege skips (114.77 s); scoped independent
review accepted. New-head hosted checks and Copilot remain required; full M1 is open.
Actual Windows 11 two-source smoke at this source passed: 44.1/48 kHz stereo, 4.007 s healthy pause with a 3 s reader deadline and unchanged primary frame counts, both sources resumed, then paused stop in 0.262 s. Actual helper/owned threads closed; provider calls were zero and owned raw audio was removed after closure. This is short native reader/IPC proof, separate from GUI, frozen and long acceptance.


Copilot review `5401125631` identified healthy inserted-silence catch-up being
classified as a stalled reader and prior normalized WAVs remaining orphaned after
managed raw-audio retries. Both findings were accepted. Final source `1fa3688`:
Each reader advances its progress counter after each successfully inserted silence write and live delivery. Per-block stop/pause checks, explicit scheduling-gap journal evidence, independent reader watchdogs and the distinct closing deadline remain in place; blocked writes still time out.
Only after final completed/failed metadata publication, cleanup considers the prior validated recoverable normalized filename (audio.wav or audio_<32 lowercase hex>.wav). It rereads committed metadata, verifies final status/identity/current audio, protects selected input, current output/recoverable references and prior/current capture sources, revalidates containment/link safety, and removes only that single unreferenced prior file. Unknown files/names remain. Expected cleanup faults warn without replacing committed success or the original provider failure. Managed normalization failure preserves all prior manifest/error/audio bytes; new-session failure publication is unchanged.
Three real-child healthy catch-up RED failures became five passing catch-up cases; reader guards passed 107 cases. Thirteen authored-PCM cleanup/normalization RED failures with two existing publication-retention guards led to a final 243-case CLI/output focus with six existing privilege skips; unknown/current/raw references and cleanup faults are covered.
Final committed-source suite on CPython 3.13.15 / Qt 6.11.2: 734
passed, 6 existing local privilege skips, 138.64 s.
Changed-file lint/format and diff checks passed; scoped independent review accepted.
New-head Windows-only CI and formal Copilot review remain required before merge.
The user explicitly omitted two-hour capture/resource and long generated-audio
transcription acceptance; neither is run or claimed as passing. Finish remaining
Windows distribution and revised-scope M1 closure, then stop.


Copilot review `5402529641` found message-less background exceptions becoming
empty error strings and being treated as successful diagnostics/retry. Source
`843279c` now preserves nonempty messages and reports the actual exception type
when its message is empty; `None` remains the success sentinel. Actual worker/QTimer
ownership and closed-job delivery semantics remain unchanged.
Three actual Qt empty-message RED failures became 22 passing background/desktop cases, including visible diagnostic errors and retained explicit retry selection. The test-only pause diagnostics separately passed real-child failure-reporting experiments and all 33 isolated-capture cases.
Final committed-source Windows suite: 740 passed,
6 existing local privilege skips in 140.88 s;
changed-file checks and scoped independent review passed.
Historical Windows CI `37150084536` at `d89a386` failed one pause-marker assertion
with 739 other cases passing. The exact case and 12 bounded real-child repetitions
passed locally; its original cause remains unconfirmed. Test-only commit `0576648`
adds failure diagnostics after actual owned reap without changing timeouts,
watchdogs or active journal access. Its isolated-capture focus passed 33 cases.
New-head Windows CI and formal Copilot review remain gates. User-omitted long
capture/resource and long transcription cases remain omitted, not passed.

Long meeting means at least 7,200 seconds of recording wall time excluding deliberate
pauses. Compare known playback markers and recorded durations; do not require exact
hardware-clock synchronization between two sources. Initial marker seek tolerance is
250 ms. Memory must remain bounded by queue and current-chunk budgets; metadata may
grow with the number of chunks. Set a concrete hardware memory budget after measuring
the bundled application's idle baseline, and report baseline plus delta.

Clean-install verification must use an independent Windows environment without the
developer virtual environment or repo on PATH. A bundled executable launched from a
temporary directory tests independence but alone does not prove a clean installation.
