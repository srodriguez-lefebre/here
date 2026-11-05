# Autonomous product delivery

The user delegated product decisions, implementation, verification, GitHub pull requests,
Copilot review requests, replies to every review observation and merges on 2026-10-03.
This document is the recovery map for continuous execution; a milestone is not complete
merely because its code exists or its unit tests pass.

## Rulings vigentes — 2026-11-04

Decisiones tomadas por el usuario en sesión el 2026-11-04. Reemplazan cualquier
instrucción anterior de este documento que las contradiga; las secciones siguientes se
conservan como historia. Formato: Ruling / Cost if wrong, como el resto del ledger.
Los rulings nuevos se escriben en español por el ruling 6.

1. **Ruling:** el objetivo vigente es preparar el plan para ejecución autónoma, no
   implementar. M2 no arranca hasta nueva instrucción explícita. Las tareas de
   preparación están en `STATUS.md`. **Cost if wrong:** tiempo de planificación sin
   código; reversible en cualquier momento con una instrucción.
2. **Ruling:** la rama `codex/ui-configuration-errors` se descarta. Contradice "sin UI
   de credenciales en M1" y no tiene PR. Se borra en local y en `origin` cuando el
   usuario confirme la acción. **Cost if wrong:** dos commits (823 líneas) con editor
   de configuración, controles del indicador y errores persistentes se rehacen desde
   el diseño de M5 si se necesitan.
3. **Ruling:** `docs/development/PRODUCT_DESIGN.md` es decisión tomada para M2–M5,
   no propuesta. Un agente la implementa sin reconfirmar, salvo lo que el documento
   marca como pendiente o lo que contradiga un principio del plan maestro.
   **Cost if wrong:** rework de un hito entero si el diseño resulta equivocado; se
   mitiga con los criterios verificables por hito antes de implementar.
4. **Ruling:** ningún agente mergea a `main`. El agente abre el PR con self-review y
   review de un agente independiente, registra la evidencia y sigue en otra rama. El
   usuario revisa y mergea. El permiso de administrador no se usa para merges.
   **Cost if wrong:** PRs esperan al usuario; el agente puede acumular ramas apiladas.
5. **Ruling:** hay máquina Windows limpia y credencial real del proveedor disponibles
   a pedido del usuario. No hay disponibilidad para la reunión de dos horas; esa fila
   de aceptación sigue "omitida, no aprobada". **Cost if wrong:** si la máquina o la
   credencial no estuvieran cuando se pidan, las filas vuelven a "pendiente".
6. **Ruling:** idioma. Planes, `STATUS.md`, criterios de aceptación y rulings nuevos
   en español. Código, commits, mensajes de PR, specs técnicos y el resto de
   `docs/development/` en inglés. Lo existente no se traduce. **Cost if wrong:** docs
   bilingües; sin impacto en el producto.
7. **Ruling:** el modelo de `INTELLIGENCE_MODEL` para M4 se decide al implementar M4,
   con un ruling que registre modelo y fecha. La frase "coherente con el modelo de
   texto ya usado" de `PRODUCT_DESIGN.md` se refiere a `CLEANUP_MODEL`
   (`gpt-4.1-mini`, limpieza de transcript); no obliga a reutilizarlo.
   **Cost if wrong:** ninguno hasta M4.
8. **Ruling:** overlay. Una cancelación, destructiva o recuperable, hace que el logo
   vuelva a reposo y desaparezca sin check ni cruz; el detalle queda en la ventana
   principal. La marca actual no se rediseña en M2–M5. Cierra las dos decisiones
   pendientes de `docs/LIVE_LOGO_PLAN.md`. **Cost if wrong:** un cambio visual
   acotado al overlay.
9. **Ruling:** versionado por hito, una minor cada uno: M2 → 0.3.0, M3 → 0.4.0,
   M4 → 0.5.0, M5 → 1.0.0. Reemplaza "v0.3.0 = producto completo" registrado más
   abajo. **Cost if wrong:** renumerar tags no publicados.
10. **Ruling:** los documentos locales de `docs/` que git ignora (`CONTEXT.md`,
    `CHUNKING_*.md`, `PYTHON_VERSION_ARCHITECTURE.md`, `future_plans.md`, el draft
    duplicado de `PRODUCT_DESIGN.md`) son históricos. Se mueven a `docs/archive/` con
    un README y no se usan como contexto. **Cost if wrong:** ninguno; siguen en disco.
11. **Ruling:** el workflow manual de Ubuntu (`.github/workflows/ci-ubuntu-manual.yml`)
    se elimina junto con sus referencias en los docs. Linux no es objetivo de producto
    ni gate de nada. Se ejecuta en un PR aparte cuando el usuario confirme.
    **Cost if wrong:** recuperar el archivo del historial de git.
12. **Ruling:** no se piden más reviews a Copilot en ningún hito. La revisión de un
    PR es self-review del autor más review de un agente independiente; el único
    review humano es el usuario, que también mergea. Generaliza la instrucción de
    cierre de M1. **Cost if wrong:** menos ojos por PR; se compensa con CI de Windows
    y los criterios verificables por hito.

### Rulings del 2026-11-05

13. **Ruling:** se levanta el ruling 1. El objetivo vigente es implementar el hito 2
    según `docs/MILESTONE_2_PLAN.md`, empezando por el grupo A
    (`M2_CATALOG_PLAN.md`, tarea 1). Los PRs #16 y #17 están mergeados en `main`
    (`16d7beb`, `1d1e96f`). **Cost if wrong:** el trabajo queda en ramas y PRs sin
    mergear; nada toca `main` sin el usuario (ruling 4).
14. **Ruling:** el agente redacta los planes siguientes por su cuenta, sin preguntar:
    al cerrar un grupo de PRs escribe el SPEC y el PLAN del grupo siguiente; al cerrar
    un hito escribe el `MILESTONE_<n+1>_PLAN.md` con criterios verificables, en el
    mismo formato y los mismos idiomas que los de M2. Cada plan nuevo se entrega en un
    PR de documentación que el usuario revisa y mergea antes de que empiece su
    implementación; esa revisión es el único punto de contacto obligatorio entre
    hitos. Las preguntas de la sección 3 de `CLAUDE.md` siguen vigentes dentro de cada
    plan. **Cost if wrong:** un plan mal encaminado se corrige en su PR antes de que
    exista código.
15. **Ruling:** anula el ruling 13. El usuario aclaró el 2026-11-05 que no quiere
    implementación: el objetivo vigente es **determinar el plan completo de M2 a M5**
    (plan por hito con criterios verificables y SPEC/PLAN de cada grupo de PR) y
    dejarlo publicado. Ninguna implementación arranca sin una instrucción explícita
    posterior. El ruling 14 sigue vigente para mantener los planes al día durante la
    ejecución futura. **Cost if wrong:** ninguno; los planes se escriben una vez y se
    corrigen en PR.
16. **Ruling:** anula el ruling 4. El agente mergea sus propios PRs a `main` usando el
    bypass de administrador del ruleset `Main` (id 14750729, rol de repositorio 5),
    con `gh pr merge <n> --merge --admin --match-head-commit <sha>`, y solo cuando se
    cumplen las cuatro condiciones a la vez: CI de Windows verde en ese head exacto;
    review de Copilot solicitada sobre ese head y recibida; toda observación de
    Copilot, inline o de cuerpo, respondida con su disposición (corregida con commit,
    o rechazada con razón escrita); review de un subagente independiente del autor
    registrada en la descripción del PR. Un merge sin alguna de las cuatro es una
    violación del plan, no un atajo. **Cost if wrong:** un defecto entra a `main` sin
    ojos humanos; se mitiga con Copilot, CI y el ledger de evidencia por PR.
17. **Ruling:** anula el ruling 12. Cada PR pide review de Copilot sobre su head
    actual (`gh pr edit <n> --add-reviewer copilot`, o el mecanismo equivalente que
    GitHub exponga en ese momento), espera a que llegue, responde cada observación en
    el hilo, y vuelve a pedirla después de cualquier corrección sustantiva. El
    silencio de Copilot no es aprobación; si no responde en un plazo razonable se
    registra en el PR y se aplica el ruling 16 solo con las otras tres condiciones.
    Las reviews de Copilot son COMMENTED, no APPROVED; el ledger lo anota así, como
    en M1. **Cost if wrong:** observaciones sin atender; se detectan en el ledger.
18. **Ruling:** el ejecutor de M2–M5 es Codex (OpenAI). `AGENTS.md` en la raíz es su
    punto de entrada y remite a `CLAUDE.md`, que sigue siendo la hoja de arranque
    única; cualquier agente que lea uno de los dos tiene las mismas reglas. Codex
    trabaja como orquestador con subagentes según `CLAUDE.md` §11: un escritor por
    conjunto de archivos, review siempre por un subagente distinto del autor, y los
    solapamientos entre grupos que esa sección permite, ninguno más. **Cost if
    wrong:** conflictos de merge o contratos desalineados entre grupos; se detectan
    en CI y en la review independiente.
19. **Ruling:** el usuario autoriza el 2026-11-05 a usar su credencial real de OpenAI
    en los smokes opt-in contra el proveedor sin volver a preguntar. La credencial se
    expone como `HERE_PROVIDER_SMOKE_KEY` en el `.env` del usuario (mismo valor que
    `OPENAI_API_KEY`); solo los tests marcados como smoke de proveedor la leen, y se
    saltean cuando no está definida. La suite normal sigue usando una key sintética y
    transporte simulado. Sigue prohibido imprimir la key, copiarla a logs, evidencia,
    QSettings, exports o commits, y usar grabaciones privadas: los smokes usan solo
    audio y texto sintéticos. Cada corrida registra modelo, fecha y costo informado
    por el proveedor en `ACCEPTANCE.md`. Precisa el ruling 5: la credencial ya no es
    "a pedido" para los smokes; la máquina Windows limpia sí lo sigue siendo.
    **Cost if wrong:** gasto en la cuenta del usuario por smokes repetidos; se acota
    porque solo corren en los pasos que los PLAN nombran, con texto corto.

20. **Ruling:** por instrucción explícita del usuario el 2026-11-05, se reemplaza
    el ruling 15 y se autoriza implementar completos M2–M5 en los nueve grupos
    de PR de STATUS, empezando por M2_CATALOG_PLAN, tarea 1. Codex orquesta RED
    por subagente, implementación GREEN, review por otro subagente independiente
    y documentación/evidencia por otro, con worktrees separados, un escritor por
    archivo y los solapamientos exclusivos de CLAUDE §11. Migraciones, pyproject y
    versión siempre en serie. Cada grupo: PR abierto contra main con evidencia y
    review independiente; Copilot solicitado sobre el head actual y recibido;
    cada observación respondida en su hilo, corregida con commit o rechazada con
    razón y review repetida tras correcciones sustantivas; CI Windows verde en ese
    head; merge commit con --admin --match-head-commit; anotación en DELIVERY.
    Ninguna condición se omite, incluso si Copilot demora. Contratos que cambien
    requieren PR de documentación del PLAN siguiente antes de implementarlo. Cada
    hito verifica todos sus criterios con corridas reales y actualiza ACCEPTANCE,
    versión (0.3.0, 0.4.0, 0.5.0, 1.0.0), README, ARCHITECTURE, WHATS_NEW y STATUS.
    Se registran defaults menores como rulings; solo se pregunta lo previsto por
    CLAUDE §3. Smokes del ruling 19 usan HERE_PROVIDER_SMOKE_KEY del .env original,
    sin imprimirla ni persistirla, con texto/audio sintéticos. Máquina limpia se
    pide con protocolo cuando corresponda. Prueba de dos horas: **omitida por
    ruling 5**, nunca aprobada. Se mantienen los tres niveles de evidencia y las
    definiciones de terminado. M5 exige instalador 1.0.0 y portable como artefactos
    Windows con SHA256SUMS, inventario, provenance y avisos; matriz completa,
    instalación limpia y upgrade desde 0.2.0 real conservando biblioteca. Bloqueos
    humanos quedan en STATUS con PR abierto y evidencia; se retoma desde STATUS.
    **Cost if wrong:** implementación y gasto de CI/proveedor de cuatro hitos;
    los gates y la evidencia impiden confundir código con certificación.

21. **Ruling:** el archivo separado de texto fuente de cada transcripción se llama
    `raw_transcript.txt`. Conserva `raw_text` independientemente del texto de
    presentación en `transcript.txt`; el nombre concreta el requisito ya aprobado
    de preservar raw text, sin agregar tiempos ni hablantes. **Cost if wrong:**
    renombrar el artefacto y ajustar sus lectores, manifiestos y tests antes de
    publicar un formato estable.
22. **Ruling:** `MemoryService.ingest_session` admite el argumento opcional y
    aditivo `revision_of`, cuyo valor es el ID de revisión activa esperada. Solo la
    vía confiable de retry explícito lo aporta para publicar una revisión nueva;
    se valida la identidad y la revisión esperada antes de activar. Una importación
    ordinaria con el mismo UUID y contenido distinto sigue rechazada como conflicto;
    el contenido de una carpeta no otorga autoridad de retry. Precisa el contrato
    de tarea 3, cuyos tests RED e implementación siguen pendientes. **Cost if wrong:**
    ajustar los adaptadores de retry, tests de conflicto/concurrencia y contrato de
    ingestión antes de que ciclo de vida dependa de él; no se reescriben revisiones
    previas para resolver el conflicto.

23. **Ruling:** la publicación y reconciliación locales del catálogo se serializan
    con `<data_dir>/.memory.lock`; la propiedad real de captura/procesamiento se
    acredita con un lock de archivo del sistema operativo en
    `<data_dir>/.owners/<uuid>.lock`, sin inferir actividad de un PID o timestamp.
    Adquisición acotada a 250 ms devuelve `Busy`; los locks no sustituyen las
    transacciones cortas ni permiten copiar/calcular hashes dentro de ellas. No se
    agrega dependencia ni proceso residente. La aplicación puede conservar en su
    carpeta de sesión propia `memory_pending.json`, con versión, identidad y la
    revisión esperada del retry explícito, para reintentar solo ingestión local
    cuando la DB estuvo ocupada o corrupta. No contiene secretos, payloads del
    proveedor ni autoridad sobre rutas externas; nunca dispara proveedor al abrir.
    Los paths y cada componente se validan con los helpers existentes. La
    reconciliación conserva trabajos con propiedad real vigente; un archivo de
    lock residual no demuestra actividad. **Cost if wrong:** ajustar el layout
    interno de locks y el lector de marcadores antes de que M2 B/M5 dependan de
    ellos; los archivos originales y revisiones previas siguen preservados.

24. **Ruling:** se adopta el default de M2 que permite una raíz de datos de red
    con advertencia visible, sin recomendarla; incluye unidades de red mapeadas,
    y SQLite usa rollback journal en vez de WAL cuando el volumen es remoto.
    Las advertencias del catálogo se presentan separadas del error de captura en
    el campo aditivo `ApplicationSnapshot.memory_warning`: una captura completada
    sigue completada y no adquiere un `last_error` falso. La ventana muestra esa
    advertencia cuando no hay un error real con prioridad, y la limpia al iniciar
    otro trabajo o completar una persistencia sin advertencias. Precisa la forma
    de mostrar el aviso que ya exige el criterio 10; no agrega pantalla ni flujo.
    **Cost if wrong:** ajustar el canal de advertencias y el sondeo del volumen;
    ninguna revisión ni audio necesita reescribirse.

25. **Ruling:** el manifiesto de una sesión importada solo puede referenciar
    artefactos conocidos de sesión y audio WAV explícito validado. Una ruta
    contenida no basta para autorizar leerla: nombres como `.env`, logs ajenos o
    archivos desconocidos se rechazan antes de abrir, sondear o calcular hashes.
    Una referencia de audio tampoco convierte credenciales en audio. La regla
    conserva los artefactos v1 conocidos y nunca copia la carpeta completa.
    Concreta el principio de privacidad y la validación de entradas del SPEC.
    **Cost if wrong:** extender explícitamente los tipos de artefacto permitidos
    cuando un formato legítimo lo requiera; las fuentes externas no se modifican.

26. **Ruling:** la reconstrucción del catálogo usa snapshots administrados
    autocontenidos con IDs, revisiones, segmentos, assets, hashes y procedencia.
    `catalog.json` dentro de la revisión se publica después del commit de DB y
    sirve como recibo de terminación; un árbol preparado sin recibo no acredita
    una revisión publicada. Los estados durables se conservan sin reescribir
    historia en `meetings/<uuid>/states/<row_version>-<uuid>.json`. Con la DB
    vigente, reconciliación puede terminar recibos pendientes desde lo que fue
    comprometido; si se pierden DB/journal antes del recibo, el árbol incierto se
    informa y no se activa por conjetura. La reconstrucción valida rutas y hashes,
    usa solo archivos administrados y no escanea fuentes externas ni llama al
    proveedor. **Cost if wrong:** versionar los snapshots/recibos y ajustar el
    lector de reconstrucción y la purga de M2 B; las revisiones anteriores no se
    modifican para corregir el formato.

27. **Ruling:** transcribir un archivo y reintentar desde la aplicación también
    mantienen estado durable `processing` y propiedad real durante el trabajo del
    proveedor, sin mantener una transacción SQLite abierta. Si un retry falla sin
    devolver una transcripción, conserva la revisión activa anterior y solo cambia
    el estado a `failed`; no fabrica una revisión nueva con texto viejo. Cuando la
    DB está ocupada, el marcador local puede distinguir un intento de ingestión de
    una actualización de estado y guardar ese desenlace junto a la revisión esperada.
    El replay local respeta esa precondición y no ejecuta de nuevo al proveedor.
    **Cost if wrong:** ajustar el lector de intención del marcador y los adaptadores
    de procesamiento; el historial de evidencia previo permanece intacto.

28. **Ruling:** el retry explícito de una sesión v1 ya importada que conserva
    `meeting_id=None` en sus archivos reutiliza la identidad canónica del catálogo
    mediante su origen normalizado y la revisión activa esperada. La huella nueva
    del resultado no crea otra reunión en ese recorrido confiable. La importación
    ordinaria conserva el contrato ruta + huella; el retry no reescribe el ID
    opcional del artefacto legado para resolver la asociación. Si el origen apunta
    a varias identidades posibles, se informa conflicto en lugar de elegir una
    por conjetura. **Cost if wrong:** ajustar la resolución del contexto de retry
    y los mappings de origen, conservando audio y revisiones ya publicadas.

## Ledger histórico (append-only)

Todo lo que sigue es historia de la entrega de M1: objetivos de cada momento, rulings
de entonces, checkpoints de evidencia, reviews y merges. No se edita; se agrega al
final. El estado vigente está en `STATUS.md` y en los rulings de arriba.

## Current authorized objective

> Superseded on 2026-11-04 by the rulings above. Kept as history.

On 2026-10-03 the user narrowed execution to **finish M1, then stop**. M2-M5 remain
future plans; their implementation is outside this run. The user also requested
Windows-only development and validation. Automatic CI and release gates use Windows;
the previous Ubuntu workflow is retained separately for explicit manual dispatch and
will not be run here. Earlier Ubuntu results below remain historical observations.
Prioritize the remaining desktop fixes, Windows distribution and necessary M1
acceptance; no later-milestone work or optional validation expands this scope.

The user subsequently instructed that Copilot review `5402608228` is the last.
No further Copilot review requests will be made, including for distribution.
Remaining delivery uses independent peer review and actual Windows source/package
CI, with replies to all already-received findings. This explicitly supersedes the
original recurring Copilot-request gate below.

The user's latest explicit instruction on 2026-10-03 omits both the two-hour
capture/pause/recovery/resource run and long generated-audio transcription with
timestamps/diarization. These cases are excluded from this delivery, not passed.
Complete the remaining Windows distribution, actual runtime/installation checks,
notices/checksums, reviewed PRs and evidence audit; then close M1 under this revised
scope and stop. Existing short hardware/fault/provider observations retain their
actual provenance. M2-M5 remain outside this run.

## Product authority and architecture

`docs/MASTER_PLAN.md` defines the five milestones. `docs/MILESTONE_1_PLAN.md` and
`docs/LIVE_LOGO_PLAN.md` govern capture and the overlay. Retain one application owner,
PySide6 Widgets, a shared application controller, Windows 11 support, explicit
cancellation semantics and local user-owned audio. Derived knowledge must always link
to its supporting meeting and original segment; missing timestamps or speakers stay
missing. No cloud synchronization, accounts, unsolicited transmission or automatic
audio deletion is introduced.

Ruling: allow a short-lived isolated hardware helper where the backend cannot safely
bound opening, reading or cancellation. The desktop spec permits that isolation;
the GUI, overlay and shared controller remain one application, with no persistent
service. A timeout cannot release audio resources until the helper actually closes.
This is a deliberate refinement of the original single-Python-process design, needed
to recover from a stalled driver without destroying a resource-owning thread. Cost
if wrong: bounded IPC complexity and runtime overhead, to verify in desktop/bundle tests.

For structured memory use transactional SQLite with foreign keys and versioned
migrations, retaining portable recoverable session artifacts. Browsing and local search
must work without an API key. Remote intelligence must be an explicit user command,
with source validation and a labeled offline extractive path. The detailed specifications
and individual execution plans are added here before their implementation.

## Delivery sequence

1. M1 capture reliability: preserve partial capture on failures, bounded live queues,
   structured timestamps/speakers at the artifact boundary; then default devices,
   diagnostics, restart recovery, stop-save-exit, per-user paths; then Windows installer
   and an honest acceptance matrix. Split into independently reviewable PRs.
2. M2 structured memory: durable identities, transcript revisions, segments,
   participants, atomic catalog updates and recovery reconciliation; lifecycle and
   deletion integrity; migration of existing artifact sessions.
3. M3 navigation: GUI and CLI meeting list/detail/filter/search, contextual results
   and original-audio playback at recorded timestamps. Missing timing never implies
   a fabricated seek position.
4. M4 intelligence: retrieve across selected meetings, answer and extract decisions,
   tasks and blockers with validated evidence references, explicit uncertainty and
   preserved generation provenance.
5. M5 daily product: configuration, data control, export/import, reliable startup and
   logs, installer/uninstaller, explicit update checks and practical interoperability.

Planned PR groups preserve task commits and independent task reviews: M1 capture,
desktop recovery and distribution/acceptance; M2 immutable catalog (tasks 1–3),
lifecycle/erasure (4–5) and shared persistence (6–7); M3 search/actions (1–2) and
library/CLI/evidence (3–6); M4 grounded core (1–5) and product flows/evidence (6–8);
M5 settings/runtime (1–2), portability/privacy (3–5) and updates/product acceptance
(6–8). These are delivery boundaries, not completed milestones or published releases.

The first group, M1 capture reliability, merged as [PR #12](https://github.com/srodriguez-lefebre/here/pull/12)
on 2026-10-03 07:52:15 UTC, merge `ca07aed8f6d5abf43c85e03ca0107d60096c3e03`.
The desktop/recovery group merged as [PR #13](https://github.com/srodriguez-lefebre/here/pull/13)
at 2026-10-03T20:41:23Z, merge `7ed78afc551c09729ebd886c27089b9c989bbafe`. Its final source is
`843279c` and reviewed PR head is `cb4fa9d`:
746 Windows tests with
0 skips, plus wheel/source builds passed.
Ubuntu remains a separate manual-only workflow and was not run for this delivery.
Copilot review `5402608228` and all earlier inline/body findings
were evaluated and answered; independent review accepted the corrected source.
Configured administrator merge permission was used with an exact-head guard.
Historical task/checkpoint observations below retain their original source heads.

The next branch is `codex/m1-windows-distribution`. Ruling: split distribution into
the remaining M1 delivery PR: actual frozen runtime, installer, notices,
clean installation proof and the revised-scope closure audit. The user explicitly
omitted the two-hour capture/resource run and long generated-audio transcription;
neither is executed or claimed as passing. Desktop delivery alone does not close M1.
After the remaining reviewed delivery merges and evidence is recorded, close M1
under this revised scope and stop. M2-M5 remain future work outside this run.

Distribution local delivery at clean incorporated-source build `1740b2d` now passed:
locked noneditable 0.2.0 wheel, windowed GUI/console CLI, bounded no-key/off-repo Qt
and helper smoke, complete native/notice inventory, and actual stable per-user
installer Start launch, active replacement refusal, reinstall/uninstall and preserved
synthetic data. See [`M1_DISTRIBUTION_EVIDENCE.json`](M1_DISTRIBUTION_EVIDENCE.json)
for hashes and exact runtime/source limits. Distribution peer review, actual clean
Windows source/package CI and guarded merge remain; local proof does not close those
gates or certify repeated installed hardware/fault tests. After those gates pass,
record the actual merge, close M1 under the revised scope and stop.

Desktop Task 1 is independently approved at `d366711`: per-user configuration,
optional credentials, writer-owned WAV/journal checkpoints, stable capture UUID and
local recovery, shared CLI pending publication and interruptible catch-up. One
important independent finding was fixed before approval: file transcription now
publishes recoverable pending metadata before provider work. The final source-head
suite passed 524 tests with six existing Windows symlink-privilege skips (34.02 s);
new real junction/hardlink and subprocess-kill cases ran. No desktop Copilot review,
hardware-long, installer or whole-M1 completion is claimed here.

Desktop Task 2 is independently approved at `76d317a`: immutable actual opened devices,
asynchronous diagnostics/recovery, atomic preparation stop, completion-gated stop/save
exit and bounded hardware helpers. Actual Windows Job Object parent-death and stalled
backend/IPC-pressure regressions ran. Final source-head suite: 558 passed, six existing
symlink-privilege skips in 53.72 s; changed-file lint/format and diff checks passed.
The current synchronous provider transport remains cooperative and can wait through
its SDK timeout/retries; Qt retains ownership without blocking its event loop.

A bounded native Qt Windows check at that same source head passed without provider
calls: no-key startup, local interrupted discovery, key preflight preserving the journal,
actual default-device names, three-second microphone no-signal and authored-tone
loopback signal diagnostics, ordinary active close/hide and repeated Salir from pause.
An intentional local transcription failure left discoverable audio; exit followed
durable persistence and actual cleanup (0.372 s for this local fixture only).
Qt heartbeat maximum observed gap was 49.6 ms across 688 samples. Probe-owned raw
audio was removed after worker/reservation cleanup. Real frozen helper launch,
installation, long recording and real-provider long acceptance remain open.
Safe observations: [`M1_DESKTOP_EVIDENCE.json`](M1_DESKTOP_EVIDENCE.json).

Whole-branch review found two recovery integration defects: a missing normalized WAV
hid valid raw-backed sessions, and a missing-key retry removed the selector choices.
Commit `490b77c` fixes both and strengthens the preparation-stop test. Five failing
regressions became eleven passing; the final committed-head suite passed 569 tests
with six existing symlink-privilege skips (56.92 s). Scoped independent review accepts
the whole-branch source gate. A separate native Qt authored-WAV probe at `490b77c`
confirmed fresh raw-backed recovery, retained choices after missing-key error and one
explicit same-window retry after editing the effective env file, reconstructing 240
frames with UUID, human ID, selected filename and raw bytes preserved. No hardware
or provider was used for this correction probe. External Copilot and head CI gates
still precede merge; installer and full-M1 acceptance remain separate.

The legacy private CLI helper deferred in PR #12 review `5399598693` is consolidated
with SessionProcessor in desktop Task 1; its actual tiny-WAV normalization-failure,
preserved-local-source and later-retry regression passes. The original deferral below
records the historical first-PR disposition, rather than an outstanding source defect.


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

## Review and merge gates

- Each implementation task follows meaningful red/green tests and self-review.
- A fresh independent reviewer checks task requirements and code quality.
- Each PR has multiple cohesive commits where appropriate and is validated on its head.
- Push and create the PR, attach it to this chat, request `@copilot` with GitHub CLI.
- Wait for Copilot review of the current head; reply to every inline observation with
  its disposition, evidence and fix commit or reasoned rejection. Fix real defects.
- Re-request review after substantive fixes; merge only after successful checks and
  no unresolved material findings. Never treat review silence as approval.
- Use merge commits to preserve the requested commit groups. Update this ledger with
  PR, head, review and validation evidence before proceeding.
- Main ruleset `14750729` requires an approving/code-owner review and explicitly
  permits repository administrators to merge. The delegated owner account has ADMIN
  permission. Use that configured exception, without changing rules, only after the
  gates above pass and with `--match-head-commit`. Copilot's COMMENTED reviews and
  independent agent reviews are recorded as such; neither is a human APPROVED review.

## Acceptance and evidence

Baseline: main `2bdb6ef`, 163 passing tests on the original workspace. Work occurs in
`C:/Users/savar/.codex/worktrees/five-milestones/here`, not the user's main checkout.
The local Windows default devices enumerate at 44.1 kHz microphone and 48 kHz loopback.
Enumeration does not certify captured signal, long meetings or a clean installation.

Define a long meeting as at least two hours. Automated virtual-duration tests measure
bounded resource behavior and completeness, but do not replace a two-hour hardware
run. Keep manual/hardware/provider/clean-install evidence explicit and leave acceptance
open wherever it has not actually been observed.

## Progress

- 2026-10-03: user delegation recorded; isolated managed worktree created.
- GitHub admin/push access and Copilot reviewer availability confirmed.
- Read-only capture audit and product architecture delegated to independent agents.
- Ruling: use the explicit autonomous delegation instead of repeated design approvals;
  preserve written specs and plans for review. Cost if wrong: reversible implementation
  rework, visible in committed decisions and PRs.
- Ruling: retain historical local docs ignored; version only the authoritative plans
  and new `docs/development/` evidence. Cost if wrong: older documents need explicit
  inclusion later; no private local scratch is accidentally published.
- Ruling: preserve supplied audio and facts; do not certify unobserved hardware tests.
  Cost if wrong: release acceptance takes longer, while code can continue advancing.
- M1 capture implementation: commits `c20ac3b` and `081ef94`; synthetic full suite
  reached 194 passing tests. Real synthetic TTS/provider smoke preserved four timed,
  scoped utterances. Wheel built and installed non-editably into isolated Python 3.13;
  GUI imports, Qt offscreen initialization and actual CLI help passed off-repository.
- Independent capture review reproduced two P1 defects: catch-up silence was omitted
  from the live timeline, and destructive cancellation during failure cleanup could
  still persist discarded audio. Both accepted for correction before PR creation.
- Ruling: serialize capture cancellation against the recovery-persistence cutover.
  Accepted capture cancellation deletes its partial sources; cancellation after the
  atomic processing cutover preserves audio and reports recoverable cancellation.
  Cost if wrong: recovery-state implementation rework; discarded private material
  must never survive an accepted destructive cancellation.
- Independent desktop-plan review adopted seven amendments covering actual WAV header
  checkpoints, atomic publication, preparation exit, diagnostic ownership, path
  validation, ready-time device provenance and credential precedence.
- Release decision: planned M1 Windows delivery is `v0.2.0`; the complete five-milestone
  product targets `v0.3.0`. These are targets until built, reviewed and published.
  Existing official release is tag `MVP`, display name `v0.1.0`, published 2026-04-11,
  without runtime assets. Treat that non-version tag as a legacy release, not a
  trusted comparable installer update. Reuse the actual M1 artifact for M5 upgrade tests.
- Review corrections in `7602c18`: every catch-up silence block follows disk order into
  live capture; shared atomic cutover prevents accepted destructive cancellation from
  persisting audio, and late processing cancellation retains matching metadata.
  Six regression cases and affected suites passed (55); full committed-head suite:
  200 passed in 16.86 seconds. Independent re-review precedes PR creation.
- First M1 subset PR: https://github.com/srodriguez-lefebre/here/pull/12. Independent
  task and whole-branch reviews accepted the implementation. GitHub recorded the
  requested Copilot review; review `5399146085` inspected head `c3dc790` and identified
  two accepted defects (comments `4171819200`, `4171819228`). Commit `4860095` fixes
  successful-result cleanup after late cancellation and valid nested timestamp alias
  fallback. Ten cases reproduced nine failures before correction; 57 focused tests
  and the full suite of 211 passed. Both observations were replied to and resolved.
- Initial CI exposed an existing UTC string-comparison assumption and a missing EGL
  runtime on Ubuntu. Commit `545174e` adds deterministic UTC/non-UTC checks using aware
  datetime equality and installs libegl1. Windows and Ubuntu jobs both passed on that
  head (run `37099618688`). Later review-fix CI runs `37102433292` (`6f1f88d`) and
  `37103530702` (`240858e`) also passed. The latter ran 266 tests on each platform,
  including the symlink case omitted locally, and built wheel/source distributions.
- Further Copilot observations were accepted and replied to: oversized timestamp
  integers (`5399210060`) fixed in `6ac6a0f`; stale segment evidence (`5399274184`)
  fixed in `e48f9ee`; cleanup ordering tightened in `6f1f88d` and explained in the
  reply to review `5399321887`. Pair-publication failure (`5399348275`, inline
  `4171988549`) fixed in `240858e`, independently accepted, replied to and resolved.
- Review `5399404470` of `240858e`, received 2026-10-03 06:39:41 UTC, identified
  unsafe recovery output entries (inline `4172034351`) and reversed timing pairs.
  Commit `a9cb09a` rejects pre-existing redirects, hardlinks and nonregular managed
  entries before retry reads/provider calls and stages text, normalized WAV and source
  copies before replacement. Commit `7ed996a` keeps segment text/speaker but sets both
  contradictory time bounds to null before offsets, including directly typed segments.
  The stale-ledger observation is addressed by this event register.
- Corrective-code checkpoint `7ed996a`: full offscreen suite **385 passed, 6 skipped
  in 10.62 s**. All skips need local Windows symlink privileges; real hardlink and
  junction cases passed. Focused security suite: 129 passed/6 skipped; focused timing
  suite: 168 passed/1 skipped. Ruff, format and diff checks passed. This local result
  does not substitute for hosted CI of the final PR head.
- Final capture corrections: `ac46ff8` validates every referenced WAV geometry before
  retry normalization/provider work and derives provenance duration from the header;
  `3162a74` coerces direct typed bounds before order checks, preserving valid companion
  values and source text. Review `5399510672` findings were replied to and resolved;
  independent scoped review accepted requirements and quality. Final code suite:
  463 passed/6 local symlink-privilege skips in 12.98 s. Hosted CI `37107198007`
  at exact `3162a74` passed 469 Windows tests without skips and 467 Ubuntu tests with
  two Windows-only junction skips; both distributions built on both systems.
- Copilot review `5399598693` inspected exact `3162a74` and confirmed those two fixes.
  Its body also identifies a retained private CLI helper's normalization-failure
  record without local raw references. The complete source call graph shows only test
  callers of `_save_transcription`; current recording commands use the shared
  controller/SessionProcessor, whose failure persistence already owns local sources.
  Ruling: consolidate this pre-existing helper in desktop Task 1, with a real tiny-WAV
  failure-to-local-retry regression, alongside early CLI managed-session preflight;
  it does not block the delivered production capture subset. Cost if wrong: external
  callers of this private helper need their original temporary audio until that PR.
  The observation is accepted and publicly answered in comment `5966927016`, not
  declared fixed or silently discarded. M1 stays open until consolidation is verified.
- All six inline threads were replied to/resolved and every review-body observation
  dispositioned before merge. PR #12 merged using the configured administrator
  permission and exact-head guard at `3162a74`, preserving its commit history in
  `ca07aed`. This is a COMMENTED Copilot review plus independent agent reviews, not a
  human APPROVED review. Full immutable evidence is in
  [`M1_CAPTURE_EVIDENCE.json`](M1_CAPTURE_EVIDENCE.json); short native observations are
  in [`M1_NATIVE_EVIDENCE.json`](M1_NATIVE_EVIDENCE.json), with original code provenance.

Desktop Task 1 also closes the parked catch-up control and CLI early-read boundaries
and preserves capture UUID as additive `meeting_id` for M2. These amendments are in
the desktop spec/plan. Journals, optional credentials and desktop controls are now
implemented and independently reviewed on this branch, with their own observations;
they were not facts implied by the first capture merge.

This register records immutable observed checkpoints. The live
[PR review/check/merge record](https://github.com/srodriguez-lefebre/here/pull/12)
is the authority for subsequent review dispositions, final-head CI and merge status;
earlier passing runs never waive a later-head gate. Desktop PR #13 is now merged
as `7ed78af`; its final review findings were adjudicated and replied to. The remaining
distribution PR incorporates that merge and has a clean-source Windows build,
actual local installer lifecycle proof and a final suite of 743 passed with six
local symlink-privilege skips at `e5520c6`. Independent distribution review,
current-head Windows source/package CI and its guarded merge complete the revised
M1 delivery. The two long acceptance runs are omitted by explicit user instruction;
M2–M5 are future plans outside this run. No further Copilot request is authorized.

## 2026-11-05 — M2 A task 1 local checkpoint

Ruling 20 authorizes the current nine-group M2–M5 execution and supersedes the
historical stop statements above. Task 1 source `9a80060` implements frozen catalog
DTOs, all twelve initial tables, numbered/checksummed transactional migrations,
managed SQLite API backups, operation-owned connections and an explicit transaction
repository. This checkpoint is implemented and automatically verified locally;
the M2 group is not merged or complete.

Evidence runs recorded in `ACCEPTANCE.md`: `M2A-T1-RED-20261105` (27 feature
failures; root confirmation 27 failures in 3.32s), initial focus 26 passed/1 skip
in 2.30s, and initial full suite 768 passed/9 skipped in 174.94s. Independent review
requested one P2 correction: migration/repository reads leaked OperationalError
under a real second-connection EXCLUSIVE lock in supported rollback journaling.
`M2A-T1-LOCK-RED-20261105` reproduced all eight boundaries (8 failed/9 deselected
in 2.51s; full focus 8 failed/26 passed/1 skipped in 18.45s).
`M2A-T1-LOCK-GREEN-20261105` passed 34 tests with one local symlink-privilege skip
in 4.70s after translation through the existing typed Busy boundary. Scoped
independent re-review accepted the correction and reported no remaining actionable
task 1 findings. The reviewer did not rerun the author's suites.

Final named run `M2A-T1-FINAL-20261105`: **778 passed, 7 skipped in 173.68s** on
Windows, CPython 3.13.15 and PySide6/Qt 6.11.2, with offscreen Qt, a synthetic
provider key and temporary roots/DBs. `HERE_WHEEL_PYTHON` selected an isolated
noneditable-wheel environment; both wheel smokes passed. Seven remaining skips
are six existing symlink-privilege cases and the new memory database symlink case.
Changed-file Ruff and format checks passed; isolated mypy passed with missing
imports ignored, while normal mypy retains the known flat-layout import limitation.
Local logs/review reports are under `.superpowers/sdd/M2_CATALOG_PLAN/` in the group
worktree; no private audio, user library, `.env` or real provider was accessed.

Task 2 has a local focused GREEN run (310 passed/1 skip in 24.60s), with its complete
suite still running; task 3 writes RED in a separate worktree. No process-kill file
publication, producer/catalog integration, FTS or lifecycle acceptance is claimed
by task 1. M2 criteria 8/14 have partial local evidence; criterion 15 requires
current-head hosted Windows CI. Hardware/provider/clean-install certification and
M2 closure are not claimed; the two-hour row remains omitted, not passed.

## 2026-11-05 — PR #20 ruling 19 documentation merge

PR [#20](https://github.com/srodriguez-lefebre/here/pull/20) merged at
2026-11-05T20:24:29Z with merge commit
`763a891fa7c0bed54a3158040094b6f083b88f32`, guarded by exact reviewed head
`2f25a0d9738e22d315b79b23a16ea3c8f3c324d3` using the admin merge-commit path.
The initial independent preparation review accepted the documentation; a distinct
reviewer accepted the follow-up M5 release-plan correction.

Copilot inline `4485511464` identified stale credential requests in
`M5_RELEASE_PLAN.md`; correction `2f25a0d` requests only the clean machine and uses
the already authorized `HERE_PROVIDER_SMOKE_KEY` for opt-in synthetic provider
smokes. Reply `4485574698` records the disposition and the thread is resolved.
Current-head Copilot review `5791519593` is **COMMENTED**, with no active issues;
it is not represented as APPROVED. Exact-head Windows CI `42102225168` passed
(2m36s) and Windows package workflow `42102224714` passed (3m34s). Root confirmed
all four review/CI gates before merge. Integrating this documentation merge into
the M2 branch does not constitute a provider run or M2 feature acceptance.

## 2026-11-05 — M2 A task 2 local checkpoint

Task source `037fc036ef43467136bcf7b8e95c6f8395d1aa5e` was integrated as
`1cdaadb`. Independent reviewer `catalog_review` accepted SPEC compliance and code
quality with no findings on that exact task head. The reviewer inspected source
and submitted evidence rather than rerunning the suites. This checkpoint is
implemented and automatically verified locally; group A and M2 remain open.

All five completion paths preserve source segments and `raw_transcript.txt`
independently of optional display cleanup. Text-only responses gain an untimed,
speakerless evidence segment; source order and request/chunk speaker scope remain.
Nullable additive timing provenance retains historical compatibility, valid
one-sided measurements and absent bounds without fabricated time. CLI publication
now delegates to `SessionProcessor.transcribe_audio_path`; input ownership,
pending manifests, failure causes and conservative replaced-audio cleanup remain.
Managed raw entry preflight and metadata-last I/O rollback preserve prior raw
bytes or remove a fresh unpublished raw file after a handled metadata failure.

Named runs in `ACCEPTANCE.md` and `M2_CATALOG_EVIDENCE.json`:

- `M2A-T2-RED-20261105`: 82 failed/153 passed/1 skipped in 21.18s.
- `M2A-T2-RAW-RED-20261105`: 2 failed/112 deselected in 2.12s before the raw
  rollback correction; both regressions pass in the final focus.
- `M2A-T2-FOCUS-20261105`: 310 passed/1 symlink skip in 24.60s, including real
  synthetic-WAV seek across a virtual five-minute wall pause.
- `M2A-T2-FIRST-FULL-20261105`: 8 failed/775 passed/8 skipped in 171.12s. Existing
  fault wrappers rejected additive `raw_text`; narrow forwarding changes preserved
  all original assertions and process-kill phases.
- `M2A-T2-CAPTURE-RECOVERY-20261105`: all 68 passed in 25.64s, including the real
  killed-subprocess and synthetic crash boundaries exposed by that first full run.
- `M2A-T2-FINAL-20261105`: **783 passed, 8 skipped in 160.71s**, Windows CPython
  3.13.15, offscreen Qt and synthetic provider key. Six skips require symlink
  privilege; two noneditable-wheel smokes lack `HERE_WHEEL_PYTHON` in this run.

All 13 touched files pass Ruff and formatting; diff checks pass. Mypy comparison
on seven production files is baseline 60/current 58 diagnostics, no new messages;
existing import/stub/Any/SDK issues remain. No private data, `.env`, hardware or real
provider was accessed. This task's handled-I/O rollback is not a filesystem
transaction against process termination/concurrent writers; catalog immutable
publication and reconciliation still belong to task 3. No current-head hosted CI,
provider/hardware/installer certification or M2 closure is claimed.

Task 3 service/artifact/controller RED remains 31 failed/18 passed in 2.50s,
confirmed by root in 4.25s. CLI RED is being completed after task 2 released its
shared files; no task 3 GREEN is claimed at this checkpoint.

Follow-up task 1 typing evidence: a temporary canonical `here/` package containing
unchanged copies of the four memory source files and the existing output path
validator passed `mypy --follow-imports=silent --no-incremental -p here.memory`
with `MYPYPATH` selecting that temporary root: **Success: no issues found in
4 source files**, without `--ignore-missing-imports`. Log:
`task-1-mypy-layout-checked.log`. Root removed only the verified owned temporary
directory afterward; no dependency or production layout changed. The ordinary
global flat-layout mypy baseline remains unclean.

Task 3 subsequently completed combined CLI/service/artifact/controller RED:
39 failed/18 passed in 7.33s (`task-3-complete-root-red.log`). A distinct agent is
implementing GREEN; no passing result or task 3 completion is claimed yet.
