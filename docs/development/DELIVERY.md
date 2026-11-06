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

29. **Ruling:** el arnés sintético de captura espera de forma acotada una
    confirmación real de que su backend fake terminó el bootstrap antes de devolver
    el helper a la prueba del dispositivo. La confirmación no simula `ready` del
    protocolo: enumeración, apertura, lectura y cierre siguen siendo ejercidos con
    sus límites y watchdogs originales. No se cambia ningún deadline del producto.
    En los tests de consumidor/pipe bloqueados, progreso primario se observa sin
    abrir el journal activo; el journal se inspecciona tras el cierre real del hijo.
    Motivo medido: un fake sano agotó el lease de 1,5 s durante imports antes del
    request; un lector abierto reprodujo `WinError 5` en el writer sin cambios.
    La comparación pre-M2 A también falló 15 casos de 33; no se atribuye a la tarea
    3 ni se presenta una causa global no probada. **Cost if wrong:** rehacer la
    instrumentación del fixture y su evidencia, manteniendo intactos los límites
    operativos y criterios de captura.

30. **Ruling:** cada staging de publicación registra antes de copiar un plan
    propio `.publication.json` con versión, ID de operación y hashes de nombres
    reconocidos. Un UUID de carpeta por sí solo no prueba propiedad de sus hijos.
    La limpieza usa únicamente ese plan validado o los nombres registrados por la
    operación persistida; archivos o subdirectorios desconocidos se conservan y
    producen advertencia/limpieza diferida antes de borrar. Los snapshots de estado
    también validan nombre generado, versión e identidad antes de una purga propia.
    **Cost if wrong:** versionar el lector del plan de staging y reintentar limpieza
    local; material desconocido y fuentes originales permanecen preservados.

31. **Ruling:** cada revisión registra explícitamente los assets de audio que
    consumió, incluso si sus bytes ya existían y no se copian otra vez. La
    migración 2 agrega `revision_audio_assets` con reunión, revisión, asset, ordinal
    y nombre de fuente; las claves foráneas compuestas impiden cruzar reuniones y
    ordinal/nombre son únicos dentro de la revisión. No se altera el checksum de
    la migración 1. `MemoryService.show` y su JSON CLI exponen
    `revision_audio_assets`. El recibo de publicación versión 2 conserva esos
    vínculos y solo sus assets; rebuild restaura la misma asociación A→B→A, sin
    elegir audio por orden de archivos o por conjetura sobre el hash agregado.
    Un recibo antiguo sin prueba inequívoca no autoriza inventar el vínculo:
    material incierto se conserva y se advierte. La biblioteca real 0.2.0 no tenía
    este catálogo; esto no autoriza reescribir sus sesiones v1. **Cost if wrong:**
    nueva migración/versión de recibo y adaptación del lector/rebuild y de M2 B,
    conservando evidencia inmutable y asociaciones ya comprobadas.

32. **Ruling:** la protección de una captura empieza antes de publicar su journal,
    mientras el factory todavía puede estar esperando apertura/`ready`. El handle
    nativo mantiene una lease real en
    `<sessions_root>/.captures/.owners/<capture_id>.lock`, con componentes validados,
    además de la lease de catálogo del ruling 23. El lock se adquiere antes de crear
    o publicar el journal; discovery y materialize prueban ambos locks antes de
    inspeccionar media cambiante o intentar persistencia/cleanup. Ausencia de la
    fila de catálogo, PID o edad del journal no prueban abandono. El handoff entre
    handle, sesión y controlador no deja una ventana sin dueño; la lease se libera
    después del cierre real del helper y del desenlace de persistencia/publicación,
    incluso si el catálogo falló. Apertura fallida y cleanup cierran idempotentemente
    cuando ya no hay trabajo propietario. Se permite extraer el lock genérico a
    `output/ownership.py` y ampliar solo la propiedad en `recording/journal.py`,
    `isolated.py` y, si hace falta handoff, `models.py`; sin proceso residente nuevo,
    cambios de protocolo ni deadlines. Es una corrección de F5, con RED independiente
    y re-review pendientes, no evidencia de una carrera observada. **Cost if wrong:**
    ajustar el handoff/cierre y los lectores de recovery, preservando journals y
    audio; nunca sustituir propiedad real por heurísticas.

33. **Ruling:** un productor explícito sobre un v1 todavía sin mapping resuelve
    primero su identidad canónica mediante importación local validada, antes de
    normalizar, escribir pending o llamar al proveedor. Para el `meeting_id=None`
    legado y para el ID opaco que el productor M1 ya acepta, conserva exactamente
    el valor en metadata y en la entrada al proveedor; `meetings.id` sigue siendo
    UUID y el vínculo usa origen normalizado, huella y revisión activa esperada.
    El permiso para el ID opaco pertenece solo al recorrido interno confiable,
    no a la importación ordinaria, que conserva su validación UUID estricta. No
    se crea una segunda identidad determinista solo para saltar la propiedad.
    Si el catálogo no puede resolver la identidad, el productor falla de forma
    recuperable antes de mutar la sesión o ejecutar al proveedor. Es cobertura
    de F2/F5, con RED y review pendientes, sin comando/flujo nuevo ni modificación
    del origen v1. **Cost if wrong:** ajustar resolución y handoff de ownership,
    conservando el ID legado original, la evidencia previa y los mappings válidos.

34. **Ruling:** precisión de implementación del ruling 32: la lease nativa vive en
    `<sessions_root>/.capture_owners/<capture_id>.lock`. Así `.captures/` conserva
    su colección de directorios UUID de journals, utilizada por sus lectores
    existentes. Un scope interno por trabajo del controlador, propagado mediante
    `ContextVar`, registra la lease antes de publicar y la retiene hasta su salida,
    después de reaping y persistencia/publicación, incluso si el constructor falla
    antes de devolver el handle. El scope no agrega un proceso ni una autoridad
    persistente: mantiene el handle del lock OS. Fuera del controlador, `wait` y
    fallo de apertura conservan su contrato de liberar tras cierre real para permitir
    recovery directo. La extracción de lock no cambia límites operativos.
    **Cost if wrong:** ajustar el scope/handoff y sus lectores, manteniendo la
    protección de todo el trabajo y la recuperación de la API directa.

35. **Ruling:** por nueva instrucción explícita del usuario, primero se cierra M2 A
    completo con review, suite estable y todos los gates de Copilot/CI/merge. Solo
    después de ese merge y de completar los PRs de documentación de contratos
    necesarios, root crea dos chats separados del proyecto: M2 B (ciclo de vida y
    borrado) y M3 A (búsqueda/acciones), cada uno con rama y worktree propios. Antes
    de iniciar, cada brief incluye objetivo, SPEC/PLAN aceptados, SHA base real,
    archivos propios y compartidos, dependencias, límites y DoD verificable. Los
    chats ejecutan sus grupos completos con subagentes RED/GREEN/review/docs; root
    sigue como único orquestador de contratos, leases por archivo, integración y
    merges. IDs de chats, ramas, worktrees y responsables se registran en STATUS
    cuando existan; ahora están pendientes y la SHA del merge A no se inventa.
    Después de M2 B puede adelantarse M5 B con claims pendientes hasta M4 B; el
    resto conserva sus dependencias. Un writer por archivo; migraciones,
    pyproject/uv.lock y versión en serie; suites sensitivas sin otros tests/builds
    pesados. Calidad, privacidad, aceptación y gates de merge siguen intactos. No
    se reinicia ni duplica trabajo realizado; no se crea ningún chat en este
    checkpoint. **Cost if wrong:** conflictos o duplicación entre frentes y
    contratos desalineados; root los evita con briefs sobre la base mergeada,
    leases exclusivos, PRs docs previos y verificación de integración.

36. **Ruling:** cada publicación explícita usa un `publication_id` UUID nuevo,
    incluso si su evidencia coincide con la anterior. El marcador versión 2
    conserva ese token; el payload de operación y el recibo versión 3 registran
    token, padre esperado, origen normalizado, huella y revisión comprometida.
    Solo esa prueba durable exacta autoriza replay idempotente de una publicación
    ya aplicada y la retirada de su marcador, también después de reconstruir la
    DB desde recibos. Rebuild restaura las operaciones completas probadas por los
    recibos v3; acepta recibos v2 sin inventarles provenance. No se agrega una
    migración: se reutiliza `pending_operations`. Token ausente, diferente o sin
    commit probado no autoriza aceptar un padre stale ni borrar intención incierta.
    El controlador consume una vez el resultado privado ya publicado por su
    `SessionProcessor`; procesadores ajenos mantienen el fallback. Ese handoff
    evita un segundo callback de publicación, pero no sustituye la prueba durable
    del replay. **Cost if wrong:** versionar los lectores de marker/recibo y el
    handoff, conservando evidencia, padres y marcadores inciertos.

37. **Ruling:** el retry legado de una carpeta con WAV válido y sin `session.json`
    conserva el uso de M1. Antes de mutar o llamar al proveedor reserva una UUID
    canónica y su lease mediante origen normalizado y huella medida del WAV;
    `ingest_sources.revision_id` puede ser NULL, con cero revisiones anteriores.
    No fabrica un manifiesto v1 para importarlo ni una transcripción histórica.
    `source_kind=legacy_audio_only` queda en la provenance durable y en los recibos
    para conservar `started_at`/`completed_at` de captura NULL durante retries y
    rebuild, aunque el v1 de compatibilidad posterior contenga fechas inferidas.
    Los timestamps reales del procesamiento/publicación siguen siendo evidencia
    de esos eventos, sin convertirse en fecha de la grabación desconocida. No
    cambia la validación de rutas/roles ni se reescribe un ID legado opcional.
    **Cost if wrong:** ajustar reserva, provenance y lectores, manteniendo audio,
    identidad y fechas desconocidas sin conjeturas.

38. **Ruling:** para cerrar GR1, el writer de artefactos ofrece un callback interno
    justo antes del replace final de `session.json`, después de publicar los
    artefactos subordinados. El productor registra primero un marcador versión 3
    con el mismo token, padre esperado y source kind, más el hash del manifiesto
    prospectivo y un sello de hashes de sus artefactos con roles validados.
    Recovery solo publica si toda esa evidencia final coincide; una intención
    previa sin resultado final coincidente se conserva con aviso, sin importar
    el resultado viejo ni ejecutar al proveedor. Se mantiene lectura de markers
    1/2 y recibos 2/3, schema 2 y conflicto estricto de UUID/parent. Para GR3, el
    controlador real difiere la primera publicación de fallo de captura hasta
    sellar el resultado bajo el cutover de cancelación existente; la llamada
    directa a persistencia y los procesadores inyectados conservan su contrato.
    Una operación conserva una sola identidad/publicación, el audio y la causa
    original de captura. Hashes, copias y proveedor quedan fuera de transacciones
    SQLite. No cambia ningún flujo, dependencia, proceso, deadline, versión ni
    definición de terminado. Es diseño autorizado para una única tanda GR1–GR3,
    implementado en la tanda integrada `7bfd0cb`, con RED/GREEN/foco local y
    re-review independiente final APPROVED. **Cost if wrong:** ajustar el sello,
    su compatibilidad y el handoff de cutover, conservando evidencia incierta y
    las precondiciones de revisión, sin aceptar como final un intento incompleto.

39. **Ruling:** para las fechas de captura desconocidas de R37, M3 lista por
    `started_at DESC NULLS LAST, meeting_id ASC`. La búsqueda ordena por rank
    ascendente y luego la misma fecha/ID, con `segment_id ASC` como desempate
    final. La paginación usa cursor opaco versionado, vinculado a sus filtros,
    y representa NULL explícitamente en su tupla; nunca usa una fecha ficticia
    ni el timestamp de procesamiento como sustituto. Sin límites de fecha se
    incluyen reuniones sin fecha; un filtro `from` inclusivo o `to` exclusivo
    las excluye mediante la semántica SQL de NULL. La UI conserva la fecha como
    desconocida. El fallback conserva estos desempates para su ranking propio.
    Es precisión interna de los defaults de paginación del SPEC, sin nueva
    pantalla, dependencia, retención ni criterio de terminado. Queda preparado
    para el PR de contratos posterior al merge de A; no inicia M3 ni modifica
    el head publicado de PR #21. **Cost if wrong:** ajustar cursores/ordenación
    y sus tests antes de publicar M3, manteniendo determinismo y fechas NULL.

40. **Ruling:** se corrigen las tres observaciones de Copilot recibidas en PR #21
    sobre `3394149` sin cambiar producto ni terminado. El callback previo al commit
    forma parte de la publicación atómica: toda excepción ordinaria suya ejecuta
    el rollback de evidencia antes de propagarse; una terminación de proceso no
    se convierte en éxito ni se oculta. El procesador por defecto comparte el
    `MemoryService` del controlador, incluida una raíz inyectada, conservando los
    procesadores explícitos del caller. El replay local rechazado de una sesión
    completed queda visible por el mecanismo de advertencia existente, con
    candidato no reintentable (`can_retry=False`) si se usa la lista de recovery;
    nunca ofrece otra transcripción al proveedor, modifica el resultado final
    ni borra su marcador incierto. Cada corrección exige RED independiente,
    GREEN, review independiente del diff, nueva solicitud/recepción de Copilot y
    CI verde en el head nuevo. Las corridas previas siguen siendo históricas.
    **Cost if wrong:** ajustar rollback, composición e informe de recovery con
    los mismos datos/contratos; preservar audio, evidencia anterior y avisos reales.

41. **Ruling:** M3 obtiene los hits parciales opt-in mediante una rama SQL literal
    de solo lectura sobre segmentos de la revisión activa parcial. El índice
    persistente sigue conteniendo solo revisiones activas completas, mantenido por
    A/B5. Con FTS disponible, sus hits BM25 se combinan con la rama parcial literal
    de rank uniforme 0; sin FTS, ambas ramas usan el fallback literal de rank 0.
    Se consultan los mismos campos autorizados, se escapan también los comodines
    de LIKE y se aplican visibilidad/filtros antes de cursor/LIMIT. R39 ordena la
    unión, cuyo cursor incluye modo/backend e include_partial. Snippets de texto
    real acotados y badge/timing reales; reindexado no altera la rama parcial ni
    crea otro escritor. Es precisión del opt-in y fallback ya aprobados, con
    verificación de equivalencia pendiente en B/M3. **Cost if wrong:** ajustar la
    consulta literal/ranking y sus tests antes del cierre M3, preservando BM25,
    la visibilidad por defecto y la evidencia de origen.

42. **Ruling:** B conserva la misma supresión mínima ya definida por
    `deleted_origins` en un registro versionado bajo `<data_dir>/.deletions/`,
    fuera del árbol que se purga: UUID, pares de origen normalizado/huella y
    control pending/complete. Este registro no guarda texto, audio, títulos,
    participantes ni credenciales. Su replace/flush durable, después de validar
    la confirmación, propiedad real e inventario, es el punto de aceptación del
    borrado y precede a la primera transacción delete. Desde ahí cualquier Busy,
    fallo de archivos o backups se informa pending; la supresión ya aceptada se
    aplica antes de exponer filas, reimportar o reconstruir recibos/FTS, incluso
    si el catálogo SQL se perdió. Los lectores B/M3 comparten ese guard antes de
    LIMIT y la UI invalida callbacks anteriores de su operación de borrado.
    Reconcile restaura supresión/orígenes y estado delete antes del rebuild
    normal; recibos sobrevivientes sirven solo para probar inventario de limpieza.
    Registro inválido/inaccesible falla la exposición de catálogo de forma cerrada
    con error recuperable, sin autorizar una purga inferida. Desconocidos se
    conservan y mantienen pending. Solo tras verificar archivos, DB/WAL y backups
    se marca complete durable; el tombstone mínimo sigue reconstruyendo
    `deleted_origins`. Otra identidad/origen explícito con igual audio sigue
    permitido. Se extiende el journal/reconcile existente, mismo data_dir y misma
    política de borrado/supresión, con un solo escritor y sin proceso nuevo.
    **Cost if wrong:** ajustar barrera/guards y recuperación de inventario antes
    de implementar/certificar B, conservando supresión y evidencia incierta;
    nunca declarar purga completa por un registro pendiente o solo por un UUID.

43. **Ruling:** la observación `4488285573` de la nueva review de Copilot
    `5794829768` se corrige dentro del mismo aviso de R40. Un manifiesto de sesión
    válido con marcador pendiente malformado conserva sesión, marcador y audio y
    produce un candidato de advertencia no reintentable, con mensaje acotado y sin
    reproducir el contenido inválido. El fallo se captura solo en la validación
    local del marcador, preservando ownership y rechazo de rutas inseguras; los
    vecinos válidos siguen su replay y nunca se invoca al proveedor. No se inventa
    un padre/token para reparar el marcador. Tests RED independientes, foco de
    recovery/compatibilidad, wheel nuevo y ambos smokes locales preceden a la
    publicación. La suite completa sobre la fuente corregida se verifica en CI
    Windows offscreen del head exacto antes del merge; el full local 5 queda
    histórico de c3cdf08. Se mantienen cobertura y los cuatro gates, sin repetir
    otro full local del mismo head además del full obligatorio de CI. **Cost if
    wrong:** ajustar la clasificación de errores/advertencia y su cobertura,
    manteniendo evidencia incierta y el botón de proveedor deshabilitado.

44. **Ruling:** la observación `4488473222` de Copilot `5795057762` se corrige
    preservando R33/R40/R43. Si el catálogo no permite resolver ownership de un
    manifiesto legado válido con ID ausente u opaco, discovery devuelve solo su
    advertencia de biblioteca no reintentable, con texto fijo acotado. No inspecciona
    audio, materializa, inventa UUID/padre, modifica fuente o catálogo corrupto,
    ni llama al proveedor. El fallo de resolución de catálogo se distingue del
    lock de un dueño real activo y de un rechazo de ruta; esos guards siguen
    impidiendo inspección/mutación. Se prueban corrupción SQLite real, identidad
    legada y controles sanos/owner activo con RED independiente y review del diff.
    La verificación completa mantiene la ruta R43: focos y wheel/smokes locales,
    suite completa Windows en el head exacto y Copilot nuevo recibido/respondido
    antes del merge. **Cost if wrong:** ajustar clasificación/advertencia y sus
    lectores, conservando evidencia incierta y bloqueo de toda acción sin ownership.

45. **Ruling:** se corrige la observación `4488791517` de Copilot `5795454802`
    sobre el PR de contratos #22. Los cursores de `search` atan las claves R39 a
    una generación lógica reproducible del corpus, calculada en el mismo snapshot
    de lectura que la página y con el guard R42 coherente. Incluye todo el corpus
    que afecta BM25, aunque una fila no coincida, un filtro la excluya o ya esté
    suprimida pero siga en FTS, y los
    metadatos/relaciones que afectan visibilidad; no solo los hits de la consulta.
    El default es un digest de metadatos existentes, identidades de revisiones
    inmutables, títulos y supresión validada, solo donde prueban equivalencia con
    el corpus FTS real; mantenimiento del índice no cambia el rank sin invalidar
    ese token. Orden/encoding canónicos y transacción de lectura explícita hasta
    terminar de obtener la página, sin migración ni caché persistente.
    No se compara `PRAGMA data_version` entre conexiones distintas. Si cambió la
    generación, se rechaza el cursor antes de usar su rank con `InvalidInput` y
    razón estructurada `corpus_changed`; la UI reinicia la consulta desde la
    primera página mediante sus requests existentes, sin anexar una página vieja.
    El listado `MeetingQuery` conserva su keyset fecha/ID y su contrato original
    frente a inserciones: no se invalida por esta regla de búsqueda. Se prueban
    mutaciones desde otra conexión, incluidas inserciones no coincidentes, y la
    continuidad entre conexiones con corpus estable. No cambian criterios del
    hito, flujos públicos, retención, índice persistente, dependencias ni umbrales
    p95/heartbeat. **Cost if wrong:** ajustar fingerprint/cursor y su recuperación
    interna antes de cerrar M3, conservando resultados completos, guards de borrado
    y la paginación de listado ya definida.

46. **Ruling:** para el handoff de modelos B/M3 se fijan defaults internos del
    SPEC sin cambiar funciones de producto. `MeetingQuery.text` filtra título;
    `SearchQuery.text` consulta texto. Fechas de query son strings ISO aware,
    validadas/canonizadas a UTC por el servicio; bounds nulos conservan su sentido.
    Ambos usan limit50, cursorNone, filtros vacíos e include_partialFalse; el opt-in
    de parciales rige listado y búsqueda como ya exige el hito. `MeetingPage.items`
    contiene `MeetingSummary(meeting, audio_available=False, partial=False)`, sin
    copiar ni inventar campos de Meeting; los tests acceden a `item.meeting.id`.
    Se aceptan los demás campos propuestos en el reporte M3 `shared-dto-request.md`
    para SearchHit/SearchPage/SearchCapabilities/CitationTarget, como dataclasses
    frozen/slots, con tuples inmutables. Disponibilidad se deriva de enlaces de
    revisión/estado registrado, no de recencia ni lectura de archivos por la UI.
    Se declara también `Citation(meeting_id, revision_id, segment_id, quote=None,
    start_char=None, end_char=None)`, solo referencia; el resolver M3 B verificará
    texto/offsets y ruta desde memoria. Target no resuelve ni autoriza audio en M3 A.
    `InvalidInput` canónico se expone desde memory.service, con razón estructurada
    keyword `reason` por defecto invalid_input; B define una sola clase y M3 la
    reutiliza, en particular corpus_changed. La factory retorna LibraryConnection
    nueva por operación y el consumidor la cierra. Models se entrega temporalmente
    tras congelar B4 y observar RED M3; ningún segundo writer, tipo duplicado o
    migración para este handoff. **Cost if wrong:** adaptar campos/serialización y
    callers antes de aceptar M3, sin rebajar los contratos de consulta, citas,
    borrado o fechas desconocidas.

47. **Ruling:** la query vacía normalizada de `SearchService.search` delega a
    `list_meetings` con los mismos filtros, límite y cursor; retorna MeetingPage,
    mientras una búsqueda no vacía retorna SearchPage. Se explicita la anotación
    interna `SearchPage | MeetingPage`: no se fabrica un SearchHit/segmento para
    representar una reunión, no se ejecuta MATCH vacío ni se amplía visibilidad.
    La UI sigue despachando texto vacío al listado. Su cursor fecha/ID conserva
    continuidad y queda fuera de la generación de búsqueda R45. Default completo
    y opt-in de revisión activa parcial siguen como el SPEC, sin inventar una
    revisión para reservas. Es precisión del flujo vacío ya acordado en §3 del
    diseño, no un comando o pantalla nuevo. **Cost if wrong:** adaptar anotaciones
    y despacho de consumidores conservando los dos DTOs, filtros y aceptación.

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

## 2026-11-05 — M2 A task 3 protocol and scoped evidence (pending final)

Candidate source remains in `m2-ingestion` on base `1cdaadb`, uncommitted at this
checkpoint. The protocol follows rulings 23–30: actual capture/processing owner
locks, local ingest/status markers, visible nonfatal warning, explicit source roles,
post-commit `catalog.json` receipts, append-only state snapshots, trusted legacy
retry lineage and registered-stage cleanup. The three maintenance CLI commands
share the local service and JSON exit codes 0/2/3/4. No final task/group completion,
stable final full-suite pass or independent task/group approval is claimed.

The source reserves and registers staging before copies, prepares the DB journal,
renames to immutable revision material, commits entities/evidence/FTS/activation,
then writes a completion receipt/state snapshot and marks the operation complete.
Rebuild uses managed committed receipts only; lost DB/journal plus missing receipt
leaves uncertain material retained with warnings. Imported source remains unchanged;
aware dates normalize to UTC and naive historical dates remain unspecified. Failed
provider retry updates only state and preserves the previous complete revision,
including when a local status marker must be replayed after Busy. Opening the app
never repeats provider work. The library warning remains separate from actual
capture/provider errors and is cleared by new work or clean publication.

Observed RED scopes include original 31 failures/18 passes (2.50s; root 4.25s),
eight CLI failures (5.94s), root combined 39/18 (7.33s), seven missing ownership
features (1.44s), wrong WAL for mapped remote storage (one failure, 2.00s), rebuild/
failed-file gaps (two Green failures, 6.41s), contained synthetic credential read
attempts stopped by traps (two author failures, 2.58s; implementer 3.37s), warning
UI field (three failures, 5.65s), provider-work ownership/state-only replay (four
author failures, 9.78s; implementer 9.26s), legacy retry duplicate UUID (one, 7.58s),
UTC offset storage (two, 2.46s), and unknown-stage file deletion (one, 2.43s).
No trap permitted an actual private credential read. The last preservation
amendment still needs final checks. Receipt privacy and audio/revision association
are concrete independent-review focus; required guarantees are not inferred from
staging or path containment alone.

Scoped GREEN results at distinct source states: 49 service/artifact/controller
cases in 28.07s, 50 CLI/processing in 36.99s, eight ownership/storage in 22.78s,
34 rebuild/service/artifacts/ownership in 53.42s, 27 warning UI/controller in 9.61s,
70 privacy/preflight/recovery in 86.61s, 48 combined memory in 51.99s, 85 processing/
ownership/recovery/CLI/controller in 79.17s, five legacy/processing in 14.10s and
two UTC in 1.83s. They cannot be summed or presented as the stable final suite.
The single old external-file identity expectation was adjusted by the independent
test author to canonical pending UUID preservation; existing-session identity
remained exact, and all three compatibility variants passed in 10.99s.

The first full diagnostic had changing source and overlapping focus/mypy work:
34 failed/822 passed/9 skipped in 363.35s. It exposed 14 early catalog-init/preflight
failures, later corrected without weakening assertions, the obsolete new-ID
expectation above, and capture opening/journal faults. Later isolated diagnostics
gave 21 failed/12 passed in 75.37s; stable quiet Green gave 17 failed/16 passed in
87.56s. A fully source-isolated prior baseline also failed 15/33 cases (18 passed,
87.08s), so those failures are not all attributed to task 3. A preliminary one-case
comparison passed at 7.41s/7.68s but its child still used the Green installation;
it is not represented as fully isolated source proof.

Independent diagnostics observed H1 in one normal fake launch: its original 1.5s
lease expired during bootstrap imports before request/device work. H2 reproduced
the Windows sharing mechanism on a disposable journal, with a held reader causing
WinError 5 and the unchanged writer succeeding after close; the original failing
case's exact overlap remains unproven. Ruling 29 authorized only test bootstrap
acknowledgement and checkpoint-based progress observation, keeping product
opening/read/close/watchdog limits, stalls and actual child reaping unchanged.
Author focus passed four cases/29 deselected in 19.71s; full isolated passed all
33 in 126.10s, each run alone. `capture_diagnostics` accepted SPEC/quality of fixture
SHA256 `665181979ffd5b0d354c473de9a6e1f12fbc3209f42561e1af5f99b0dad67ca1`
without rerunning suites. That review does not approve task 3.

Pre-amendment static evidence: Ruff/format on 24 files passed. Raw flat-layout mypy
had 232 errors/43 files; canonical-layout baseline/current had 79/79 diagnostics
over 62/64 modules with no new touched-file messages. Registered-stage preservation
was added afterward and requires fresh checks. Final stable source result/head,
independent task/whole-branch review, hosted Windows CI and received/responded
Copilot remain gates. M2 B needs its contract-adjusted documentation PR before
implementation; version 0.3.0 is only for closing B/M2. Two-hour acceptance stays
omitted, never passed. No hardware/provider/installer certification is claimed.

## 2026-11-05 — Task 3 frozen candidate, suite result and seven open findings

The candidate is now committed **only in `m2-ingestion`**: synthetic capture fixture
`efb73a92ea89bc53318ce1a811eaf61129a76fbf` and task source
`6c422f0b8671d2f736645abb6835d351783cbcc7`. All 27 file hashes match the independent
review manifest `task-3-candidate-files.json`; docs independently compared them
read-only. These commits establish immutable FIX_BASE, not task acceptance or
integration into the root catalog branch.

After the earlier checkpoint, registered cleanup and receipt/pending/state privacy
guards were verified. Stable memory/UI/controller/processing focus: 122 passed in
108.36s. Additional concrete privacy RED: three failures in 5.00s and one unknown
state-name access in 3.08s; traps stopped before reading any synthetic credential.
Shared role validation now precedes hash/read/delete, undeclared WAV inventory does
not confer ownership, and generated state validation precedes state reads.
Immediate post-fix focus passed 46 in 66.62s (earlier related focus 31/34.78s).
Earlier missing-guard observations describe superseded source and are not a finding
against this frozen candidate. Final candidate Ruff/format/diff pass on 27 files;
canonical mypy remains baseline/current 79/79, no new touched-file diagnostics.

The first stable full, `M2A-T3-STABLE-FULL1-20261105`, reported **6 failed,
877 passed, 9 skipped in 486.12s**. All six failures were the existing postcommit
audio-cleanup fault test's read/missing_metadata/invalid_json × completed/failed
parameters: its global metadata-read injection also counted the new independent
catalog consumer. The RED author scoped the injection/count only to the real
cleanup delegate while retaining exact-one, real JSON corruption, outcome, original
cause and all path/audio-byte assertions. Author ten-case focus passed in 16.76s;
Green transferred rerun passed ten in 17.10s. No product code or deadline changed
for this correction. First log/XML remain intact.

The second stable full, `M2A-T3-CANDIDATE-FULL-20261105`, ran alone against frozen
source and reported **883 passed, 9 skipped in 598.56s**. Its XML has 892 cases,
zero failures/errors and nine skips; first full XML has 892 cases/six failures/
zero errors/nine skips. Logs and XML are separately retained under
`.superpowers/task3-full-final.*` and `task3-full-final-2.*` in Green. Pytest wall
summary and internal JUnit durations are distinguished in `ACCEPTANCE.md`.
Seven skips need symlink privilege (including task 1's memory case); two wheel
smokes lack a noneditable environment. This is local automatic verification of
covered behavior, not hosted CI, a wheel pass or hardware/provider certification.

Independent static review by `ingestion_review` requests SPEC and quality changes:

- F1 P1: explicit immutable revision/audio relationship, including reused A→B→A.
- F2 P2: preserve imported legacy None identity in existing-session CLI trans.
- F3 P2: identical explicit retry must complete durable state/revision.
- F4 P2: report committed publication success after housekeeping failure.
- F5 P2: require actual owner before producer/provider or recovery mutation.
- F6 P2: malformed marker envelope/types must not abort valid siblings.
- F7 P2: typed input/conflict/I/O categories must preserve CLI 2/3/4.

All seven remain open. The reviewer ran no tests and the passing suite does not
resolve these static paths. F5 includes second-instance discovery→materialize→
persistence/cleanup reachability before owner exclusion; discovery alone does not
change durable catalog status, and no actual audio loss is observed or claimed.
Round-1 RED/fixes/re-review and a corrected stable suite remain required, followed
by whole-group review and the exact-head GitHub gates. Ruling 31 above authorizes
the pending migration 2/receipt v2/show/rebuild relationship fix without changing
migration 1 or inferring ambiguous old evidence. It is not implemented yet.

Evidence levels remain separate: candidate code exists with a passing suite;
automatic verification is local and scoped to its covered cases; no new real
hardware/provider/clean-install certification. Task 3, group A and M2 are not
accepted or closed. Version remains 0.2.0; 0.3.0 belongs to B/M2 closure. Preparation
of B's adjusted SPEC/PLAN stays a separate documentation PR gate, not authorization
to implement B or bypass the seven findings.

## 2026-11-05 — Task 3 round-1 integration, NB1 and five integrated failures

Root integrated the fixture/candidate/fixes as `1277e19`, `3370483`, `3a83482`
and `1f1bc88513840b84c388a61d9c077907cf2fba2e`. This matches Green's reviewed
`c28472b5bd651f4d7f7fe2f40449f5a23e325f67` after fixes `16263c4` + `c28472b`:
src tree `780529d42936c979740443c51f1022fd2bb99a5e` and tests tree
`d226358ec038837dba6e40790c77eac74e74b538` are identical. Read-only Git comparison
also finds no source/config/scripts/workflow differences. Integration is not
acceptance or merge of group A. Earlier candidate-only statements above retain
their original checkpoint scope.

Forty new independent RED cases passed together in 69.70s after fixes. Ruff and
format checks on 20 touched files passed; resolved mypy remains baseline/current
79/79 with no new diagnostic messages. Independent scoped static re-review
`task-3-fix-1-review-report.md` marks the original F1–F7 **ADDRESSED**, including
migration 2/explicit revision-audio links/receipt v2, trusted legacy identity,
identical explicit retry, postcommit outcomes, actual ownership, marker validation
and typed CLI errors. Migration 1 remains unchanged.

The re-review still requests changes for **Important NB1/P2**. A real controller
retry invokes publication again after the real SessionProcessor has committed:
its stale parent recreates the origin marker and returns warning/conflict. Actual
postcommit-before-origin-marker-removal replay is also not recognized as the
already committed publication. The fix must preserve one new immutable revision
per fresh explicit retry while making that publication's replay idempotent using
durable proof. Arbitrary stale-parent acceptance, unchanged for a fresh identical
retry, and deleting uncertain intent are not acceptable shortcuts. Round-2 RED in
`tests/test_memory_publication_replay.py` is being prepared/executed by its author;
no result has been delivered at this checkpoint. The review accepts neither task
3 nor the group.

`M2A-INTEGRATED-FULL1-20261105` closed at **5 failed, 920 passed, 7 skipped in
553.39s**. Root log/XML are `.superpowers/m2a-integrated-final-1.txt`/`.xml`:
932 cases, five failures, zero errors, seven symlink-privilege skips; internal XML
time is 552.876s. Both noneditable-wheel smokes passed. Pending diagnosis/fix,
without omitting or relaxing assertions:

- Three capture-recovery snapshots include newly introduced owner-lock files:
  `test_failed_manifest_copies_with_same_uuid_are_both_discoverable` and both
  canonical-interrupted/canonical-failed variants of
  `test_retry_copied_session_preserves_independent_capture`.
- `test_trans_command_completes_failed_session_audio` fails with an sf.info stub
  lacking subtype (command exit 1 instead of 0).
- `test_legacy_audio_only_session_still_retries` is blocked by missing session.json
  during legacy identity resolution.

Fresh CPython 3.13.15 wheel and sdist builds succeeded. The off-repo `-I` probe of
the isolated noneditable wheel verified all 65 installed Python files byte-identical
to source, schema 2 including revision_audio_assets, FTS5, empty foreign_key_check
and integrity_check=ok. `.superpowers/m2a-wheel-catalog-probe.json` records hashes:
wheel `bb7a49a9f5b5c6a6b25f48f39b036badc489bee3870c69b0024af8f0d61dd7bd`,
sdist `8ebd3a3a44d48df68d94dcb8748adf8af52dfde334465bc2bcecfa0d95513358`.
No hardware/provider was used. This is automatic local distribution/catalog proof,
not installer or real-device certification and not a passing integrated suite.

Evidence levels: round-1 code is integrated but task completion remains unaccepted;
automatic checks provide scoped passes plus the explicit five-failure full result;
no new real certification exists. Next: NB1 and all five failures, focused checks,
re-review, stable full and whole-group review, then current-head PR/Copilot/Windows
CI/guarded merge. No M2 A/M2 closure or 0.3.0 bump.

Ruling 35 records the user's revised orchestration order: only after A is merged
and required contract-doc PRs are completed does root create separate project
chats for complete M2 B and M3 A groups. Discovered project:
`local-e307e59e09194b354122137b16ef6d77`, host `local`, Git enabled, original root
`C:/Users/savar/Documents/proyects/here`. Actual chat IDs, branches, worktrees,
executors and merged-A base SHA are all pending; none was created. Four B/M3 A
SPEC/PLAN files have been prepared in `m2-segments` with no PR yet (reports
`lifecycle-plan-preparation-report.md`, `m3-plan-preparation-report.md`). Root owns
contracts, file leases, integration and merges; each future brief states actual
base/ownership/dependencies/limits/DoD before RED/GREEN/review/docs execution.
M5 B may advance after M2 B with claims pending until M4 B; all other dependencies,
privacy/quality/acceptance/merge gates remain intact. Sensitive suites keep exclusive
windows. No previous work is restarted or duplicated.

## 2026-11-06 — Task 3 rounds 2/3 integrated, full 2 green, group review pending

Root `8087e252112fd2e6de1efeff1d0f3f531d4be77d` matches Green
`4a3cb7ced77a73046098e20cb34aeb00af0e6f61` after round-2 `944e358` and round-3
`4a3cb7c`; src tree `948efa342c9569ca751cffb4718d82f3dd872de6`, tests tree
`30c62e6f7e9e0a818e125a6c7762db91335b47f1`. Read-only docs-author Git verification
confirms matching trees and empty source/config/scripts/workflow diff.

Rulings 36/37 are implemented. Marker 2 includes a fresh publication UUID and
source_kind. Operation payload and receipt 3 retain exact token/expected parent/
normalized origin/measured fingerprint/committed revision/source-kind proof;
rebuild restores proven complete operations. Receipt 2 remains accepted without
invented proof; DB schema stays 2. The real processor's private manifest-bound
result is consumed once by the controller, while durable replay independently
covers real postcommit child crashes with existing/rebuilt DB and equal/changed
evidence. Fresh explicit identical retries still produce a new revision. Valid
WAV-only legacy retry reserves canonical UUID/lease before mutation/provider,
with NULL origin revision and zero historical transcript revisions; durable
source_kind preserves NULL capture dates across retry/rebuild. No process,
protocol, device/test deadline, dependency or version change was made.

Independent scoped round-2 review marks NB1 and the named integration regressions
ADDRESSED. Final covering focus: 61 passed in 68.06s; existing controller/processing/
UI: 69 passed in 23.02s. Metadata-only inspection does not create new lock files;
journal mutation upgrades ownership without dropping current leases. The CLI
fixture author supplied real PCM16 instead of invalid bytes/incomplete sf.info;
original outcome assertions remain and production validation was not weakened.
An intermediate replay focus reported 16 passed/1 failed in 315.60s, a 30-second
child timeout before commit proof/replay. Identical unchanged case passed alone
in 5.02s and in final combined focus (call 2.72s). Mechanism remains unproven; no
timeout/assertion change or root-cause claim is made. Logs remain retained.

Round-2 review introduced NB2: missing-metadata capture identity had to be validated
before deriving the journal probe path. Independent author RED: two failures/one
opaque-existing-manifest control pass in 2.45s. Round-3 scoped review
`task-3-fix-3-review-report.md` marks NB2 ADDRESSED, no new Critical/Important
breakage in that diff; covering runs 3 passed/2.40s and 27 passed/20.86s.
Existing manifests retain optional/opaque identity. Ruff/format clean on ten and
two files respectively; resolved comparative mypy baseline/current 79/79 with no
new diagnostic messages, no globally clean claim. Reviewers inspected supplied
logs/source and ran no tests.

`M2A-INTEGRATED-FULL2-8087e25`, started alone at 2026-11-06T00:01:58Z, completed
**940 passed, 7 skipped in 325.09s**, zero failures. Root log/XML:
`.superpowers/m2a-integrated-final-2.txt`/`.xml`. Independent read-only XML confirms
947 cases, zero failures/errors, seven symlink-privilege skips, internal 325.029s;
both fresh noneditable wheel CLI/GUI smokes passed. Earlier full 1f1bc88's five
failures/920 passes/seven skips/553.39s remain historical evidence above; all five
have later focused coverage and addressed scoped review. No assertion was omitted
or relaxed to count this run as passing.

Fresh CPython 3.13.15 wheel/sdist and off-repo `-I` noneditable probe of the exact
8087e25 source verify 65 Python files byte-identical, DB schema 2, FTS5, empty FK
check and integrity_check=ok. Probe
`.superpowers/m2a-wheel-catalog-probe-8087e25.json`, artifacts
`.superpowers/m2a-final-dist-8087e25`; wheel SHA256
`4f37ca61d50df87e588ecdd1c2e9221690fd3aa9472dc33081c4116600147ad8`, sdist SHA256
`b5110b1dd0c78c485e52a738e147f21a5a64e6dfaa5aaf515ae68b8d10de5271`.
Prior probes remain historical with their own hashes. This is local automatic
verification, not real hardware/provider/clean-install certification.

Whole-group review by `catalog_group_review` is still in progress on
763a891..8087e25 with three provisional Important findings: completed metadata can
commit before the first pending marker, leaving a kill without startup replay and
an interrupted row; non-object receipt JSON can reach `.get` and abort valid rebuild
siblings; accepted cancellation after capture-failure publication leaves cancelled
UI/manifest but failed catalog, conflict and a stuck marker. The complete source
report `group-review-report.md` records GR1/GR2/GR3 Important/P2 OPEN and source
verdict changes requested. Its global assessment remains provisional pending the
committed docs supplement. No final all-source/docs acceptance or fixes are claimed.
Scoped NB1/NB2 acceptance and a green full do not accept Task 3/group A/M2. Await
final findings, RED/GREEN/re-review and verification after any source correction,
then current-head PR/Copilot received/responded/Windows CI/guarded merge gates.
There is no M2 A PR or hosted group CI approval yet.

Rulings 20–37 remain unchanged. R35 ordering is preserved: finish/merge A and
required contract-doc PRs before root creates separate complete-group project chats
M2 B and M3 A. Future chat IDs/branches/worktrees/executors/merged-A base remain
pending; none created. Root alone orchestrates leases/contracts/integration/merges,
with serialized migrations/pyproject/version and exclusive sensitive suites.
Prepared four B/M3 A docs in m2-segments were not changed by this writer. Version
remains 0.2.0; 0.3.0 only at B/M2 closure; final 1.0.0 goal and real acceptance
requirements remain intact.

## 2026-11-06 — Combined GR1–GR3 wave integrated, full 3 in progress

Root `7bfd0cbf477a42aacc44b83b3d2da1e438801756` integrates Green
`ec55b9564d3af01247d3367675a340a361f6b6ef`. Read-only docs-author Git inspection
confirms matching source tree `2b4036f7deb905efb0a8836f378d267a066564b4` and tests
tree `a4310416bd10dc922b49e010d8495a98bfdfd585`; root reports empty source/tests/
config/scripts/workflow diff. Exactly six production files and three independently
authored new regression files; no dependency, schema migration, version, process,
protocol, deadline or product-flow change. Worker tracked files are clean; existing
untracked uv-setup.log was preserved. This is integration, not acceptance or merge.

The single whole-group source/docs review is final changes requested on
763a891..8087e25 plus committed docs supplement e00dd08: GR1–GR3 Important/P2,
GD1/GD2 Minor/P3. Minor checkpoint corrections are committed in 7be93af. There is
one scoped independent re-review of the combined wave and documentation, not a
second whole-group review or restart of NB1/NB2. Provisional source inspection
considers GR1–GR3 ADDRESSED with no new actionable breakage; final source/docs
verdict awaits committed documentation and actual full 3 evidence. Keep GR1–GR3
OPEN until that final independent verdict. Hosted gates remain pending.

R38 implemented: all final producer paths call the output writer's internal
before_metadata_commit callback after subordinate artifacts and before the actual
final session.json replace. Marker3 retains token, expected parent, original
optional ID, source kind and intent, adding exact role-validated hashes including
the actual staged prospective manifest bytes. A pending/unmatched final outcome
retains source/marker bytes with visible uncertainty through the existing
RecoveryCandidate.error_summary; it cannot adopt the old result or invoke the
provider. Markers1/2 remain readable without invented seal/proof; receipts 2/3 and
DBschema 2 stay unchanged. Real controller capture-failure publication is privately
deferred through the existing cancellation cutover so one accepted cancelled
outcome is sealed before first catalog ingest. Canonical and None-source-ID shapes
retain one identity/partial revision, original capture error and audio. Direct
persist_capture_failure and injected processors keep their contract. Receipt
object/nested container/hash validation isolates malformed meeting trees with an
identified warning while valid siblings/repeat discovery/maintenance JSON work.

Independent RED original 12 cases:11 failures/1 passing interrupted-attempt control,
four sequential retained logs totaling 27.48s. Initial GREEN 12 passed/21.67s. Affected
21-file focus 404 passed/1 symlink-privilege skip/159.03s precedes only the narrow
pending-v3 warning correction. The 13th additional pre-final-replace control was
authored without baseline execution; actual run against the first candidate gave
1 failure/5 deselected/3.93s, exactly absent visible uncertainty warning while old
R1/interrupted state/source bytes/marker were preserved. After the narrow correction,
final group plus recovery focus 26 passed/34.46s (all 13 group cases plus malformed
marker, legacy-discovery ownership and NB2 controls). These overlapping runs are
not summed as one suite. Raw logs .superpowers/group-fix-*.txt and implementation
report group-fix-implementation-report.md are retained. Docs read final summaries,
source diff and report; docs executed no tests/imports/build/mypy.

Nine-file Ruff/format checks pass. Canonical mypy against8087e25: baseline/current
79 errors in 20 files/65 source files; new touched-file diagnostic messages {}.
This is comparative no-regression evidence, not globally clean typing. Formatting
normalized mixed processor-call newlines only; assertions were not changed.

Fresh CPython3.13.15 wheel/sdist build and off-repo -I noneditable probe passed on
exact 7bfd0cb at 2026-11-06T00:43:49.472905Z. Probe
.superpowers/m2a-wheel-catalog-probe-7bfd0cb.json confirms 65 byte-identical Python
files/schema 2/FTS5; root reports empty FKcheck/integrity_check=ok. Artifact directory
.superpowers/m2a-final-dist-7bfd0cb; wheel SHA256
e2f9c3ff09a5d5b41fcec9c35c1fab149294e9901f276e042bd2ce0bd12b0827,
sdist499d9249202451e95dafb3718dd78f2004bc825659f5810659f401a5707e92ee.
No provider/hardware used. Both wheel-smoke outcomes await full 3 result.

Full 3 started alone at 2026-11-06T00:44:04.5472024Z, source 7bfd0cb, fresh wheel;
logs/XML .superpowers/m2a-integrated-final-3.*. Final result: **8 failed, 945 passed,
7 skipped in 364.23s**. Docs independently parsed XML: 960 cases/8 failures/0 errors/
7 skips/internal 362.276s. Both wheel CLI/GUI cases pass with no failure/skip nodes
(3.593s/2.747s). All eight traces show TypeError for additive before_metadata_commit
in two old test_capture_recovery publication wrappers, real-child publish and
simulated publish_at. The independent RED author owns the narrow optional-keyword
forwarding correction on codex/m2-capture-fixture; every assertion and real/simulated
crash cutpoint stays intact. Its 13-case focus (6 real/7 simulated), test-only commit,
root integration and corrected full 4 are pending; no product source fix diagnosed.
Current source/wheel/probe 7bfd remain exact if only tests change. Historical full2
at 8087e25 remains valid 940 passes/7skips/325.09s with both wheel smokes passing for
its own source; full1's five failures/920 passes/7 skips and prior diagnostics/timeout
remain preserved. No passing aggregate is claimed for the new source.
Evidence levels: implementation integrated with final focused GREEN; automatic
local synthetic focus/static/wheel evidence, failed full 3 and corrected full 4/hosted CI
pending; no new hardware/provider/clean-install certification. Task 3/group A/M2
remain unaccepted; version 0.2.0, 0.3.0 only at B/M2 closure. Two-hour acceptance
remains omitted by ruling 5. No new ruling or DoD change. R35 still requires A
merge plus required contract-doc PR merges before root creates project chats B/M3 A;
IDs/branches/worktrees/executors/real merged-A base stay pending, none created.
Prior future docs/fixtures remain untouched. Copilot received/responded on current
head, exact-head Windows CI, guarded admin merge and ledger remain required.

## 2026-11-06 — Current integrated full 4 green; final scoped docs verdict pending

Execution head ef70f78cfc9c8f2910e77b2b2818ce2e6b4d71ae integrates the TEST-ONLY
fixture correction c3a3ccbcf2928502cbf0d0881fdfee3d7f86d84a. It changes only two
publication-wrapper signatures/delegations in tests/test_capture_recovery.py to
accept/forward optional before_metadata_commit. All 114 assertions are identical
by author AST comparison; all real/simulated crash cutpoints and deadlines remain.
Docs independently read the committed 24-line diff and report and final focus log:
13 passed in 24.82s, covering six real kills/seven simulated crashes. File SHA256
5E542C069A811CC3F209B55815EC52B5D953ED5F977699AF18D7C86B9582E134. Report/log:
.superpowers/sdd/M2_CATALOG_PLAN/capture-fixture-forwarding-report.md and
capture-fixture-forwarding-focus.txt. No product source changed.

Product source tree remains 2b4036f7deb905efb0a8836f378d267a066564b4 from 7bfd0cb;
tests tree is now 30c88848b3cf42c9ce4443328e3c8e0100e2af44. Root verifies no
source/config/scripts/workflow diff. The existing fresh 65-file noneditable wheel
and probe of 7bfd0cb therefore remain exact; no redundant build or new artifact
hash is claimed. Full 3's eight failures/945 passes/seven skips remain preserved
above with their actual TypeError diagnosis, not relabeled as passed.

M2A-INTEGRATED-FULL4-ef70f78 ran alone from 2026-11-06T00:54:42.9822634Z with
QT_QPA_PLATFORM=offscreen, synthetic OPENAI_API_KEY and the same fresh noneditable
wheel environment. Root exit 0; actual result **953 passed, 7 skipped in 358.94s**.
Logs/XML .superpowers/m2a-integrated-final-4.txt/.xml. Docs independently read final
summary and parsed XML: 960 cases, zero failures/errors, seven symlink-privilege
skips, internal 358.621s. Both wheel smoke cases pass without failure/skip nodes:
CLI 3.018s, GUI 2.634s. This is a passing current integrated full on unchanged product
source plus the corrected test fixtures, not a new hardware/provider/installer
certification or hosted CI result. All historical failed suites, old full2's
940/7/325.09s and the unexplained unchanged child timeout retain their own scope.

R38 is implemented and locally automatically verified. The same independent
reviewer provisionally considers GR1–GR3/GD1/GD2 addressed and the narrow fixture
accepted; the final source/docs verdict awaits this committed documentation
supplement. GR1–GR3 formal status remains OPEN until that final independent
assessment. No second whole-source review, new ruling, DoD change, task/group/M2
acceptance or version bump. Version 0.2.0, two-hour omission under ruling 5 and final
1.0.0 acceptance requirements remain unchanged. PR/current-head Copilot received/
responded, exact-head Windows CI and guarded admin merge are pending. R35 still
requires A merge plus necessary contract-doc PR merges before project chats B/M3 A;
no chats or future-front execution have started, and future IDs/base remain pending.

## 2026-11-06 — Final independent technical verdict and routine ledger cleanup

Same scoped reviewer APPROVED source7bfd0cb, fixtureef70f78 and committed docs
8845877d75342d1c0f65f1a6674155c7439ab4ce in group-fix-review-report.md:
GR1/GR2/GR3/GD1/GD2 ADDRESSED, no remaining Critical/Important finding or new
source defect. Task3 is implemented, locally automatically verified and technically
accepted. Full4 evidence and every historical failed run remain unchanged.
GD3 Minor/P3's three stale current phrases (JSON source state, STATUS released
lease/completed full4, SPEC aggregate verification) are corrected in this normal
verdict update. No additional review/runtime is requested. Docs lease released.
Root may open the M2 A PR; Copilot received/responded, exact-head Windows CI,
guarded merge and ledger remain pending. Group A/M2 are not closed or certified;
version0.2.0 and R35 future-chat/contract-doc gates remain unchanged. No new ruling.

## 2026-11-06 — PR #21 Copilot correction wave 1 integrated; full5 passed

Copilot review5794475912 COMMENTED on published3394149 at01:21:05Z confirmed
C1 rollback (inline4488007346), C2 shared default catalog (4488007374) and C3
blocked completed replay warning (4488007407). R40 corrections are integrated in
c3cdf085c03718ca6a9d5db87ea0de28fa9652b0 from61fb8e2815887c19a1d6dc64572e5f469ec5cee9;
source b465d3198840f96d5d15a85eeb45e4312e4c4dec, tests293bc5038bae501dee07631704899e5f385e9025,
identical to worker. Three source files/three new independent tests, no other product
contract/version/schema/dependency/deadline change. Existing R41/R42 next-contract
planning rulings remain intact; neither is implemented by this correction.

Ordinary precommit Exception restores previous evidence before propagating;
BaseException/process termination unchanged, failed rollback retains backup/cause.
Default SessionProcessor shares the controller's actual MemoryService/custom root;
explicit processors retain contract. Rejected completed local replay becomes an
existing completed/can_retryFalse recovery warning before retry-audio probes;
metadata/marker/prior revision remain, no provider retry or new UI flow. Successful
replay/no-marker controls unchanged. Independent scoped verdict C1-C3 is pending.

Independent RED8 cases:5 failed/3 controls passed, sequential26.91s. GREEN8 passed
9.77s; affected15-file focus325 passed/1 symlink skip/141.36s. Ruff/format6 pass;
canonical mypy339/current79 errors/20 files/65 sources, no new diagnostics{}. Report
copilot-1-implementation-report.md and RED report/logs retained in local ledger.
Docs read exact raw bot comments, actual source diff/reports/probe; no runtime.

Fresh c3cdf08 wheel/sdist/probe at2026-11-06T01:47:42.753122Z:65 source files equal,
noneditable/schema2/FTS5; FK/integrity root-reported. Wheel SHA256
acbc092001eda20bcf980bf9d74d2c53c01698322641e9f8a9d313c34efec051; sdist
f7d80d8612aa6e2de13662324c16f670baa5022a2ad4dfcd976df1e6847b33ab.
Full5 started alone2026-11-06T01:47:43.289345Z, .superpowers/m2a-integrated-final-5.*:
**961 passed,7 symlink skips,0 failures in371.80s**, root exit0. Docs independently
read final log/parsed XML:968 cases/0 failures/0 errors/7 skips/internal371.710s.
Both wheel CLI3.747s/GUI2.978s pass with no failure/skip nodes. Current source-only
scoped inspection provisionally marks C1-C3 addressed/no new defects; final
technical verdict awaits committed docs supplement and actual full5 evidence. Prior independent approval/full4 and hosted Windows339 remain
historical. CI42142394863 on339 passed958/2wheel skips/250.79s; package42142395026
passed, proving339 only. Published PR head is still339, not localc3cdf08.

Root still must receive final current scoped acceptance, reply each inline, request
and receive Copilot on corrected head and get exact-head Windows CI before guarded
merge. No A/M2 close/certification/version change or future-chat creation. Two-hour
acceptance remains omitted by ruling5. R35 ordering and all historical failed
full1/full3/diagnostics/timeout plus full2/full4 scope remain unchanged.

Final Copilot-wave1 independent verdict: APPROVED, source/tests c3cdf08 plus
committed documentation f06b33b. C1/C2/C3 all ADDRESSED; no Critical/Important or
new actionable source defect. Reviewer directly verified full5 961/7/371.80s,
968 XML cases/zero failures/errors and both wheel smokes, with scope limits in
copilot-1-review-report.md. CD1 stale full5 heading/publication wording corrected
in this normal ledger update. Source/tests unchanged; no further runtime/review
requested. Root may publish, reply separately to inline4488007346/7374/7407 with
c3cdf08, and request/receive Copilot on the actual corrected head. Windows CI on
that head and guarded merge remain pending; no group/milestone closure or real
certification is inferred from this technical acceptance.

## 2026-11-06 — Copilot C4 integrated; exact-head complete Windows suite pending

Review5794829768 COMMENTED on published D23 at02:09:04Z resolves C1–C3 and adds
C4/4488285573. R43 was committed in3cf8726 before code. Source56af5253a8eec498c594bd6f1e1e456ad69f9453
integrates0a28842e9ac0a9f670a5e800077bf0fce8ae9e6d: src a17ec107c212d62e1984bb1cede094b579db5137,
tests c6a29f96af9ffc1e0c488ca73a430ef9a99b86b6, identical to worker. Only recovery.py
and one new independent test change. Marker-data ValueError is caught locally,
UnsafeSessionPath re-raised; fixed bounded non-echoing nonretryable warning retains
valid identity/status before WAV probes. No fallback proof, byte changes or provider.

Actual RED6 failures/2 controls in6.69s; GREEN8/8.18s; affected124/90.52s, zero
skips; Ruff/format2 clean, canonical mypyD23/current79/79 with no new diagnostics.
Fresh noneditable probe56af525 at2026-11-06T02:26:03.485121Z verifies65 equal source
files/schema2/FTS5, root-reported FK/integrity. Wheel SHA256
cdc99d7b0b76bdce2d8c606edb7adabcd335984be4fc0bce000690648a5bc37a; sdist
df30e3ffc9aa4e33d4b5b995f982d1c52097541ff1633b0035ee00074d686eb3.
Distribution file4 passed7.44s, XML4/0F0E0S/internal7.314s; installed wheel
CLI3.928/GUI2.748 pass, mutex0.005/path0.570 pass. Docs read source diff/reports/
probe and parsed actual XML; no runtime. Logs .superpowers/m2a-copilot-2-wheel-smokes.*.

Final independent scoped verdict pending committed docs. R43 requires the complete
Windows suite and package CI on the corrected exact head before merge; no duplicate
localfull6 and no waived full-suite gate. Full5CDF and CI_D23 are historical only:
D23 tests42147566672 passed966/2wheel skips/331.02s, package42147566683 passed.
Published PR21 stillD23; C4 unanswered/unpublished. C1–C3 replies4488243766/
4488243915/4488244052 resolved. Root must replyC4 with reachable correction SHA,
request/receive new Copilot, obtain exact-head complete-suite/package CI green,
then guarded merge. No group/M2 close, certification, version bump or future chats.
R39–R43 and all historical failures/timeouts retain their scope; no new ruling here.

C4 final independent verdict: APPROVED for source56af525 and committed six-doc
supplement58176ce; C4 ADDRESSED, no Critical/Important or new actionable source
defect. Reviewer checked actual focus reports, fresh65-file byte-equal wheel/probe
and distribution log/XML4 passed/7.44s, including both installed-wheel entries.
CD2 carried-forward CDF/current-pending note is marked historical in the normal
verdict update. No further local runtime/review requested. R43 still requires the
COMPLETE Windows suite/package on the actual corrected published head before
merge, plus newly received/responded Copilot. No localfull6, group/milestone close
or real certification is asserted by this scoped technical approval.

## 2026-11-06 — Copilot C5 integrated; current full-suite hosted gate pending

Review5795057762 COMMENTED on5b515e8844cc0d135461517dd55e907b65845c4d at
02:42:41Z confirms C4 resolved and adds C5/inline4488473222. C4 reply4488439090
and threadPRRT_kwDORsQx4M60We91 are resolved. Complete Windows CI5b42151367541
passes974/two wheel-environment skips/349.23s and package42151367491 passes;
these results prove5b only and remain historical for the new C5 correction.

R44 was committed4375247 before implementation. Rootc59ae331ebf124400ef3e476e5f0fcc029e83999
integratese8bd181f12fcfc25b8d66365689823552fa4e50b: src5c66169c0a93a8f0606a011bd9570bbdd9137903,
testsfb958d021eaf7d149c00a3a1be174e36fbef2a1c, identical to worker. Only recovery.py
and one new independently authored test change. Actual retry_parent LibraryError
or ambiguous-origin ValueError becomes a private signal; discovery keeps already
validated legacy metadata visible with its real identity/status/duration and a
fixed bounded nonretryable warning. It inspects no marker/error/audio/native
material for that warning and changes no source/catalog bytes. Actual owner-busy
and unsafe-path guards remain strict; stale materialize fails closed. No invented
UUID/parent/proof, provider call, repair, schema/dependency/version/deadline change.

Independent RED: initial five cases2 failed/3 controls passed10.98s, then one real
ambiguous-origin case1 failed2.95s on the same frozen5b baseline. Six distinct cases,
three RED/three PASS across two sequential runs (13.93s summed), not one six-case
run. The first five-case prefix stayed byte-identical. Final test SHA256
F7308104984F667919810A10F4575C01AE9B7F4D0B554A6D34468643CBD45696.
GREEN6 pass4.24s; affected17-file focus169 pass114.05s, zero failures/skips.
Ruff/format2 clean; canonical mypy5b/current79 errors/20 files/65 sources in both,
new touched messages{}. Reportscopilot-3-red-report.md/copilot-3-implementation-report.md
and raw logs remain in the local ledger. RED/GREEN files/runtime released.

Fresh wheel/sdist and noneditable probe c59ae33 at2026-11-06T03:08:14.816511Z:
65 Python files byte-equal, schema2/FTS5; FK/integrity passed as reported by root.
Wheel SHA2562c49dd9d896ce4b46d932bddc2f58cfb09380c7a22fe6d0d6a1f2e2f2c096de5;
sdist952b4706e290bc8e9134e61def5dadc09b341edf311ca136c46923ba8b7ee9a4.
Distribution file4 pass7.51s, XML4/0F0E0S/internal7.456s: installed CLI3.996s,
GUI2.757s, mutex0.007s and inherited-path0.559s. Docs read actual probe/stdout and
parsed XML, without runtime. Logs .superpowers/m2a-copilot-3-wheel-smokes.log/.xml.

Implementation and covered automatic local verification exist; final scoped
independent source/docs verdict is pending this committed supplement. R43/R44
require the COMPLETE exact-head Windows suite and package CI before merge; no
duplicate localfull7 or waiver. Published PR21 remains5b; C5 is not yet replied or
published, and new received/responded Copilot/current-head CI/guarded merge remain
pending. Full5CDF and every failed historical run retain their actual scope.
No group/M2 closure, real hardware/provider/clean-install certification, new ruling
or version bump. Two-hour acceptance is omitted by ruling5. R35 still requires A
merge plus required contract-doc PR merges before the two project chats; none
created, future IDs/branches/worktrees/responsibles remain pending.

C5 final scoped independent verdict: APPROVED for sourcec59ae33 and committed
six-document supplement3fdbd05; C5 ADDRESSED, no new actionable Critical,
Important or Minor finding. Reviewer directly inspected the source/test correction,
six committed docs, actual probe and distribution stdout/XML:65 byte-identical
installed files,4 passed/7.51s including both wheel entries, zero failures/errors/
skips. Report: `copilot-3-review-report.md`. No extra review/local full requested.
This routine verdict ledger changes no source/tests or acceptance definition.
Root must still publish/reply C5, receive new Copilot and complete exact-head
Windows suite/package success before guarded merge. No group/milestone closure
or real certification follows from technical approval alone.

## 2026-11-06 — M2 A closed and PR #21 merged with every gate

The catalog group is implemented and automatically verified, then merged; M2 as a
milestone remains OPEN pending group B/lifecycle/deletion and its closure criteria.
No new hardware/provider/clean-install certification or version bump:0.2.0 remains
current. Two-hour acceptance remains **omitida por ruling 5**, never passed.

The CLAUDE §7 procedure was completed in this exact order:

1. Non-draft PR [#21](https://github.com/srodriguez-lefebre/here/pull/21) targeted
   main with task/whole-group evidence and independent source/docs review in its
   description. Final scoped C5 approval covers sourcec59ae33 and committed docs
   3fdbd05 in `copilot-3-review-report.md`; earlier accepted findings remain scoped
   history, with no open technical finding.
2. Copilot requested on published head8a498e207812592ff84a0acc5ba4e9f392e02373 at
   2026-11-06T03:15:54Z and received as actual bot review5795325438 COMMENTED at
   03:20:34Z on that exact head: no open issues, C5 resolved, zero new inline
   observations. C1/C2/C3 replies4488243766/4488243915/4488244052; C4 reply4488439090;
   C5 reply4488660276 at03:15:37Z references reachable correctionc59ae33. All five
   threads are resolved; C5 threadPRRT_kwDORsQx4M60W6_- is included. Each substantive
   correction received a new review; no silence bypass was used. The bot's review
   text predated final test completion and did not replace the subsequent CI gate.
3. Exact-head Windows checks passed: complete test run
   [42155415514](https://github.com/srodriguez-lefebre/here/actions/runs/42155415514),
   **980 passed/2 skipped in307.46s**, completed03:21:08Z; package/installer run
   [42155415501](https://github.com/srodriguez-lefebre/here/actions/runs/42155415501),
   success completed03:18:48Z. Both head_sha values equal8a498e2. The two skips
   are installed-wheel acceptance cases without HERE_WHEEL_PYTHON, not passing
   cases; separate fresh exact-source distribution4/7.51s covers both installed
   entries. Copilot checkrun42155492338 also succeeded.
4. Root executed `gh pr merge 21 --merge --admin --match-head-commit
   8a498e207812592ff84a0acc5ba4e9f392e02373` only after those guards. GitHub merged
   at2026-11-06T03:22:06Z with merge commit
   **53712c5beedafc2d40e2abf10c13362a1c3c9f78**. Local Git verifies parents763a891
   and8a498e2; this is a merge commit preserving the group history, not a squash.
5. This DELIVERY entry records PR, exact reviewed head, bot review, per-observation
   dispositions, Windows runs and merge. No gate or acceptance definition changed.

Final sourcec59ae331ebf124400ef3e476e5f0fcc029e83999 and merged head have identical
src tree5c66169c0a93a8f0606a011bd9570bbdd9137903 and tests tree
fb958d021eaf7d149c00a3a1be174e36fbef2a1c. Named acceptance run:
M2A-CI-FINAL-8a498e2-20261106. Actual complete log retained in
`.superpowers/m2a-ci-42155415514.log`; docs read its actual980/2/307.46 summary and
skip reasons, final raw bot review/checks/thread JSON and the final independent
report. Docs executed no tests/build/mypy/GitHub actions and makes no new run claim.
R43/R44's complete CI route is now satisfied for A; no duplicate localfull7 ran.
Fresh wheel probe/distribution evidence remains sourcec59ae33, not a second CI run.
All historical failed suites/diagnostics/timeouts and source-specific passes remain.

Next action is a documentation maintenance PR from actual A merge53712c5 on
`codex/m2b-m3a-contracts`, same primary orchestration worktree. It adjusts the
existing M2B/M3A SPEC/PLAN contracts and M3 schedule before either implementation.
PR number/final review/merge still pending at this checkpoint. R35's two user-visible
project chats have not been created; their branch/worktree/executor/contract-base
IDs stay pending. Root remains sole orchestrator/contract/lease/integration/merge
owner; later M5B starts only after M2B and leaves claims integration until M4B.

## 2026-11-06 — Contract PR #22, BM25 continuation correction

Prepared-contract review PC1–PC3 finalized APPROVED on1fd228f, after A-closure
commitd8ebe3b. PR #22 opened non-draft with that independent result and evidence;
Copilot requested on exact1fd at03:36:31Z. Review5795454802 COMMENTED03:41:49Z
opens inline4488791517: corpus-wide BM25 changes can invalidate a saved-rank
cursor even after a nonmatching insert. Official SQLite BM25/data_version docs
confirm the rank inputs and connection-local version limitation.

R45 corrects the planned search contract: canonical complete-corpus generation,
one explicit read transaction plus coherent suppression, typed stale-cursor
rejection before old-rank use and restart through existing request handling.
Date/ID listing keeps its original insertion behavior. This changes no product
implementation, schema, retention, public flow or milestone acceptance clause.
Static checks retain all13/12 original SPEC clauses, prepared acceptance sections,
12/14 unchecked step lines and the full M3 numbered criteria section. The added
mutation tests are planned, not executed search evidence. Scoped independent
review of this committed correction remains pending.

Historical exact1fd CI passed980/two wheel skips/294.91s in42157559503 and package
42157559427 passed; root retained/read `.superpowers/contracts-ci-42157559503.log`.
These are not the corrected-head gates. Reply4488791517 with the reachable fix,
request/receive new Copilot and get complete exact-head Windows tests/package
green before guarded merge. No future chat or implementation has started.

## 2026-11-06 — Contract PR #22 merged; separate fronts initialized

The scoped independent R45 review APPROVED96f76db, inline4488791517 ADDRESSED.
Root published96f, replied4488851997 at03:53:51Z with that fixing commit and
resolved threadPRRT_kwDORsQx4M60XqLK. New Copilot requested03:54:08Z on exact
96f; review5795547739 COMMENTED03:57:00Z confirms no open issues and R45 resolved,
with zero new inline observations. Complete Windows tests42159238848 passed980/
two installed-wheel environment skips/304.66s; package42159238847 passed on96f.
Raw log `.superpowers/contracts-ci-42159238848.log`, checks/reviews/comments/threads
JSON retained alongside it. Previous1fd CI remains historical. No search code or
search-runtime acceptance is inferred from this documentation-only validation.

Root checked non-draft/main/exact head, received bot identity, every individual
reply/resolved thread and all exact-head checks before invoking
`gh pr merge 22 --merge --admin --match-head-commit 96f76dbbcb5ab4ba92866d9acf73c2673e140ee8`.
Actual merge:73d3cb384bd422de6899d8c7b761f3f3159ea3f1 at03:59:54Z. This completes
the required contract maintenance before dispatch; no milestone DoD was changed.

As explicitly requested, root created exactly two project chats through the app,
each with its own native worktree, from actual main73d3cb3:

- M2·B: ciclo de vida y borrado, chat01a1af04-b158-75a2-8933-98783880f3e8,
  branchcodex/m2-lifecycle, C:/Users/savar/.codex/worktrees/faab/here.
- M3·A: búsqueda y acciones, chat01a1af04-e779-7363-be61-c840651b0b18,
  branchcodex/m3-search, C:/Users/savar/.codex/worktrees/2710/here.

Root independently verified both actual branches, exact base SHA and clean Git
initialization, with accepted A53712c5 ancestry. Initialization reads only; no
future source/test/doc implementation, suite/build or provider invocation before
the full GO. STATUS records the IDs/bases/responsible executors and leases. Root
remains sole global contract/file/runtime/integration/merge orchestrator; each
chat executes its whole group with distinct RED/GREEN/review/docs subagent roles.
The first reserved host runtime window is B4 RED; M3 prepares without running
tests until a separate grant. Shared metadata/types/UI handoffs are explicit.

Real M1/0.2 upgrade baseline preparation is also retained under
`.superpowers/m5-upgrade-baseline-0.2.0/`: original run42102224714/artifact13450512413,
six supplied checksums and730 portable inventory entries verified. No installer
or binary was executed; this is fixture preparation, not upgrade certification.
Report `.superpowers/m5-upgrade-baseline-preparation-report.md` preserves actual
installer/portable hashes and provenance, separately from the artifact archive
digest. M2 remains open/version0.2.0; two hours remains omitted under ruling5.
