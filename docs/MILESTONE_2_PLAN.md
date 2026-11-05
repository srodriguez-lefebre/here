# Hito 2 — Memoria estructurada

Estado: **en ejecución desde el 2026-11-05**, por instrucción explícita del usuario
(ruling 20, reemplaza ruling 15). Grupo actual A; el hito no está cerrado.
El diseño detallado está en
[`development/PRODUCT_DESIGN.md`](development/PRODUCT_DESIGN.md), secciones 2, 3 y 4,
que es decisión tomada (ruling 3). Este plan no lo repite: fija alcance, orden, criterios
verificables y lo que queda fuera.

## Propósito

Convertir cada sesión grabada en una reunión con identidad propia, relaciones claras y
ciclo de vida consistente, de modo que M3 pueda buscar y reproducir, y M4 pueda citar,
sin depender de nombres de carpeta ni de convenciones frágiles. La memoria debe poder
reconstruirse desde los archivos, borrar una reunión sin dejar restos, y operar sin
credencial del proveedor.

## Qué deja M1 y qué falta

Lo que M1 ya entrega y M2 consume tal cual:

- `SessionMetadata.meeting_id`: UUID asignado al iniciar la captura, estable a través de
  recuperación, fallback y retry.
- `segments.json` versionado con texto fuente, tiempos nulos permitidos, hablante nulo
  permitido, `timing_reference=recording_seconds` y `speaker_identity_scope=request`.
- Escritura atómica por artefacto (`staged_artifact_path`, metadatos publicados al final)
  y validación de rutas contra traversal, symlinks y junctions.
- `RecoveryService.discover()` sobre la raíz de sesiones y `.captures`.
- Ajustes por usuario en `%LOCALAPPDATA%/here` con `HERE_DATA_DIR` como override.

Lo que falta y M2 construye:

- Un catálogo con autoridad sobre identidad, título, participantes, relaciones y
  revisiones. Tarea 3 implementa identidad e ingestión con verificación final pendiente.
- Revisiones inmutables de transcripción. Tarea 3 conserva snapshots propios;
  los archivos compatibles de la sesión pueden actualizarse durante retry.
- Un estado durable propio de la reunión, separado de los nueve estados del
  controlador y de los cuatro literales de `session.json`.
- Integración del catálogo con la vía de persistencia de aplicación. La tarea 2
  movió las escrituras directas de CLI a `SessionProcessor`; la tarea 3 agrega
  publicación y replay local, todavía sin aceptación final del grupo.
- Borrado íntegro: FTS, revisiones, archivos propios, backups administrados y, más
  adelante, respuestas de M4 que citen la reunión.

## Alcance

Entra en M2:

- Base SQLite local por usuario con migraciones versionadas, foreign keys, WAL y
  backup por API antes de migrar.
- Las tablas mínimas de `PRODUCT_DESIGN.md` §4, con `meetings`, `audio_assets`,
  `transcript_revisions`, `segments`, `participants`, `speaker_assignments`,
  `lifecycle_events`, `ingest_sources`, `pending_operations` y `deleted_origins`.
- Ingestión idempotente de sesiones v1 existentes y de toda sesión nueva producida
  por grabación, transcripción de archivo y retry, desde la capa de aplicación.
- Revisiones: cada retry crea una nueva, la activa al terminar y conserva la anterior
  como histórica.
- Participantes con alta manual y asociación explícita y reversible a etiquetas de
  hablante por alcance de solicitud.
- Borrado transaccional con operación persistida, reconciliación al arrancar y purga
  de todo lo derivado.
- Índice FTS5 preparado para M3, derivado y regenerable.
- CLI mínima para ingestar, reconciliar, listar por ID y borrar con `--yes`. La
  navegación de producto es M3.

No entra en M2:

- Listar, filtrar, buscar y reproducir como experiencia de usuario (M3).
- Preguntas, extracciones y citas generadas (M4).
- Exportación e importación portables, bandeja, inicio automático, ajustes en UI (M5).
- Cambios en la captura, el overlay o el pipeline de transcripción, salvo propagar
  `meeting_id` y segmentos donde todavía no llegan.
- Diarización global o identidad de personas inferida: un `speaker_0` de dos
  solicitudes distintas no es la misma persona salvo asociación manual.

## Grupos de PR

Dos grupos, cada uno con su SPEC y PLAN en `development/`, revisables por separado.
El diseño de `PRODUCT_DESIGN.md` §4 enumera cinco tareas; el ledger de M1 mencionaba
siete. Valen las cinco del diseño.

| Grupo | Tareas del diseño | Documentos | Estado |
|---|---|---|---|
| A. Catálogo inmutable | 1 modelos, migraciones y repositorio · 2 segmentos y raw text en todos los recorridos · 3 ingestión, identidad y revisiones | `M2_CATALOG_SPEC.md`, `M2_CATALOG_PLAN.md` | tareas 1/2 implementadas con review independiente aceptada y evidencia local; tarea 3 en RED/GREEN pendiente |
| B. Ciclo de vida y borrado | 4 participantes, edición con `row_version`, reconciliación · 5 borrado transaccional y FTS | `M2_LIFECYCLE_SPEC.md`, `M2_LIFECYCLE_PLAN.md` | plan escrito, sin ejecutar |

Orden obligatorio: A antes que B. Si al mergear A cambió la forma de
`pending_operations` o de los contratos del catálogo, el PLAN de B se ajusta en su
propio PR de documentación antes de empezar (ruling 14).

## Decisiones acordadas

- Identidad: `meetings.id` es el `meeting_id` de M1. `session_id` se conserva como
  `legacy_session_id`. Una sesión v1 sin `meeting_id` recibe UUID al importar y el
  mapping ruta normalizada + huella de origen evita duplicarla en importaciones
  repetidas.
- Raíz de datos: `%LOCALAPPDATA%/here` con `library.sqlite3`, `meetings/<uuid>/`,
  `.staging/`. `TRANSCRIPTIONS_DIR` sigue siendo fuente de importación explícita y
  salida compatible durante la migración; no se recorre todo el disco.
- Fuente de verdad: la DB manda sobre identidad, título, participantes, relaciones y
  revisiones. Audio y snapshots de transcripción son archivos inmutables dentro de la
  raíz administrada. JSON y Markdown legibles son artefactos de recuperación y
  exportación; una edición externa no modifica la DB en silencio.
- Estados durables: `recording`, `processing`, `completed`, `failed`, `cancelled`,
  `interrupted`, `deleting`. `paused` queda en eventos, no es estado de reunión.
- Publicación de una revisión: preparar en staging, validar hashes, registrar
  operación, renombrar a destino inmutable, transacción que inserta y activa, marcar
  operación completa. Un crash entre pasos deja una operación reconciliable.
- Sin credencial: todo M2 funciona sin `OPENAI_API_KEY`. Ninguna operación de
  catálogo llama al proveedor.
- Versión: 0.3.0 al cerrar M2 (ruling 9).
- Idioma: este plan y los criterios en español; SPEC y PLAN en inglés (ruling 6).

## Criterios verificables de terminado

El hito se considera terminado cuando se cumplen conjuntamente:

1. Una grabación nueva desde la GUI y desde la CLI termina con una fila en `meetings`
   cuyo `id` es el `meeting_id` del journal, una `audio_assets` con hash SHA-256
   verificado y una `transcript_revisions` activa con sus `segments`.
2. Una transcripción de archivo (`here trans`) y un retry pasan por la misma vía de
   persistencia de aplicación; no queda ninguna llamada directa a
   `write_session_artifacts` fuera de ella.
3. Un retry crea una revisión nueva con número secuencial, la activa solo después de
   que su escritura y transacción terminan, y la anterior sigue legible e identificada
   como histórica.
4. Importar la misma sesión v1 dos veces produce `inserted` la primera y `unchanged`
   la segunda; importar una sesión con el mismo UUID y contenido distinto produce un
   conflicto legible sin sobrescribir.
5. Dos reuniones distintas con audio idéntico siguen siendo dos reuniones.
6. Los segmentos con tiempo nulo conservan su ordinal y su texto; ningún tiempo se
   inventa, ningún `speaker_0` se cruza entre solicitudes sin asociación manual.
7. El test de las dos pausas: dos fragmentos con cinco minutos de pausa entre ellos
   abren el segundo en el punto correcto del WAV, sin cinco minutos de silencio.
8. Una migración de esquema corre en transacción, pasa `foreign_key_check` e
   `integrity_check`, y deja un backup previo administrado; un cliente viejo rechaza
   una DB de versión futura con un error claro.
9. Matar el proceso en cada frontera de publicación (antes de staging, después de
   renombrar, antes de commit, después de commit, antes de marcar completa) y
   reiniciar deja exactamente una reunión con una identidad y sin archivos huérfanos
   fuera de `.staging/`.
10. Un fallo del catálogo durante la grabación no toca el audio: el WAV y
    `session.json` quedan, la UI muestra "Audio guardado; biblioteca pendiente de
    recuperar", y la siguiente apertura reintenta solo tareas locales.
11. Borrar una reunión retira en la misma operación sus filas, su FTS, sus archivos
    propios bajo la raíz administrada, sus backups administrados y sus temporales; un
    reinicio después de un kill a mitad de borrado termina el borrado sin resucitarla.
12. Borrar una reunión importada no borra la carpeta externa de origen, y `deleted_origins`
    impide que la reconciliación la vuelva a importar sola.
13. Una ruta de journal, sesión o destino con `..`, absoluta, symlink o junction fuera
    de la raíz se rechaza sin leer, escribir ni borrar, y las entradas válidas vecinas
    siguen apareciendo.
14. Una DB ocupada por otra conexión devuelve un error recuperable acotado por
    `busy_timeout`; ninguna transacción espera al proveedor, copia audio ni calcula
    hashes adentro.
15. Toda la suite pasa en CI de Windows sin credencial y sin display; las pruebas usan
    DBs y raíces temporales y nunca el `.env` real ni grabaciones privadas.
16. `README.md`, `ARCHITECTURE.md`, este plan, `ACCEPTANCE.md` y `WHATS_NEW.md`
    reflejan lo implementado; las filas M2 de `ACCEPTANCE.md` nombran la corrida que
    las cierra.

### Evidencia parcial de ejecución — 2026-11-05

La tarea 1 de A aporta modelos inmutables, las doce tablas, migraciones numeradas,
backups por API de SQLite y repositorio transaccional. RED inicial de 27 fallos;
primera suite completa local: 768 pasan/9 skips en 174,94 s. Una review independiente
detectó lecturas bloqueadas sin error tipado; ocho regresiones con EXCLUSIVE real
fallaron antes de la corrección. Foco corregido: 34 pasan/1 skip de privilegio de
symlink en 4,70 s, con re-review independiente aceptada. Corrida final
`M2A-T1-FINAL-20261105`: 778 pasan/7 skips en 173,68 s; ambos smokes del wheel
no editable pasaron. Límites y comandos en `development/ACCEPTANCE.md`.

| Criterio | Estado de evidencia |
|---|---|
| 1–5. Productores, identidad y revisiones | Ronda 1 integrada, F1–F7 addressed por review; NB1 y cinco fallos de suite integrada bloquean aprobación, nueva suite/re-review/CI pendientes |
| 6–7. Segmentos y pausa | Tarea 2 verifica evidencia/seek sintético; focos de tarea 3 preservan segmentos en catálogo, con verificación final/review pendientes |
| 8. Migración, backup y esquema futuro | Verificación automática local de la base de catálogo; falta CI de Windows en el head del grupo |
| 14. DB ocupada y transacciones cortas | Verificación automática local de escrituras y lecturas bloqueadas; la publicación con archivos/proveedor aún depende de tarea 3 |
| 15. Suite en CI de Windows | Pendiente; una suite local no cierra este criterio |
| 9–10, 13. Crash, catálogo pendiente y rutas | Guards y F4–F6 addressed; replay postcommit NB1 y regresiones de recovery/legacy siguen pendientes, sin aceptación global |
| 11–12. Borrado sin resurrección | Grupo B pendiente; no lo cierra la limpieza de staging de tarea 3 |
| 16. Docs y evidencia de cierre | Checkpoints locales documentados; cierre de grupo/hito pendiente |

La tarea 2 (`037fc03`, integrada como `1cdaadb`) preserva `raw_transcript.txt`
separado del texto limpio y `segments.json` en vivo, offline, fallback, archivo y
retry, por la vía de aplicación. Tiempos inválidos siguen nulos; `timing_source`
es aditivo y nullable para documentos históricos. Review independiente de SPEC y
calidad PASS, sin hallazgos. Foco 310 pasan/1 skip en 24,60 s; captura/recuperación
con fases de kill conservadas 68 pasan en 25,64 s. Corrida
`M2A-T2-FINAL-20261105`: 783 pasan/8 skips en 160,71 s (seis symlinks y dos smokes
de wheel sin entorno opt-in). Tarea 3 completó RED conjunto de 39 fallos/18 pasan
en 7,33 s y fijó el candidato `6c422f0` en su worktree, sin integración ni aprobación.

Checkpoint de tarea 3: implementa ingestión, staging registrado, recibos
post-commit y snapshots de estado sin reescribir historia; retries confiables con
revisión activa esperada, incluso para sesiones v1 sin UUID en el manifiesto;
propiedad real de captura/procesamiento; replay solo local de ingestión o desenlace
fallido; FTS derivado, fechas UTC y aviso no fatal en la ventana. La segunda suite
estable del candidato pasó 883 tests/9 skips en 598,56 s. La primera dio seis
fallos de instrumentación de limpieza; se conservó y el autor acotó solo la inyección
sin cambiar assertions (diez pasan/16,76 s; Green 17,10 s). Detalles y XML en
`ACCEPTANCE.md`. La preservación desconocida y los guards de recibos/estado fueron
verificados en el candidato; el hallazgo previo de guards ausentes quedó superado.
La review inicial pidió siete correcciones. Ronda 1 Green `c28472b`, integrada
root `1f1bc88`, las marca ADDRESSED en re-review con 40 casos nuevos pasando
en 69,70 s. Schema 2 y recibo v2 de R31 están implementados sin alterar migración 1.
Sigue abierto NB1 Important/P2: controller publica de nuevo el retry ya comprometido
y el replay del marcador postcommit usa un parent obsoleto. La suite integrada root
dio cinco fallos/920 pasan/7 skips en 553,39 s; tres snapshots de recovery con nuevos
locks, stub sf.info incompleto y retry legacy solo-audio aún requieren diagnóstico.
Ambos smokes del wheel pasaron; siete skips de symlink. Wheel/sdist y probe aislado
no editable verifican 65 archivos Python iguales, schema2/FTS5/FK/integrity, sin
certificar hardware/proveedor ni cerrar el hito. Se requiere fix/re-review y suite
corregida antes de aceptar tarea/grupo.
La corrección del arnés sintético de captura
pasó 33 casos y review independiente, sin cambiar deadlines del producto; no
certifica hardware ni aprueba los contratos cuestionados.
No hay certificación nueva de hardware, proveedor o instalación limpia, ni cierre
del grupo A o de M2. La versión sigue en 0.2.0 hasta el cierre de hito previsto.

La instrucción R35 exige primero cierre completo y merge de A y sus PRs docs de
contratos necesarios. Solo entonces root crea chats separados M2 B y M3 A, con
ramas/worktrees propios y briefs que fijan base mergeada real, objetivo, SPEC/PLAN,
archivos, dependencias, límites y DoD. Chats/IDs/asignaciones todavía pendientes;
root conserva leases, contratos, integración y merges. M2 B puede avanzar junto
con búsqueda independiente de M3 A; las acciones que dependan de B esperan su
contrato aceptado. Tras B se permite adelantar M5 B, con claims pendientes hasta
M4 B. No se reinicia trabajo, cambian criterios ni se anuncia versión 0.3.0 ahora.

## Lo que queda pendiente de decidir

- Si el índice FTS5 se construye en el grupo A (junto con la revisión activa) o se
  difiere al B. Default: en A, como índice derivado sin consulta de producto, para que
  B pueda probar que el borrado lo purga.
- Ubicación exacta de la DB cuando `HERE_DATA_DIR` apunta a una carpeta de red o
  sincronizada. Default: permitirla con advertencia visible, sin recomendarla.
- Tamaño de lote y política de progreso para importar un `TRANSCRIPTIONS_DIR` grande.
  Default: una sesión por transacción, progreso por conteo, cancelable entre sesiones.

Estos defaults no requieren consulta; se registran como ruling si se adoptan.

## Fuera de alcance explícito

- Búsqueda de texto completo como experiencia de usuario, listado con filtros,
  reproductor y seek (M3).
- Cualquier llamada a un modelo de lenguaje sobre el contenido de la memoria (M4).
- Exportación `.here.zip`, importación de paquetes, retención automática, cifrado
  propio (M5).
- Rediseño de la ventana principal o del overlay.
- Soporte de macOS, Linux o web.
