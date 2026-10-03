# here — diseño de producto autónomo, hitos 2–5

Fecha: 2026-10-03. Documento de diseño y secuencia de implementación; no acredita funcionalidades implementadas ni certificación de hardware. Las decisiones se resuelven bajo la delegación del usuario. La implementación debe conservar los compromisos del hito 1 y actualizar los planes específicos con evidencia real al cerrar cada hito.

## 1. Resultado esperado y punto de partida

Una aplicación local para Windows 11 permite grabar, recuperar, encontrar y escuchar reuniones; consultar evidencia de una o varias reuniones; y administrar sus datos sin depender de conocer directorios. La UI es principal y la CLI ofrece las mismas operaciones de aplicación. El usuario controla explícitamente las solicitudes a proveedores; no hay sincronización, telemetría remota ni carga automática del historial.

La base observada tiene controlador asíncrono, estados inmutables, puente Qt, overlay y persistencia recuperable. Hay brechas específicas que condicionan el diseño:

- `TranscriptionResult` solo transporta `raw_text`, `final_text` y chunks. Los pipelines live/offline construyen `SegmentTimeline`, pero `finalize_transcription` no conserva esos segmentos en el resultado final. `SessionProcessor` escribe únicamente texto final. Debe corregirse el recorrido completo, no reconstruir tiempos desde texto.
- `SessionMetadata.session_id` y las carpetas se derivan de la fecha de finalización, con sufijo ante colisión. Se necesitan identidades de producto independientes del nombre de carpeta.
- `session.json` tiene estados pending/completed/failed/cancelled; estos no representan todos los estados transitorios del controlador. No se deben mezclar dos máquinas de estado distintas.
- `Settings.OPENAI_API_KEY` es obligatoria al cargar ajustes. Abrir la biblioteca y trabajar localmente debe funcionar sin credencial.
- La CLI tiene rutas que llaman directamente a `write_session_artifacts`. Escuchar únicamente `SESSION_PERSISTED` en la GUI dejaría sesiones sin indexar; además, el controlador aísla fallos de listeners. La persistencia de producto debe formar parte de la aplicación, con resultado verificable.
- Los archivos se escriben directamente y `retry` actualiza la misma carpeta. Introducir revisiones y escrituras atómicas evita invalidar citas anteriores y evita interpretar archivos incompletos como finales.

## 2. Arquitectura elegida y alternativas

Se elige SQLite local mediante `sqlite3` estándar, con FTS5 cuando esté disponible, más archivos de audio y artefactos versionados en disco. Es una biblioteca local por usuario, no un servidor. Qt permanece en un único proceso; trabajadores ejecutan operaciones de memoria e inteligencia fuera del hilo visual. La captura mantiene su controlador y sus contratos actuales.

| Alternativa | Ventaja | Coste y decisión |
|---|---|---|
| Archivos JSON y escaneo en cada consulta | Mínima dependencia | Relaciones, búsquedas, borrado e historial frágiles. Conservar archivos para recuperación, no usarlos como único catálogo. |
| SQLite + archivos, elegida | Transacciones, relaciones, búsquedas y distribución sencilla | Requiere reconciliar operaciones entre disco y DB. Resolver con journal de operaciones y pasos idempotentes. |
| Servidor/ORM pesado/vector DB | Escala multiusuario y semántica futura | Añade servicios, migraciones y privacidad sin necesidad actual. Fuera del alcance de estos hitos. |

La DB es autoridad de identidad, título, participantes, relaciones, revisiones e inteligencia. El audio y snapshots de transcripción son archivos canónicos inmutables dentro de una raíz administrada. Los JSON/Markdown legibles son artefactos de recuperación/exportación; ninguna edición externa modifica silenciosamente la DB. La importación explícita o reconciliación valida versión y hashes.

Paquetes propuestos:

```text
src/memory/models.py       DTOs inmutables, estados, citas, filtros
src/memory/database.py     conexiones, transacciones, migraciones
src/memory/repository.py   consultas, invariantes y FTS
src/memory/service.py      casos de uso de biblioteca
src/memory/artifacts.py    importación v1, revisión de archivos, reconciliación
src/memory/portability.py  paquete portable y validación
src/intelligence/models.py
src/intelligence/service.py
src/intelligence/providers.py
src/config/preferences.py ajustes no secretos y rutas
src/ui/library.py          lista, búsqueda y detalle
src/ui/playback.py         adaptador QMediaPlayer
src/ui/intelligence.py     preguntas, extracciones, citas
src/ui/settings.py        privacidad, operación e integraciones Windows
```

Agregar los nuevos paquetes al empaquetado explícito de `pyproject.toml`. Los servicios no importan Qt ni Typer; la UI no interpreta archivos, SQL ni respuestas crudas del proveedor. Inyección por constructor permite pruebas con SQLite temporal, proveedor falso y reproductor falso. No convertir cada función pura en una interfaz.

## 3. Contratos compartidos exactos

Los nombres pueden adecuarse a convenciones existentes, pero las responsabilidades y resultados son obligatorios:

```python
class MemoryService:
    def ingest_session(self, session_dir: Path) -> IngestResult: ...
    def list_meetings(self, query: MeetingQuery) -> MeetingPage: ...
    def get_meeting(self, meeting_id: str) -> MeetingDetail: ...
    def update_meeting(self, meeting_id: str, patch: MeetingPatch,
                       expected_version: int) -> MeetingDetail: ...
    def assign_speaker(self, meeting_id: str, speaker_key: str,
                       participant_id: str | None) -> MeetingDetail: ...
    def search(self, query: SearchQuery) -> SearchPage: ...
    def resolve_citation(self, citation: Citation) -> CitationTarget: ...
    def delete_meeting(self, meeting_id: str) -> DeletionResult: ...
    def export_meetings(self, ids: tuple[str, ...], destination: Path,
                        include_audio: bool = True) -> ExportResult: ...
    def import_archive(self, source: Path) -> ImportReport: ...
    def reconcile(self) -> RecoveryReport: ...

class IntelligenceService:
    def ask(self, request: QuestionRequest,
            cancellation: Event) -> Answer: ...
    def extract(self, request: ExtractionRequest,
                cancellation: Event) -> Extraction: ...

class IntelligenceProvider(Protocol):
    def generate(self, request: GroundedRequest,
                 cancellation: Event) -> ProviderResult: ...
```

`MeetingQuery`: texto de título opcional, UTC desde inclusivo/hasta exclusivo, estados, participante, limit entre 1 y 100, cursor estable por fecha/id. `MeetingPatch`: título y notas de usuario opcionales; nunca campos libres SQL. `SearchQuery`: texto, mismos filtros, límite y cursor. Una consulta vacía lista reuniones; no lanza MATCH vacío. `MeetingPage` y `SearchPage` contienen items y next_cursor; fechas ISO UTC se presentan en zona local. `IngestResult` distingue inserted/unchanged/revised/rejected, meeting_id y advertencias.

`Citation`: meeting_id, revision_id, segment_id y extracto exacto opcional con offsets de caracteres validados. `CitationTarget`: título/fecha, texto fuente, speaker visible, comienzo/fin opcionales en milisegundos, ruta de audio validada opcional y estado current/historical/missing. El proveedor nunca suministra rutas, títulos, fechas ni timestamps utilizados para navegación: se resuelven desde la memoria.

Resultados de error tipados: NotFound, Conflict, InvalidInput, UnsupportedSchema, IntegrityFailure, Busy, PermissionFailure, ProviderUnavailable. Mensajes visibles están redactados; no contienen credenciales ni contenido de la reunión por defecto. Los IDs de solicitudes en UI permiten descartar resultados antiguos tras cambiar filtro o selección.

## 4. Hito 2 — memoria estructurada

### Persistencia, identidad y migraciones

Raíz predeterminada: `%LOCALAPPDATA%/here`; biblioteca en `library.sqlite3`, reuniones administradas en `meetings/<uuid>/`, staging en `.staging/`, journal de borrado/operaciones en DB. Una raíz configurada por el usuario debe ser local y writable; no ofrecer carpetas de red o sincronizadas como opción recomendada. El directorio histórico `TRANSCRIPTIONS_DIR` se conserva como fuente de importación explícita y salida compatible durante migración, sin recorrer todo el disco.

Nueva reunión: UUID generado al iniciar captura y conservado en recuperación, fallback y retry. Compatibilidad: `legacy_session_id` conserva el ID v1 y un mapping de ruta normalizada + identidad de origen evita duplicados en importaciones repetidas. No deduplicar dos reuniones diferentes únicamente porque tengan audio idéntico. UUID importado + hash de manifest identifica duplicado idéntico; mismo UUID con contenido distinto produce conflicto legible, sin sobrescritura.

Tablas mínimas:

| Tabla | Datos e invariantes |
|---|---|
| schema_migrations | Versión, nombre y checksum de cada migración aplicada. |
| meetings | UUID PK, legacy ID opcional, título, notas, started_at/completed_at UTC opcionales, duration_ms, paused_ms, status, active_revision_id, row_version, created_at/updated_at. |
| audio_assets | UUID PK, meeting_id FK, ruta relativa administrada, tamaño/hash SHA-256, codec, sample rate, canales, duración, estado available/missing/corrupt. |
| transcript_revisions | UUID PK, meeting_id FK, número secuencial único por reunión, texto raw y presentación opcional, proveedor/modelo, created_at, origen, content_hash, complete/partial. |
| segments | UUID PK, revision_id FK, ordinal único por revisión, texto fuente, start_ms/end_ms nulos permitidos, timing_source, speaker_key opcional. |
| participants | UUID PK, display_name; nombre no implica identidad biométrica. |
| meeting_participants | meeting_id/participant_id FK, rol opcional, alta manual. |
| speaker_assignments | reunión, revisión o scope de proveedor, speaker_key, participant_id opcional; asociación explícita y reversible. |
| lifecycle_events | UUID, meeting_id, kind, occurred_at, recorded_duration_ms, payload validado sin secretos. |
| ingest_sources | Identidad externa y fingerprint → reunión/revisión; ingestión idempotente. |
| pending_operations | ID, kind, phase, meeting_id, rutas relativas y datos mínimos para completar/revertir pasos de filesystem. |
| deleted_origins | Fingerprint de origen eliminado y fecha, sin transcript, título ni nombres; impide resurrección en reconciliación. |

Hito 4 añade answers, claims y claim_citations con FKs. Un delete no puede dejar respuestas cacheadas que reproduzcan la reunión eliminada.

Conexiones por trabajador/operación; nunca compartir una conexión Qt con trabajadores. Activar foreign_keys por conexión, busy_timeout acotado, WAL en disco local y synchronous=FULL en escrituras durables. Transacciones cortas BEGIN IMMEDIATE para mutaciones; no esperar proveedor, copiar audio o calcular hashes dentro de la transacción. Bloqueo ocupado retorna un error recuperable. Escrituras incompatibles sobre una reunión se serializan mediante lock del servicio y estado DB; varias instancias verifican también el estado DB.

Migración: número de versión explícito, transacción y validación foreign_key_check/integrity_check en actualización. Backup mediante API de backup de SQLite, nunca copiar solo el .sqlite con WAL activo. Solo un migrador; clientes antiguos rechazan una DB de versión futura. El backup previo a migración se administra y cuenta entre los datos que hay que purgar al borrar; no mantener copias ocultas indefinidamente.

### Segmentos, hablantes y fuente verificable

Agregar `segments` a `TranscriptionResult` con default vacío compatible y propagarlo desde el proveedor a finalize, live, offline, fallback, retry y writer. Guardar `segments.json` versionado junto con el texto raw. La limpieza de texto es una vista derivada: las citas siempre apuntan al texto fuente, no se alinean artificialmente con texto reescrito.

Validar floats finitos antes de convertir a milisegundos; inicio no negativo, fin >= inicio cuando ambos existen. No ordenar segmentos sin tiempo por un tiempo inventado: conservar ordinal. Registrar timing_source=provider, imported_explicit o absent. Si un fragmento carece de tiempo, conservar texto y speaker disponibles; no perder todos los demás segmentos porque un chunk no trae timestamps. El contexto de chunk puede mostrarse como aproximación explícita, pero nunca como timestamp exacto del fragmento.

Los offsets se aplican una sola vez en el timeline de audio grabado. Pausas de pared no se suman al seek. El test crítico es dos fragmentos con una pausa de 5 minutos entre ellos: el segundo debe abrir el punto correcto del WAV sin cinco minutos de silencio. No asumir que `speaker_0` de solicitudes separadas representa a la misma persona: scope por chunk/proveedor salvo garantía expresa. Vincular speakers a participantes requiere edición explícita del usuario; no inferir personas globales por nombres o etiquetas similares.

Reintentar crea una nueva revisión y activa esa revisión solo al completarse la escritura/transacción. Revisiones anteriores permanecen inmutables para citas existentes y se señalan como históricas. No borrar texto previo al fallar un retry. Un resultado textual heredado sin segmentos se importa como uno o varios segmentos de párrafos con timing absent; no parsear heurísticamente horas escritas en el texto como tiempos de audio.

### Estados, publicación y recuperación

Estado durable: recording, processing, completed, failed, cancelled, interrupted, deleting. `paused` sigue en eventos/snapshot de captura, no es una transcripción completa. `recording` se crea con ID al empezar y se elimina en cancelación destructiva del hito 1. Si el proceso cae, al arrancar se marca interrupted con materiales disponibles y acciones explícitas de recuperación; no iniciar una solicitud de red solo por abrir la aplicación.

Guardar archivos pequeños usando temporal + flush/fsync + replace en el mismo volumen. Publicación de una revisión: preparar bajo staging → validar archivos/hashes → registrar operación prepared → renombrar revisión a destino inmutable → transacción insertar entidades/indexar/activar revisión → marcar operación complete. Crash antes de commit deja una operación reconciliable; crash después de commit deja archivos ya publicados. Solo las revisiones activas completas entran por defecto en búsqueda; parciales requieren badge y selección explícita.

La grabación no depende de indexar a tiempo: fallo de catálogo conserva el audio y muestra «Audio guardado; biblioteca pendiente de recuperar». La siguiente apertura reintenta únicamente tareas locales pendientes. Persistencia informa el error; un callback UI nunca es el único lugar donde se escribe memoria. Centralizar todas las rutas de `SessionProcessor` y `here trans`, manteniendo compatibilidad de importación v1.

### Borrado íntegro

La UI confirma una vez lo que se elimina; CLI requiere `--yes`. Servicio rechaza reunión con captura/procesamiento/import/export en curso. Detener reproducción y liberar handle antes de borrar. Marcar deleting y retirar de búsqueda inmediatamente; usar operación persistida para eliminar archivos propios y relaciones y luego finalizar. Si Windows deniega acceso, mostrar eliminación pendiente y reintentar localmente al iniciar; no declarar éxito.

Solo se eliminan rutas resueltas y comprobadas bajo la raíz administrada, sin seguir symlinks/junctions externos. Importar una carpeta histórica copia materiales al espacio administrado; borrar la reunión no borra el origen externo. La UI explica esta copia cuando corresponda. Retirar FTS, revisiones, citas, archivos temporales, exportaciones temporales y caches. Si una respuesta contiene una cita a una reunión borrada, eliminar la respuesta completa y sus claims; así no permanece una síntesis que revele contenido borrado. Limpiar backups administrados que contengan la reunión y checkpoint del WAL cuando sea posible. Purga lógica inmediata, limpieza física de archivos con estado verificable; no prometer borrado forense de SSD ni copias externas del usuario.

### Tareas y aceptación H2

1. Introducir modelos/migraciones/repositorio y validar nueva DB + DB v1 de prueba + rechazo de esquema futuro.
2. Preservar segmentos y raw text en todos los recorridos de transcripción y snapshots atómicos.
3. Integrar ingestión/identidad/revisiones con grabación, transcripción de archivo y retry.
4. Implementar participantes, asociación manual, edición con row_version y reconciliación.
5. Implementar borrado transaccional con recuperación y FTS preparado para H3.

Pruebas necesarias: duplicados/retry sin duplicar identidad, rollback de migración, IDs estables, FKs, timestamps ausentes/inválidos, speaker scope, pausa, cleanup sin alterar fuente, fallo entre cada fase de publicación/borrado, DB ocupada, path traversal y symlink, borrado con permiso denegado. Aceptación: el catálogo se reconstruye/recupera sin perder audio ni resucitar eliminados y puede operar sin API key.

## 5. Hito 3 — memoria navegable

### Interacción de escritorio

Mantener el panel de captura y overlay existentes; añadir Biblioteca y Ajustes mediante navegación simple o tabs. Biblioteca: búsqueda arriba, filtros fecha/estado/participante, listado con título editable, fecha, duración, estado y señal de audio disponible. Detalle muestra participantes, transcript original por segmentos, errores recuperables y controles de audio. Estados vacíos explican «Aún no hay reuniones» y «No hay coincidencias» con acciones distintas.

Búsqueda por FTS5 con unicode61/remove_diacritics para texto y título; parámetros SQL siempre vinculados y consulta literal normalizada/escapada. No pasar texto arbitrario como sintaxis MATCH. Ranking por BM25, después fecha y ID; topes y paginación evitan cargar todo el historial. Filtros se aplican antes del límite. Los snippets se generan/escapan como texto seguro; no renderizar HTML procedente del transcript. Si FTS5 falta, fallback literal documentado con límites; no bloquear toda la biblioteca. El índice es derivado y regenerable por `reindex`.

Cada resultado conserva meeting/revision/segment IDs y snippet. Abrirlo selecciona el segmento y centra el detalle; con tiempo conocido se ofrece «Reproducir desde 04:12». Sin tiempo se abre el texto y se indica «Sin marca de tiempo»; no reproducir desde cero aparentando precisión. Ofrecer contexto anterior/siguiente y mostrar si una revisión ya es histórica.

QMediaPlayer + QAudioOutput como adaptador de UI. Usar archivo local validado, esperar media loaded y seekable antes de `setPosition`; milliseconds de DB se corresponden con el WAV. Play/pause, slider, tiempo actual/total, volumen y velocidad moderada. Al cambiar reunión detener la anterior; errores de codec/archivo faltante muestran texto útil sin afectar búsqueda. No iniciar playback automáticamente al recibir una respuesta de IA. Durante grabación advertir o bloquear reproducción de reuniones para evitar recaptura involuntaria por loopback; decisión inicial: deshabilitar reproducción mientras se graba.

Consultas trabajan fuera del hilo Qt con debounce de 200 ms y secuencia de solicitud. Resultados de una consulta anterior no reemplazan la actual. Cerrar un detalle invalida sus callbacks. Tabulación, nombres accesibles, contraste y selección por teclado son parte de aceptación, no una fase cosmética final.

### CLI compartida

```text
here meetings list [--from ISO] [--to ISO] [--status ...] [--json]
here meetings show ID [--json]
here meetings search "consulta" [--meeting ID] [--json]
here meetings rename ID "Título"
here meetings participant ID "Nombre" [--speaker KEY]
here meetings retry ID
here meetings play ID [--segment SEGMENT_ID]
here meetings import-session PATH
here meetings reindex
here meetings delete ID --yes
```

`play` abre un pequeño reproductor Qt con el objetivo resuelto; si el entorno no tiene multimedia, retorna error explicativo y el target en `--json`, sin afirmar reproducción. CLI no implementa búsqueda ni persistencia paralelas. Códigos: 0 éxito, 2 entrada inválida, 3 no encontrado/conflicto, 4 fallo de integridad/IO, 5 proveedor. IDs completos en JSON; abreviaturas solo si resolución es inequívoca.

### Tareas y aceptación H3

1. SearchService/queries y FTS sincronizado transaccionalmente con revisión activa.
2. Listado/detalle Qt con workers, filtros, edición y estados vacíos.
3. Reproductor y salto a evidencia con target común desde lista, búsqueda y citas.
4. Comandos CLI y JSON estables.

Fixture sintético: al menos 1.000 reuniones/100.000 segmentos. Medir búsqueda caliente en equipo documentado; objetivo p95 <500 ms sin bloquear el event loop. Comprobar tildes, Unicode, comillas, tokens cortos, filtros antes de límite, orden/paginación estable y no resultados de reuniones eliminadas. Test Qt selecciona hit → detalle → seek solicitado exacto; prueba real de sonido confirma posición dentro de tolerancia de 250 ms en WAV, sin atribuir esa validación a un fake.

## 6. Hito 4 — inteligencia con citas entre reuniones

### Dos modos explícitos

Modo local predeterminado: recuperación lexical + extracción determinista de evidencia. Devuelve pasajes relevantes con citas y candidatos de decisiones/tareas/bloqueos identificados por patrones multilingües conservadores. No presenta una lista de citas como si fuese razonamiento generativo completo. Label: «Evidencia local» y «Candidato detectado». Si falta evidencia, responde «No encontré evidencia suficiente en las reuniones seleccionadas». No necesita red ni credencial.

Modo proveedor: adaptador real OpenAI con el SDK existente, modelo configurable dedicado (`INTELLIGENCE_MODEL`, valor inicial coherente con el modelo de texto ya usado en el proyecto). La presencia de API key habilita disponibilidad; el usuario elige explícitamente modo proveedor en una consulta o preferencia. La UI informa que se enviarán la pregunta y fragmentos seleccionados. La acción autorizada permite realizar la solicitud sin un segundo diálogo repetitivo. No se envía audio, biblioteca completa ni nombres de archivos por defecto. Configuración de proveedor se verifica en ejecución, sin fingir integración por pasar mocks.

Un error de proveedor muestra su estado y permite cambiar a evidencia local; no producir una respuesta de reglas silenciosamente etiquetada como IA. Credenciales ausentes o modelo no disponible no rompen navegación/captura local.

### Recuperación y evidencia

QuestionRequest contiene pregunta, IDs de reuniones seleccionadas o alcance explícito «toda la biblioteca», filtros, modo y presupuesto. Nunca interpretar una selección vacía accidental como enviar toda la biblioteca. Recuperar segmentos de revisiones activas con búsqueda literal OR de términos informativos españoles/ingleses; puntuar con FTS y contexto adyacente. Diversificar por reunión: máximo inicial 6 segmentos por reunión y 24 en total; presupuesto duro de texto y salida. La UI muestra cobertura «Se consultaron N fragmentos de M reuniones»; una consulta limitada no se presenta como revisión exhaustiva.

Para «resumir reunión» o extracción completa, seleccionar todos los segmentos y procesar en lotes acotados con progreso/cancelación; si se aplica un límite se muestra explícitamente. Deduplicar candidatos por contenido y citas, manteniendo todas las fuentes. No inferir contradicción por dos números distintos sin contexto: conflictos detectados son «posible cambio/conflicto» con ambas citas y fechas.

GroundedRequest envía bloques de datos inertes con IDs cortos E1/E2 y texto original, separado de instrucciones de sistema. Transcript puede contener instrucciones maliciosas: nunca invocar herramientas, seguir enlaces, ejecutar código ni cambiar la política a petición del texto. Usar structured output (Pydantic/JSON schema), `store=False` si el endpoint lo admite, timeout y salida acotados. No generar URLs/citas desde texto libre del modelo.

ProviderResult es una lista de claims `{kind, text, evidence_ids, quotes, uncertainty}` y extracciones `{kind, text, owner_text|null, due_text|null, evidence_ids}`. Tipos de claim: fact, synthesis, uncertainty. Tipos de extracción: decision, task, blocker. Toda afirmación relevante exige evidencia. Verificador comprueba ID dentro del contexto enviado, mismo snapshot/revisión, quote substring exacto normalizado mínimamente y campos acotados. Rechaza IDs desconocidos, citas vacías, tiempos inventados y respuestas sin evidencia; una reparación acotada opcional, luego fallo visible/evidencia local ofrecida.

La validación mecánica demuestra existencia de la cita, no implicación semántica. La UI etiqueta síntesis y conserva quote visible para revisión. No prometer garantía de ausencia absoluta de alucinación. Añadir fixture de negaciones («no se aprobó»), hipótesis, ironía, fechas contradictorias y tareas sin dueño. Para modo local se conserva extracto literal y no se normaliza dueño/fecha a un hecho no expresado. Para proveedor, nombre/fecha inciertos quedan null o marcados inciertos; «el viernes» conserva texto original si la normalización no está soportada de forma inequívoca por fecha de reunión.

Citas se renderizan como botones con título real, fecha real y timestamp real opcional. Abren el mismo detalle/reproductor de H3. Las citas sin timestamp siguen siendo verificables mediante texto; no se exige inventar tiempo para cumplir la UI.

### Persistencia y concurrencia

Persistir pregunta, modo/proveedor/modelo, versión de prompt/schema, selección, fingerprints de revisiones, cobertura, claims y citas verificadas. Uso/coste solo si el proveedor devuelve datos suficientes; nunca estimación monetaria presentada como factura. No persistir payloads de red crudos ni secretos.

Cache key incluye pregunta, configuración y content hashes de todas las revisiones usadas. Retry de transcripción hace histórico el resultado, no lo actualiza en silencio. Antes de mostrar/guardar, revalidar que reuniones no estén deleting/eliminadas; si cambiaron las revisiones, mostrar histórico o regenerar explícitamente. Si una reunión se borra durante una solicitud, descartar la respuesta completa y limpiar su texto en memoria/UI al notificarse borrado. Cancelar no puede retirar bytes ya enviados, pero evita siguientes llamadas y publicación tardía.

CLI: `here ask "pregunta" [--meeting ID ...] [--all] [--mode local|provider] [--json]`; `here extract ID [--mode ...]`. UI Preguntar tiene alcance visible, progreso, cancelación, answer cards y panel de evidencia; extraer ofrece decisiones/tareas/bloqueos con sus citas.

### Tareas y aceptación H4

1. Modelos/citas y recuperación lexical diversificada con budgets.
2. Extracción local conservadora y abstención honesta.
3. Adaptador OpenAI real y schema/verificador, cancelación/redacción.
4. Persistencia, invalidación y protección frente a borrado concurrente.
5. UI/CLI y navegación a evidencia.

Pruebas: pregunta entre tres reuniones produce citas de al menos dos cuando el corpus las requiere; no-hit abstiene; cada cita resuelve; speaker/timestamps ausentes se conservan ausentes; falso evidence_id falla cerrado; quote inexistente falla; prompt injection no tiene efecto en comandos/red; negación no se invierte en extracción local; selección limita payload; budgets limitan petición; error y cancelación no crean respuesta «completa»; borrado concurrente no filtra material. Test opt-in con proveedor real usa texto sintético, credencial disponible y autorización explícita de ejecución; registrar modelo/fecha y no guardar secreto. Sin prueba real, distinguir «adaptador implementado y verificado con transport simulado» de «proveedor certificado».

## 7. Hito 5 — producto cotidiano

### Ajustes, primera ejecución y privacidad

Separar ajustes locales de obtención de API key. Abrir biblioteca funciona sin key. Ajustes no secretos versionados y escritos atómicamente: data_dir, accent, close_to_tray=false, launch_at_login=false, intelligence_mode=local, intelligence_model, log level y política de retención manual. Mantener compatibilidad de `.env` existente; no copiar automáticamente credenciales a DB/QSettings, exports o diagnósticos. La primera ejecución explica dónde se guardan datos, qué operaciones usan red y cómo configurar credencial. Guardar nueva key en UI solo si se implementa almacén de credenciales Windows correctamente; en la primera entrega es aceptable mostrar configuración por entorno/.env y estado disponible/ausente sin revelar valor.

El directorio instalado no es una raíz de datos writable. Preferencias de ubicación solo cambian nuevos datos tras validación; migrar biblioteca existente exige caso de uso explícito que detiene trabajos, copia/verifica, cambia configuración y conserva rollback, no mover carpetas por efecto de un textbox.

Retención predeterminada indefinida con borrado manual. Mostrar uso de disco y opción «Eliminar reuniones seleccionadas». No programar purgas automáticas sin opt-in y sin reutilizar el flujo de borrado íntegro. No añadir cifrado casero; documentar protección del usuario Windows y compatibilidad con cifrado de disco del sistema sin prometer cifrado propio.

### Segundo plano e integración Windows

Conservar conducta H1 por defecto: cerrar en reposo sale; cerrar con captura/procesamiento oculta y overlay permite volver; al completar oculto sale tras indicador terminal. Opción «Mantener en bandeja» habilita residencia y un tray icon accesible; overlay se reserva para estado activo. El tray permite abrir, nueva grabación y salir. «Salir» con trabajo activo detiene/guarda antes de salir. Nunca usar servicio Windows ni arrancar micrófono solo por login.

Inicio al iniciar sesión es opt-in en HKCU/Startup del usuario actual con ruta del ejecutable correctamente citada, reversible y sin privilegios administrativos; no habilitar en entornos de desarrollo con rutas temporales. Single-instance de GUI mediante QLocalServer/QLockFile o equivalente: nueva apertura restaura ventana existente. Un lock de captura por usuario/raíz evita que CLI y GUI graben simultáneamente; comandos de consulta pueden coexistir con SQLite.

No añadir integraciones calendario/correos/videollamadas sin necesidad demostrada. La integración cotidiana suficiente es audio del sistema, archivos portables, apertura de carpetas/archivos elegidos, tray y startup opt-in. No introducir cuentas ni nube para declarar cerrado H5.

### Exportación/importación portable

Formato `.here.zip` versionado: manifest.json, meetings/<id>/ metadata y revisiones/segments, audio opcional, participantes/asignaciones y claims/citas incluidos solo cuando todas sus fuentes están en el paquete. Manifiesto incluye hashes/tamaños, versión, IDs y marca audio omitted cuando corresponde; nunca DB SQLite cruda, rutas absolutas, key, .env ni logs. Exportación texto/Markdown simple también disponible desde detalle, siempre con evidencia y limitaciones visibles.

Exportar toma snapshot consistente de metadatos y referencias; operaciones sobre reuniones seleccionadas se bloquean hasta terminar o se valida versión antes de publicar. Escribir archivo temporal y rename al completar. Paquete con audio requiere espacio disponible suficiente y progreso/cancelación. Sin audio, biblioteca importada conserva búsqueda/citas textuales y deshabilita playback.

Importación: inspeccionar zip sin extraer inicialmente; limitar número de archivos, tamaño comprimido/descomprimido y ratio; rechazar rutas absolutas, `..`, UNC, drive prefixes, alternate data streams, dispositivos Windows, duplicados normalizados, enlaces y manifests de versión futura. Validar hashes, tamaños y relaciones en staging; publicar mediante operaciones recuperables de H2. No aceptar rutas del paquete como ubicaciones fuera de raíz. Reimportar idéntico es no-op; conflicto UUID distinto se reporta sin sobrescribir. Nunca abrir contenido ejecutable del paquete.

### Diagnósticos y distribución

Logs rotativos locales con tope de 10 MiB por archivo y tres archivos, nivel INFO por defecto. Solo IDs de operación, etapas, tipos de error y métricas agregadas; no transcript, preguntas, prompts, nombres de participantes, headers o secretos. Sanitizar también exception messages y causa; los logs actuales que registran `str(exc)` requieren auditoría. Diagnóstico exportado se genera bajo acción explícita y enumera lo que incluye; no agrega sesiones ni .env automáticamente.

Entrega inicial Windows 11 x64: wheel/uv para desarrolladores y build de PyInstaller onedir con entrada windowed + instalador por usuario mediante Inno Setup o herramienta equivalente ya disponible. Preferir onedir para validar Qt multimedia y plugins frente a incertidumbre de extracción onefile. Build reproducible con lockfile, versiones de herramientas documentadas y artefacto checksum SHA-256. Instalador conserva datos al actualizar/desinstalar; borrar datos es una acción separada y explícita. No afirmar firma Authenticode si no existe certificado de publicación.

Actualización sostenible inicial: pantalla Acerca de con versión, enlace a canal oficial configurado desde metadata del proyecto y acción manual para abrir releases. Verificar que el enlace exista; si no hay canal publicado, entregar procedimiento de build/instalación y señalar la limitación. No inventar dominio ni prometer actualizador. Ejecutar instalador de versión nueva cierra de forma segura trabajos, conserva datos y migra con backup. Detección automática de versiones puede ser opt-in posteriormente; jamás descargar/ejecutar binarios silenciosamente. Una actualización real solo se declara validada tras ensayo N→N+1 y desinstalación en Windows limpio.

### Tareas y aceptación H5

1. Ajustes/rutas/primera ejecución sin credencial y diagnóstico redactado.
2. Bandeja opcional, single-instance, startup opt-in y salida segura.
3. Export/import validado con audio y sin audio; hooks de privacidad/borrado.
4. Build empaquetado/instalador y documentación de actualización real.
5. Pruebas Qt, inspección visual, accesibilidad y matriz Windows final.

Aceptación: sesión completa visible y buscable después de reinicio; biblioteca usable offline; export→import en raíz nueva mantiene IDs, participantes, tiempos, citas y hash de audio; paquete malicioso no escribe fuera de staging; borrar no deja transcript en DB/FTS/caches/backups administrados; logs no contienen secretos sintéticos ni frases canario; close/startup/tray respetan preferencias; app empaquetada reproduce WAV y graba ambas fuentes en Windows limpio.

## 8. Secuencia transversal y requisitos de cierre

Implementar slices que cierran recorridos: H2 persistencia y recuperación → H3 navegación y playback → H4 pregunta y cita → H5 portabilidad/operación/distribución. No desarrollar UI de citas sobre segmentos todavía inestables. No posponer el modelado de borrado hasta el final: H2 lo define y H4/H5 se integran a ese mismo contrato.

Cada hito tiene plan específico antes de su implementación y checklist final con evidencia: archivos/cambios, tests automatizados, pruebas manuales ejecutadas, límites pendientes y diferencias del diseño. `README.md`, `ARCHITECTURE.md`, planes y What's New se actualizan al implementar. No cambiar la definición de terminado para convertir un bloqueo de hardware en éxito de producto.

Usar fixtures sintéticos, DBs temporales y roots temporales en pruebas. Nunca leer `.env` real, transcripciones privadas ni colecciones personales para validar. Pruebas de red usan mock transport por defecto; solo smoke test opt-in envía fixtures sintéticos. Comprobar importabilidad/CLI sin API key ni Qt display; UI headless con pytest-qt y QT_QPA_PLATFORM=offscreen donde corresponda. Suite completa y checks del repositorio deben pasar; un test de widget aislado no verifica todo el recorrido.

## 9. Riesgos y certificación que requieren evidencia real

| Riesgo | Mitigación y evidencia requerida |
|---|---|
| Pérdida durante grabación larga/crash | Journal/checkpoints de captura del H1, recuperación y ensayo real de dos horas con marcadores conocidos. El catálogo no sustituye durabilidad de WAV. |
| Diarización inestable entre chunks | Scopes independientes, asociación manual y validación de modelo real. No afirmar identidad biométrica. |
| Precisión de tiempos | Propagación de segmentos + fixture pausa/overlap; ensayo de audio con marcas audibles y tolerancia registrada. |
| DB y filesystem no atómicos juntos | Journal idempotente y fault injection en cada frontera. |
| Residuos de contenido tras delete | FKs, FTS, respuestas cruzadas, WAL y backups administrados; límite explícito para exports externos/SSD. |
| Cita real pero síntesis incorrecta | Claims separados, quotes visibles, rechazo de citas inválidas y corpus de evaluación semántica. |
| FTS/Qt multimedia ausente en bundle | Probe en inicio, fallback de búsqueda, test del ejecutable empaquetado en máquina limpia. |
| Dependencia de API/modelo/coste | Adaptador configurable, budgets/timeouts, modo local, prueba opt-in y errores honestos. |
| GUI/CLI simultáneas | Transacciones, locks por operación y protección de captura; pruebas de proceso concurrente. |
| Actualizador/instalador sin canal real | Artefactos y procedimiento de release concretos; no inventar publicación, firma o auto-update. |

Matriz manual de liberación: Windows 11 x64 en al menos dos equipos; micrófono integrado y USB; loopback de altavoces y auriculares; fuentes a 44.1/48 kHz; sesión de dos horas, pausa/reanudar, silencio, desconexión, red caída, proveedor que falla, espacio insuficiente, suspensión/reanudación y cierre forzado. Medir bloques/frames capturados, duración audible, crecimiento de memoria y disco, cola de procesamiento y latencia de UI. Registrar tolerancias antes del ensayo; ausencia de errores unitarios no certifica este comportamiento.

El cierre final debe distinguir tres afirmaciones: (1) implementación terminada, (2) verificación automatizada y smoke tests ejecutados, (3) hardware, instalador y proveedor real certificados. Si no está disponible un equipo limpio, audio real o credencial autorizada, entregar artefactos y protocolos reproducibles y declarar pendiente esa certificación. No pedir al usuario decisiones arquitectónicas ya delegadas; sí ser preciso sobre evidencia no obtenida.

## 10. Referencias técnicas verificadas durante el diseño

- [SQLite FTS5](https://www.sqlite.org/fts5.html): índice derivado y consideraciones de eliminación. La opción secure-delete de FTS y la del núcleo son distintas; comprobar capacidades/versiones del runtime antes de asumir limpieza física y probar contenido residual.
- [QMediaPlayer para PySide6](https://doc.qt.io/qtforpython-6/PySide6/QtMultimedia/QMediaPlayer.html): reproducción, carga y seek mediante estado observable.
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs): esquema estructurado para claims; conformidad al esquema no equivale a veracidad semántica.

Las interfaces anteriores son decisiones de producto de este documento. Las referencias respaldan capacidades técnicas, no implican garantía de rendimiento, compatibilidad de cualquier modelo o certificación del código del repositorio.
