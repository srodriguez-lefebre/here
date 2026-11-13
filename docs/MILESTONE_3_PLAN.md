# Hito 3 — Memoria navegable

Estado: M3 A en integración y verificación. El candidato `3148869` reúne
búsqueda y Biblioteca sobre M2 B mergeado en `b4b3d1b`. La revisión
independiente pidió corregir la continuación de consultas largas y la composición
de recuperación con una raíz inyectada. La suite funcional integrada
`m3-final-functional-1` falló. El benchmark de producto medido en `ca80f49`
sigue por encima del límite y el diagnóstico aislado R66 también falló el
presupuesto. El contrato de M3 B se mergeó en PR #27 (`94e175f`). B espera el
PR revisado de A abierto y el despacho de root según CLAUDE §11. El diseño está en
[`development/PRODUCT_DESIGN.md`](development/PRODUCT_DESIGN.md), secciones 2, 3 y 5,
que es decisión tomada (ruling 3). Este plan no lo repite: fija alcance, orden,
criterios verificables y lo que queda fuera.

M2 A se mergeó en `53712c5` (PR #21). M2 B agregó edición, participantes,
reindexado y la barrera de borrado (PR #25). Las corridas históricas de M3
conservan su fuente y alcance en la [matriz de aceptación](development/ACCEPTANCE.md).

## Propósito

Que una persona encuentre una reunión y el momento exacto que busca sin saber cómo
está guardado nada. Listar, filtrar y buscar sobre la memoria de M2, abrir el detalle
por segmentos, y volver al audio original en el instante correcto cuando el tiempo
existe, o al texto cuando no. M3 se mide por si se encuentra lo que se busca; M2 se
mide por integridad.

## Qué deja M2 y qué falta

Contratos del catálogo M2 que consume M3:

- `library.sqlite3` con `meetings`, `transcript_revisions`, `segments` (con
  `start_ms`/`end_ms` nulos permitidos, `ordinal`, `timing_source`, `speaker_key`),
  `participants`, `speaker_assignments`, `audio_assets` con hash y relación explícita
  `revision_audio_assets` (schema 2). Las fechas de captura desconocidas siguen NULL.
- El índice FTS5 sobre las revisiones activas completas, creado en el grupo A de M2,
  sincronizado dentro de la transacción de publicación. M2 B añade su mantenimiento
  en edición, reindexado y borrado, sin un segundo escritor de índice en M3.
- `MemoryService.ingest_session`, `show` y `reconcile`, con la CLI de catálogo
  `here memory ingest|show|reconcile`.
- Revisiones históricas legibles e identificadas como tales.
- Todo operando sin `OPENAI_API_KEY`.

M2 B aporta participantes y asignaciones, `update_meeting` con `row_version`,
`delete_meeting`, `reindex` y la CLI de mantenimiento. M3 A usa la barrera
R42/R50 para consultar y los servicios de M2 para edición y mantenimiento. Root
serializa las ediciones compartidas y los cambios de migraciones o versión.

Alcance de M3:

- Un `SearchService` con consultas tipadas, ranking, filtros, paginación estable y
  snippets seguros.
- Una vista de biblioteca en la app: lista con filtros, detalle por segmentos,
  edición de título, estados vacíos.
- Un reproductor dentro de la app con seek al milisegundo del segmento elegido.
- Un target de cita común (`meeting_id`, `revision_id`, `segment_id`, tiempo opcional)
  que usen la lista, la búsqueda y, en M4, las citas.
- La CLI `here meetings …` con salida JSON estable y códigos de salida.

## Alcance

Entra en M3:

- `SearchService` y `MeetingQuery`/`SearchQuery`/`MeetingPage`/`SearchPage` según
  `PRODUCT_DESIGN.md` §3 y §5: FTS5 con `unicode61 remove_diacritics 2`, BM25 y
  después fecha e id, filtros aplicados antes del límite, cursor estable, consulta
  vacía que lista reuniones, texto literal normalizado y escapado antes de `MATCH`.
- Fallback literal documentado cuando FTS5 no está disponible, sin bloquear la
  biblioteca, y `reindex` para reconstruir el índice derivado.
- Pestaña Biblioteca en la ventana Qt: búsqueda arriba, filtros por fecha, estado y
  participante, listado con título editable, fecha, duración, estado y señal de audio;
  detalle con participantes, transcript por segmentos, errores recuperables y
  controles de audio. Consultas fuera del hilo Qt con debounce de 200 ms y secuencia
  de solicitud.
- Reproductor con `QMediaPlayer` y `QAudioOutput`: play/pause, slider, tiempo actual
  y total, volumen, velocidad moderada. Seek solo después de `loaded` y `seekable`.
  Reproducción deshabilitada mientras se graba.
- Navegación a evidencia: abrir un resultado selecciona y centra el segmento, ofrece
  "Reproducir desde mm:ss" con tiempo real o muestra "Sin marca de tiempo", da
  contexto anterior y siguiente, y marca si la revisión es histórica.
- CLI `here meetings list|show|search|rename|participant|retry|play|import-session|reindex|delete --yes`
  con `--json`, códigos 0/2/3/4/5 e IDs completos.
- Accesibilidad como aceptación: tabulación, nombres accesibles, contraste y selección
  por teclado.

No entra en M3:

- Preguntas, extracciones, respuestas ni citas generadas (M4). M3 solo deja el target
  de cita que M4 va a consumir.
- Exportación e importación portables, bandeja, inicio automático, pantalla de
  ajustes (M5).
- Cambios en captura, overlay, chunking o transcripción.
- Búsqueda semántica, embeddings o cualquier llamada a un modelo.
- Soporte de formatos de audio distintos del WAV que produce M1.

## Grupos de PR

Dos grupos, cada uno con su SPEC y PLAN en `development/`, revisables por separado.
Las cuatro tareas de `PRODUCT_DESIGN.md` §5 se reparten así:

| Grupo | Tareas del diseño | Documentos | Estado |
|---|---|---|---|
| A. Búsqueda y acciones | 1 `SearchService`, consultas y FTS sincronizado · 2 listado y detalle Qt con workers, filtros, edición y estados vacíos | `M3_SEARCH_SPEC.md`, `M3_SEARCH_PLAN.md` | Candidato en verificación, con correcciones de revisión pendientes |
| B. Biblioteca, reproductor y CLI | 3 reproductor y salto a evidencia con target común · 4 comandos CLI y JSON estables | `M3_LIBRARY_SPEC.md`, `M3_LIBRARY_PLAN.md` | PR de contratos #27 mergeado, despacho de fuente pendiente |

B depende del target de cita y los DTOs de búsqueda de A. CLAUDE §11 permite
arrancar B sobre la rama revisada de A con su PR abierto, una vez aceptados los
contratos y con despacho de root. B se rebasa después del merge de A. R39 fija
desempates y cursores con fechas NULL explícitas.

## Decisiones acordadas

- Servicios sin Qt ni Typer; la UI no ejecuta SQL ni lee archivos. El target de cita
  se resuelve siempre desde la memoria, nunca desde texto libre.
- Solo las revisiones activas y completas entran por defecto en búsqueda y listado;
  las parciales se muestran con badge y requieren selección explícita.
- El tiempo de un segmento se muestra y se usa para seek solo cuando
  `timing_source` no es `absent`. Nunca se reproduce desde cero aparentando
  precisión; el contexto de chunk puede mostrarse como aproximación explícita.
- Los snippets se generan como texto plano escapado; el transcript nunca se renderiza
  como HTML.
- Los resultados de una consulta anterior no reemplazan la actual; cerrar un detalle
  invalida sus callbacks.
- Durante grabación la reproducción queda deshabilitada, con el motivo visible, para
  evitar recaptura por loopback.
- Ninguna respuesta de M4 inicia reproducción automática; M3 lo deja escrito en el
  contrato del reproductor.
- Navegación de la ventana: pestañas Captura y Biblioteca. Ajustes es M5.
- Versión: 0.4.0 al cerrar M3 (ruling 9).
- Idioma: este plan y los criterios en español; SPEC y PLAN en inglés (ruling 6).

## Criterios verificables de terminado

El hito se considera terminado cuando se cumplen conjuntamente:

1. Una consulta vacía lista reuniones ordenadas por fecha e id, paginadas por cursor;
   avanzar y retroceder dos páginas devuelve los mismos elementos sin duplicados ni
   huecos, aunque se inserte una reunión entre llamadas.
2. Una búsqueda con tildes, mayúsculas, comillas, un token de dos letras y un
   término inexistente devuelve lo esperado en cada caso y nunca lanza un error de
   sintaxis `MATCH`; el texto del usuario nunca llega crudo a FTS5.
3. Los filtros de fecha, estado y participante se aplican antes del límite: con
   `limit=10` y 50 coincidencias de las cuales 5 cumplen el filtro, se devuelven las 5.
4. Ninguna búsqueda devuelve segmentos de una reunión en estado `deleting` ni de una
   revisión histórica, salvo que la consulta pida revisiones parciales o históricas
   de forma explícita.
5. Con el fixture sintético de 1.000 reuniones y 100.000 segmentos, la búsqueda
   caliente tiene p95 menor a 500 ms en el equipo documentado en `ACCEPTANCE.md`, y
   el hilo Qt no se bloquea más de 50 ms durante la consulta.
6. Si FTS5 no está disponible, la biblioteca abre, lista y filtra; la búsqueda usa el
   fallback literal con su límite visible, y `reindex` lo informa en lugar de fallar.
7. `reindex` reconstruye el índice desde las revisiones activas y el resultado de una
   búsqueda es idéntico antes y después.
8. En la vista Qt, escribir rápido dispara una sola consulta tras 200 ms de pausa, y
   una respuesta tardía de una consulta anterior no reemplaza el resultado actual.
9. Editar el título con un `row_version` viejo produce un conflicto visible y no
   sobrescribe; con el actual, actualiza y refresca la lista.
10. Los estados vacíos "Aún no hay reuniones" y "No hay coincidencias" aparecen en el
    caso correcto y ofrecen acciones distintas.
11. Abrir un resultado selecciona y centra el segmento, muestra contexto anterior y
    siguiente, y marca la revisión histórica cuando corresponde.
12. Un segmento con tiempo muestra "Reproducir desde mm:ss" y el reproductor pide el
    seek exacto a `start_ms`; un segmento sin tiempo muestra "Sin marca de tiempo" y
    no hay botón de reproducción desde cero.
13. El seek sobre un WAV real con marcas audibles cae dentro de 250 ms de la marca,
    verificado con reproducción real y registrado en `ACCEPTANCE.md`; un fake no
    acredita este criterio.
14. Cambiar de reunión detiene la reproducción anterior; un archivo faltante o un
    codec no soportado muestra un mensaje útil y la búsqueda sigue funcionando.
15. Con una grabación en curso, los controles de reproducción están deshabilitados y
    el motivo es visible; al terminar, vuelven a habilitarse.
16. Cada comando `here meetings …` devuelve el código de salida documentado en sus
    casos de error, y `--json` emite IDs completos y fechas ISO UTC; `play` sin
    multimedia disponible devuelve error explicativo y el target en JSON sin afirmar
    reproducción.
17. La biblioteca se opera entera por teclado: tabulación, nombres accesibles en lista,
    filtros y reproductor, y contraste verificado en ambos temas.
18. Toda la suite pasa en CI de Windows sin credencial y sin display; `README.md`,
    `ARCHITECTURE.md`, este plan, `ACCEPTANCE.md` y `WHATS_NEW.md` reflejan lo
    implementado, con las filas M3 de `ACCEPTANCE.md` nombrando la corrida que las
    cierra.

## Lo que queda pendiente de decidir

- Dónde vive `SearchService`. Default: `src/memory/search.py`, sobre el repositorio
  de M2, sin SQL en la UI.
- Rango de velocidad de reproducción. Default: 0,75×, 1×, 1,25× y 1,5×.
- Longitud de snippet y marcador de coincidencia. Default: alrededor de 160
  caracteres con `snippet()` de FTS5 y marcadores de texto plano, nunca HTML.
- Equipo de referencia para el benchmark. Default: la máquina de desarrollo del
  usuario, documentada en `ACCEPTANCE.md` con CPU y disco; los números del runner de
  CI se registran pero no son gate.
- Si las revisiones parciales entran en el listado con badge o quedan ocultas tras un
  toggle. Default: ocultas por defecto, toggle "Mostrar parciales" en filtros.
- Cómo abre `here meetings play` sin GUI en marcha. Default: una ventana mínima con el
  mismo adaptador de reproducción; sin multimedia disponible, error y target en JSON.

Estos defaults no requieren consulta; se registran como ruling si se adoptan.

## Fuera de alcance explícito

- Cualquier llamada a un modelo sobre el contenido (M4).
- Exportación `.here.zip`, importación de paquetes, ajustes en UI, bandeja (M5).
- Transcripción, captura, overlay o chunking.
- Búsqueda semántica o por embeddings.
- Edición del texto del transcript; solo título, notas y asociación de hablantes.
- Rediseño visual de la ventana principal más allá de agregar la pestaña Biblioteca.
