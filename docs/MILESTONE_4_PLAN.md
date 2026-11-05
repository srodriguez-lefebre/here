# Hito 4 — Inteligencia sobre la memoria

Estado: plan preparado el 2026-11-05. **No está en ejecución (ruling 15).** Arranca
solo con una instrucción explícita del usuario, después de que M3 esté cerrado. El
diseño detallado está en [`development/PRODUCT_DESIGN.md`](development/PRODUCT_DESIGN.md),
secciones 3, 6, 8 y 9, que es decisión tomada (ruling 3). Este plan no lo repite: fija
alcance, orden, criterios verificables y lo que queda fuera.

## Propósito

Convertir el historial de reuniones en conocimiento accionable sin ocultar la evidencia
de origen. El usuario pregunta sobre una o varias reuniones y recibe respuestas y
extracciones (decisiones, tareas, bloqueos) donde cada afirmación relevante lleva una
cita verificable que abre el fragmento original en el detalle y el reproductor de M3.
Hechos, síntesis e incertidumbre se distinguen a la vista; cuando no hay evidencia, el
sistema lo dice en vez de inventar.

## Qué dejan M2 y M3 y qué falta

Lo que M2 y M3 entregan y M4 consume tal cual (si `MILESTONE_3_PLAN.md` todavía no
existe, se asume exactamente lo que describe `PRODUCT_DESIGN.md` §5):

- Catálogo SQLite con `meetings`, `transcript_revisions` inmutables con `content_hash`,
  `segments` con tiempos nulos permitidos y `speaker_key` por solicitud, e índice FTS5
  sobre revisiones activas completas.
- `MemoryService.search(SearchQuery) -> SearchPage` con ranking BM25, filtros antes del
  límite y snippets como texto seguro.
- `Citation` (`meeting_id`, `revision_id`, `segment_id`, extracto opcional con offsets) y
  `MemoryService.resolve_citation(Citation) -> CitationTarget` con estado
  `current/historical/missing`, usado por lista, búsqueda y detalle.
- Detalle por segmento y reproductor con seek solo cuando hay tiempo real.
- Borrado íntegro con `pending_operations` y `deleted_origins`.
- Cliente OpenAI ya construido en `src/transcription/client.py` (`build_client`,
  `require_provider_key`) y un modelo de texto en uso, `CLEANUP_MODEL`.

Lo que falta y M4 construye:

- Un servicio de inteligencia con dos modos explícitos: evidencia local sin red y
  proveedor con salida estructurada.
- Recuperación léxica diversificada por reunión con presupuestos duros.
- Un verificador mecánico de citas: todo ID y toda cita que el proveedor devuelva se
  comprueba contra el contexto que se le envió.
- Persistencia de respuestas, claims y citas con invalidación por revisión y borrado.
- Vista "Preguntar" y vista de extracción en Qt, y los comandos `here ask` y
  `here extract` en la CLI.

## Alcance

Entra en M4:

- `src/intelligence/` con modelos, servicio, recuperación, extracción local, adaptador
  de proveedor y verificador. Sin Qt ni Typer.
- Modo local por defecto: pasajes relevantes con citas más candidatos de decisión,
  tarea y bloqueo por patrones conservadores en español e inglés. Etiquetas
  "Evidencia local" y "Candidato detectado". Abstención explícita.
- Modo proveedor: el usuario lo elige por consulta o preferencia; la UI informa qué se
  envía; se envían pregunta y fragmentos seleccionados, nunca audio, biblioteca
  completa ni nombres de archivo.
- `GroundedRequest` con bloques de evidencia inertes `E1`, `E2`, … separados de las
  instrucciones; salida estructurada con esquema; `store=False` si el endpoint lo
  admite; timeouts y tamaños acotados.
- Claims `fact`, `synthesis`, `uncertainty` y extracciones `decision`, `task`,
  `blocker` con `evidence_ids`, citas textuales y campos `owner_text` y `due_text`
  nulos cuando no están en el texto.
- Tablas `answers`, `claims`, `claim_citations` por migración, con FKs a reuniones y
  revisiones; clave de caché con hashes de contenido de todas las revisiones usadas.
- Vista "Preguntar": alcance visible, cobertura "Se consultaron N fragmentos de M
  reuniones", progreso, cancelación, tarjetas de respuesta y panel de evidencia que
  abre el detalle y el reproductor de M3.
- Vista de extracción por reunión y comandos CLI con salida JSON.

No entra en M4:

- Embeddings, base vectorial o búsqueda semántica. La recuperación es léxica sobre FTS5.
- Resúmenes automáticos al terminar una grabación o cualquier llamada al proveedor sin
  acción explícita del usuario.
- Identidad global de hablantes, normalización de nombres o fechas que el texto no
  expresa.
- Ajustes de proveedor en UI, bandeja, exportación, retención (M5).
- Garantía de ausencia de alucinación. Se garantiza que cada cita existe en el
  contexto enviado, no que la síntesis sea correcta.

## Grupos de PR

Dos grupos, cada uno con su SPEC y PLAN en `development/`, revisables por separado.
Valen las cinco tareas de `PRODUCT_DESIGN.md` §6.

| Grupo | Tareas del diseño | Documentos | Estado |
|---|---|---|---|
| A. Núcleo con evidencia | 1 modelos, citas y recuperación con presupuestos · 2 extracción local y abstención · 3 adaptador OpenAI, esquema y verificador | `M4_GROUNDED_SPEC.md`, `M4_GROUNDED_PLAN.md` | plan escrito, sin ejecutar |
| B. Persistencia, UI y CLI | 4 persistencia, invalidación y borrado concurrente · 5 UI, CLI y navegación a evidencia | `M4_PRODUCT_SPEC.md`, `M4_PRODUCT_PLAN.md` | plan escrito, sin ejecutar |

Orden obligatorio: A antes que B. B persiste y muestra lo que A produce; no se
construye UI de citas sobre un verificador que todavía no rechaza citas inválidas.

## Decisiones acordadas

- Modelo: `INTELLIGENCE_MODEL` se decide al implementar M4 y se registra como ruling
  con modelo y fecha (ruling 7). `CLEANUP_MODEL` (`gpt-4.1-mini`) es el único modelo
  de texto en uso hoy y nada obliga a reutilizarlo.
- Modo por defecto: local. El modo proveedor requiere credencial disponible y elección
  explícita; sin credencial, navegación y modo local siguen funcionando.
- Presupuestos por defecto: 6 segmentos por reunión y 24 en total por consulta,
  presupuesto duro de caracteres de entrada y de tokens de salida. Una consulta
  limitada se muestra como limitada, nunca como revisión exhaustiva.
- Alcance explícito siempre: IDs de reuniones seleccionadas o "toda la biblioteca"
  marcado a propósito. Una selección vacía no significa toda la biblioteca.
- El proveedor nunca aporta rutas, títulos, fechas ni timestamps para navegar; todo
  eso se resuelve desde la memoria con `resolve_citation`.
- Texto de transcript como dato inerte: ninguna instrucción dentro de un transcript
  cambia herramientas, red, política ni esquema.
- Borrado: una respuesta que cita una reunión borrada se elimina completa con sus
  claims; una respuesta en curso cuando se borra una de sus reuniones se descarta.
- Prueba con proveedor real: opt-in, con texto sintético, leyendo
  `HERE_PROVIDER_SMOKE_KEY` del `.env` del usuario sin preguntar (ruling 19);
  registra modelo, fecha y uso informado, y no guarda la clave.
- Versión: 0.5.0 al cerrar M4 (ruling 9).
- Idioma: este plan y los criterios en español; SPEC y PLAN en inglés (ruling 6).

## Criterios verificables de terminado

El hito se considera terminado cuando se cumplen conjuntamente:

1. Una pregunta sobre tres reuniones sintéticas cuyo corpus exige evidencia de al
   menos dos produce una respuesta con citas de al menos dos reuniones distintas, en
   modo local y en modo proveedor con transporte simulado.
2. Toda cita devuelta resuelve con `resolve_citation` a un segmento existente de la
   revisión activa en el momento de la consulta, con título y fecha reales y tiempo
   solo si el segmento lo tiene.
3. Una pregunta sin evidencia en las reuniones seleccionadas termina en "No encontré
   evidencia suficiente en las reuniones seleccionadas", sin claims inventados, en
   ambos modos.
4. La recuperación respeta los presupuestos: con un corpus de 50 reuniones y 2.000
   segmentos coincidentes, la consulta envía como máximo 6 segmentos por reunión y 24
   en total, y la cobertura mostrada coincide con lo enviado.
5. Una selección vacía no consulta nada y lo dice; "toda la biblioteca" solo se activa
   con la marca explícita.
6. El verificador rechaza, con prueba por caso: un `evidence_id` que no estaba en el
   contexto enviado, una cita vacía, una cita cuyo texto no es subcadena normalizada
   del segmento, un timestamp que el segmento no tiene, y campos fuera de los límites
   del esquema. Tras una reparación acotada que vuelve a fallar, el error es visible y
   se ofrece el modo local.
7. Un transcript sintético que contiene instrucciones (cambiar de modo, visitar un
   enlace, ejecutar un comando, ignorar reglas) no altera comandos, red, esquema ni
   política; la respuesta lo trata como texto citado.
8. La extracción local conserva el extracto literal: una negación ("no se aprobó")
   no se invierte, una hipótesis no se vuelve decisión, una tarea sin dueño queda con
   `owner_text` nulo y "el viernes" conserva el texto original.
9. En modo proveedor, un dueño o fecha inciertos quedan nulos o marcados inciertos;
   ninguna fecha absoluta aparece si el texto no la expresa.
10. Claims y extracciones persisten con FKs; borrar una reunión elimina toda respuesta
    que la cita, sus claims y sus citas, y ningún texto de esa reunión queda en
    `answers`, `claims`, FTS ni caché después de reiniciar.
11. Borrar una reunión mientras una consulta sobre ella está en curso descarta la
    respuesta completa y limpia su texto de la UI al notificarse el borrado.
12. Cancelar una consulta evita llamadas siguientes y publicación tardía; el estado
    resultante es "cancelado", nunca "completo".
13. Un reintento de transcripción que cambia la revisión activa marca como históricas
    las respuestas que la citaban; la UI lo muestra y ofrece regenerar explícitamente.
14. La clave de caché incluye pregunta, configuración y los `content_hash` de todas las
    revisiones usadas; una revisión nueva produce una clave distinta.
15. No se persisten payloads crudos del proveedor ni secretos; una prueba con una
    clave canario sintética y una frase canario en el prompt no las encuentra en la
    DB, en los logs ni en los artefactos.
16. La vista "Preguntar" muestra alcance, cobertura, progreso y cancelación; cada
    cita es un botón que abre el detalle y el reproductor de M3 en el segmento citado;
    los claims de síntesis llevan etiqueta visible y la cita textual queda a la vista.
17. `here ask` y `here extract` producen JSON estable con los mismos modos y los
    códigos de salida 0, 2, 3, 4 y 5; sin credencial, `--mode provider` sale con 5 y
    un mensaje que explica cómo configurarla.
18. Toda la suite pasa en CI de Windows sin credencial y sin display; existe una
    prueba opt-in con proveedor real y texto sintético, corrida con
    `HERE_PROVIDER_SMOKE_KEY` y documentada en `ACCEPTANCE.md` con modelo, fecha y
    uso informado, sin guardar la clave; `README.md`,
    `ARCHITECTURE.md`, este plan y `WHATS_NEW.md` reflejan lo implementado.

## Lo que queda pendiente de decidir

- Términos de la recuperación léxica: lista de stopwords español e inglés y tokens
  mínimos. Default: stopwords cortas fijas en el paquete, tokens de tres caracteres o
  más, OR de términos informativos, sin stemming.
- Presupuesto duro de entrada al proveedor. Default: 24.000 caracteres de evidencia
  y 2.000 tokens de salida, configurables por ajuste no secreto.
- Reparación acotada tras un rechazo del verificador. Default: una sola repetición con
  el error descrito; después, fallo visible y oferta de modo local.
- Mensaje de consentimiento del modo proveedor. Default: una vez por sesión de la
  aplicación, con el texto fijo de qué se envía; sin diálogo repetido por consulta.

Estos defaults no requieren consulta; se registran como ruling si se adoptan.

## Fuera de alcance explícito

- Embeddings, vectores, rerankers o cualquier dependencia nueva de búsqueda.
- Resúmenes o extracciones automáticas sin acción del usuario.
- Envío de audio, biblioteca completa o nombres de archivo al proveedor.
- Edición manual de claims o citas; corrección humana del resultado.
- Integraciones con calendario, correo o chat.
- Garantías de veracidad semántica; el verificador demuestra existencia de la cita.
