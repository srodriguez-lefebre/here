# Hito 1 — Captura confiable con una aplicación mínima de Windows

## Propósito

Entregar la primera forma instalable y cotidiana de `here`: una aplicación mínima de Windows capaz de capturar reuniones largas con micrófono y audio del sistema, transcribirlas y conservar resultados recuperables, sin que una falla parcial obligue a perder la conversación.

Este hito no busca una experiencia visual definitiva. Busca demostrar que el núcleo de captura y transcripción puede operar de manera confiable detrás de dos interfaces: una aplicación de Windows como interfaz principal y la CLI como interfaz secundaria. Ambas deben compartir los mismos casos de uso; la UI no ejecutará comandos de la CLI como subprocesos.

## Alcance del hito

Al cerrar el hito, una persona debe poder instalar `here`, encontrarlo en Inicio de Windows, abrir una ventana simple, iniciar y detener una captura, entender el estado del trabajo y obtener una sesión recuperable. La captura combinada de micrófono y audio del sistema es el recorrido principal; los modos de una sola fuente pueden seguir disponibles cuando sean útiles.

Durante una grabación activa, la aplicación muestra un pequeño overlay flotante que funciona como recordatorio inequívoco de que `here` está grabando. Cerrar la ventana principal mientras hay grabación o procesamiento la oculta, pero no termina `here` ni el trabajo activo. Un clic principal en el logo restaura la ventana; durante la captura, el menú de clic derecho ofrece tres acciones diferenciadas y con icono propio: detener y guardar/finalizar, pausar o reanudar según el estado, y cancelar.

El hito debe entregar los estados, eventos, telemetría de audio y acciones estables necesarios para alimentar ese overlay, sin acoplar el motor a su representación. El overlay funcional, la restauración de la ventana, la detención con guardado, la pausa/reanudación y la cancelación son requisitos del hito. Pausar deja de capturar audio hasta reanudar: no inserta silencio ni aparenta continuidad, y registra eventos de pausa/reanudación con tiempo de pared y duración total pausada.

El lenguaje visual, los morphs, las animaciones, los tiempos y su pulido se desarrollan en paralelo según [el plan del logo vivo](LIVE_LOGO_PLAN.md). Separar los planes evita que la confiabilidad del motor dependa de terminar el diseño visual, sin relegar esta iniciativa a un hito posterior.

La aplicación se mantendrá inicialmente como un solo proceso por usuario: ventana principal, overlay, coordinador y trabajadores en segundo plano. Solo permanece activa sin ventana principal cuando existe grabación o procesamiento en curso; en reposo, cerrar la ventana termina por completo la aplicación. No se recomienda un servicio tradicional de Windows para capturar audio ni hace falta un proceso separado. Una bandeja técnica podría evaluarse como detalle secundario de integración si fuera necesaria, pero no es la experiencia solicitada ni el recordatorio de grabación.

La dirección prevista para la primera interfaz es PySide6 con Qt Widgets, dentro del repositorio actual y sobre el núcleo Python existente. Esa elección debe confirmarse al comenzar la fase de UI, pero el plan no depende de detalles visuales ni de una arquitectura multiproceso prematura.

## Lectura del estado actual

El inventario siguiente registra el diagnóstico inicial que motivó este plan. La
ejecución autónoma iniciada el 2026-10-03 mantiene el estado actualizado en
[`development/DELIVERY.md`](development/DELIVERY.md) y su evidencia en
[`development/ACCEPTANCE.md`](development/ACCEPTANCE.md). En esa fecha ya existen
la capa de aplicación compartida, captura controlable, UI PySide6 y overlay, y la
línea base completa pasa 163 pruebas. El hito sigue abierto por robustez, recuperación
tras reinicio, distribución y aceptación real; las brechas iniciales resueltas no
deben reinterpretarse como trabajo pendiente ni como certificación del producto.

El repositorio ya contiene buena parte del motor del hito. Sin embargo, “existe código y tiene pruebas unitarias” no equivale todavía a “está validado como producto Windows”. El inventario distingue esas dos condiciones.

### Ya construido y comprobado en aislamiento

| Capacidad | Evidencia actual | Estado dentro del hito |
|---|---|---|
| Captura de micrófono, audio del sistema y ambas fuentes en Windows | `recording/windows.py` implementa entrada y WASAPI loopback; `recording/service.py` hace el despacho por plataforma; `test_recording_service.py` verifica el despacho | Implementada; el acceso a hardware real aún requiere validación |
| Escritura progresiva a WAV temporal | Los adaptadores de grabación escriben bloques con `soundfile` y devuelven `RecordingSession` con información de fuente | Funcional en código; falta someterlo a sesiones reales largas y fallos del dispositivo |
| Procesamiento en vivo por bloques | `LiveTranscriptionController` separa captura, generación de chunks y transcripción en trabajadores de fondo | Cubierto por pruebas de orden, dos fuentes y propagación de fallos |
| Cortes próximos a silencios y overlap | `audio/silence_boundaries.py`, `audio/chunking.py` y el controlador live seleccionan cortes y conservan solapamiento | Cubierto por pruebas sintéticas; falta validar calidad con conversaciones reales |
| Normalización, remuestreo y mezcla | `audio/mix.py` materializa una mezcla mono normalizada y controla clipping | Cubierto por pruebas unitarias de mezcla, tamaño y materialización |
| Transcripción live y fallback offline | La orquestación actual intenta completar el pipeline live y, si falla, procesa el audio recuperable offline | Cubierto por pruebas de la CLI y del servicio de transcripción |
| Merge textual y por timestamps | `audio/text_merge.py` deduplica overlap; `transcription/segments.py` desplaza y combina segmentos temporales | Cubierto por pruebas de texto, timestamps, speakers y overlap |
| Diarización cuando el modelo la ofrece | El cliente solicita y transforma salida diarizada según el modelo, preservando speaker y tiempos | Implementada y probada con respuestas simuladas; requiere validación contra el proveedor real |
| Sesiones recuperables | `output/session_writer.py` produce audio, texto, Markdown, `session.json`, `chunks.json` y, ante fallos, `errors.json` | Cubierto por pruebas de sesiones completas, fallidas y reintento desde `audio.wav` |
| Diagnósticos de dispositivos y señal | `recording/diagnostics.py` enumera las fuentes Windows usadas y mide peak/RMS sin necesitar transcripción | Cubierto con adaptadores simulados; falta verificar variedad de hardware real |
| Suite automatizada amplia | Hay pruebas para captura, pipeline live, mezcla, chunking, segmentos, persistencia, cliente y CLI | La mayor parte ejecuta; la línea base completa no está verde |

### Funcionalidad que todavía necesita prueba real

Antes de declararla confiable hay que validar, en Windows y con dispositivos reales:

- captura simultánea de micrófono y loopback con distintas frecuencias y cantidades de canales;
- continuidad, uso de memoria, uso de disco y calidad del resultado durante reuniones largas;
- comportamiento frente a silencio prolongado, pausas, clipping, dispositivos sin señal y cambios o desconexiones de dispositivo;
- recuperación ante cierre inesperado, error de proveedor, pérdida de red y fallo durante el procesamiento live;
- calidad de cortes, overlap y merge sobre habla natural, incluyendo el riesgo de pérdida o duplicación;
- timestamps, etiquetas de hablante y variantes de respuesta de los modelos reales compatibles;
- posibilidad de reanudar el procesamiento desde el audio conservado sin repetir la reunión;
- claridad de diagnósticos y mensajes para configuraciones comunes de Windows.

Estas validaciones no deberían mezclarse con una promesa absoluta de “sin pérdida”. El plan de validación debe definir casos, señales observables y tolerancias aceptables antes de dar por cerrado el hito.

### Línea base de pruebas observada

Al elaborar este plan, `uv run pytest -q` no completa la colección: `tests/test_transcription_client.py` importa `httpx`, pero el entorno resuelto no lo ofrece. Al excluir únicamente ese archivo, el resultado es `75 passed`. La primera fase debe corregir la declaración o resolución de dependencias y recuperar una ejecución completa, reproducible y verde antes de una refactorización amplia.

## Brechas para liberar una aplicación Windows confiable

1. **Orquestación acoplada a la CLI.** `cli.py` crea el controlador live, coordina captura, fallback, persistencia y traducción de errores. Esos casos de uso no son reutilizables limpiamente desde otra interfaz.
2. **Ciclo de captura ligado a la consola.** Los adaptadores exponen funciones “until Enter” y llaman a `input()`. No existe todavía un contrato programático para iniciar, pausar, reanudar, detener, cancelar y observar una grabación.
3. **Ausencia de estados, eventos y telemetría de aplicación.** El logging informa actividad, pero no hay un modelo estable de estados, progreso, resultado, error y nivel de audio para la ventana y el overlay de grabación.
4. **Ausencia de UI, overlay y entrada de aplicación.** No hay dependencia, paquete, punto de entrada ni pruebas para PySide6/Qt Widgets o la ventana flotante.
5. **Selección y recuperación de dispositivos limitada.** Se usan los dispositivos predeterminados. Falta decidir si la primera versión permite elegirlos, cómo muestra los elegidos y qué hace ante cambios o pérdida de señal.
6. **Ciclo de vida de UI incompleto.** El aborto actual está orientado a limpiar trabajadores internos. Falta implementar el cierre que oculta mientras hay trabajo activo, la continuidad con overlay, la reapertura y el cierre total cuando la aplicación está en reposo.
7. **Robustez sin validar en duración y fallos reales.** Hay buena cobertura aislada, pero no una matriz de pruebas de varias horas, hardware diverso, suspensión, desconexión, falta de espacio, red inestable o fallos del proveedor.
8. **Distribución Windows inexistente.** El proyecto instala comandos Python, pero no una aplicación con instalador, acceso en Inicio, recursos, configuración y desinstalación comprobables.
9. **Configuración y privacidad de la primera ejecución.** Debe definirse cómo se proporciona la credencial, dónde se guardan las sesiones y cuál es la política de conservación del audio, sin exponer secretos ni sorprender al usuario.

## Arquitectura objetivo del hito

La meta no es reescribir el núcleo, sino introducir fronteras que permitan usarlo desde CLI y UI sin duplicar comportamiento.

### Núcleo y dominio

Contiene conceptos y reglas independientes de interfaz: sesión de captura, fuentes, segmentos, chunks, estados, errores, timestamps y resultados. Las funciones existentes de mezcla, chunking, merge y metadatos pertenecen aquí o a servicios de dominio cercanos. Esta capa no debe importar Typer, PySide6 ni detalles de widgets.

### Capa de aplicación y casos de uso

Coordina los recorridos completos: iniciar, pausar, reanudar y detener una grabación; finalizar el pipeline live; ejecutar fallback; persistir una sesión; reintentar una sesión recuperable; y cancelar de forma controlada. La pausa deja de capturar y registra el hueco mediante eventos con tiempo de pared y duración acumulada, mientras audio y transcript mantienen un timeline relativo solo al material grabado. Cancelar durante grabación, después de una confirmación explícita, elimina definitivamente el material y sus temporales sin crear sesión. Cancelar durante procesamiento, también con confirmación, detiene ese trabajo sin eliminar el audio: deja una sesión persistida, identificable como no completada y apta para reintento. La capa debe exponer operaciones y eventos estables para que cualquier interfaz observe estados y resultados sin interpretar logs.

Esta capa será dueña del ciclo de vida de una ejecución, pero dependerá de contratos para captura, transcripción y persistencia. También publicará estado y telemetría de audio acotada para las interfaces, sin exponer bloques completos ni obligarlas a interpretar logs. La lógica que hoy vive en funciones privadas de `cli.py` debe migrar gradualmente aquí y quedar cubierta por pruebas antes de adelgazar la CLI.

### Puertos y adaptadores

- **Captura:** contrato programático para abrir fuentes, entregar bloques, emitir mediciones de nivel, pausar, reanudar y detener o cancelar; el adaptador Windows seguirá usando WASAPI/PyAudioWPatch. La telemetría se deriva del mismo flujo capturado, sin abrir otra captura para el overlay, y se interrumpe visualmente durante la pausa.
- **Transcripción:** adaptador al proveedor actual detrás de un contrato que distinga progreso, error recuperable, error final y capacidades como timestamps o diarización.
- **Persistencia:** adaptador que materializa audio y escribe artefactos recuperables. Debe ampliar los metadatos para registrar eventos de pausa/reanudación, sus timestamps de pared y la duración total pausada, sin insertar silencio en el audio. También debe distinguir una cancelación destructiva de grabación —sin sesión— de una cancelación recuperable de procesamiento.
- **Configuración y diagnóstico:** adaptadores para credenciales, rutas, enumeración de dispositivos y pruebas de señal.

La inversión de dependencias debe ser proporcional: contratos explícitos donde exista una frontera externa o una necesidad real de prueba, sin convertir cada función pura en una interfaz.

### CLI

Typer queda como adaptador de entrada secundario. Traduce argumentos y salida de consola, invoca los mismos casos de uso y transforma su resultado en códigos de salida. No debe conservar una versión paralela de la orquestación.

### UI PySide6 e integración del overlay

La ventana Qt será la interfaz principal. Durante una captura, una segunda ventana mínima, transparente, sin marco, siempre visible y arrastrable mostrará solo el logo. El clic principal restaurará la ventana. Durante grabación, el menú de clic derecho ofrecerá detener y guardar/finalizar, pausar o reanudar, y cancelar, con iconos que distingan cada acción; durante procesamiento ofrecerá “Cancelar procesamiento”. El hilo de UI solo administrará widgets, visuales y señales; la captura, transcripción y persistencia correrán en trabajadores de fondo dentro de la misma aplicación.

La ventana y el overlay invocarán la capa de aplicación directamente y representarán sus eventos, sin lanzar la CLI ni analizar logs para inferir progreso. El núcleo aporta estado, telemetría reducida durante captura y comandos; el plan compañero define cómo se representan visualmente grabación, procesamiento, éxito y error.

### Composición

Un punto de composición construirá configuración, adaptadores y casos de uso para cada entrada. CLI y UI podrán tener puntos de inicio distintos, pero compartirán la misma construcción del núcleo. Mientras hay grabación o procesamiento, el ciclo de eventos de Qt y el overlay mantienen vivo el único proceso de usuario aunque la ventana principal esté oculta. Sin trabajo activo, cerrar la ventana finaliza ese proceso. Un servicio tradicional de Windows y un segundo proceso quedan descartados para este hito.

## Fases recomendadas

### Fase 1 — Restaurar una línea base verde

Resolver la dependencia de `httpx` de forma explícita y reproducible, alinear el entorno soportado y ejecutar la suite completa. Registrar qué pruebas requieren simulación y cuáles son específicas de Windows. La salida de esta fase es una línea base confiable desde la que se pueda refactorizar sin confundir defectos previos con regresiones.

### Fase 2 — Extraer la orquestación de la CLI

Mover los recorridos de grabación, transcripción live, fallback, persistencia y reintento a una capa de aplicación. Preservar el comportamiento actual mediante pruebas de casos de uso y convertir la CLI en un adaptador delgado. La fase termina cuando la misma operación puede invocarse desde Python sin Typer, `input()` ni una terminal.

### Fase 3 — Crear la UI mínima y su punto de entrada

Incorporar una ventana Qt Widgets deliberadamente pequeña y la integración mínima del overlay. Implementar el ciclo de vida que, durante una grabación, permite ocultar la ventana principal, restaurarla con clic principal y acceder por clic derecho a detener/finalizar, pausar/reanudar y cancelar, manteniendo un único proceso. Establecer la conexión con la capa de aplicación mediante trabajadores y señales; el detalle visual avanza en paralelo según `LIVE_LOGO_PLAN.md`.

### Fase 4 — Control programático de inicio y detención

Reemplazar el contrato “grabar hasta Enter” por una sesión controlable: iniciar, recibir bloques, pausar, reanudar, detener de forma ordenada y devolver las fuentes capturadas. Pausar deja de capturar audio; los eventos de pausa/reanudación, sus timestamps de pared y la duración total pausada quedan en metadatos, mientras audio y transcript conservan el timeline del material realmente grabado. Adaptar micrófono, sistema y captura combinada sin duplicar la lógica. Mantener la CLI mediante un adaptador que traduzca Enter a la misma orden de detención.

### Fase 5 — Estados, progreso, cancelación y errores

Definir un modelo pequeño de estados observables —preparando, grabando, pausado, deteniendo, procesando, completado, fallido y cancelado, o su equivalente final— y eventos útiles sin prometer porcentajes falsos. La ventana debe permanecer responsiva, evitar acciones incompatibles, mostrar el dispositivo y destino efectivos, y distinguir entre detener una captura para finalizarla, pausar, cancelar una grabación de forma destructiva y cancelar procesamiento conservando el audio. Toda acción “Cancelar” exige confirmación previa y debe explicar su efecto real sobre los datos.

Publicar los eventos `grabando`, `pausado`, `procesando`, `completado`, `cancelado` y `error`, o sus equivalentes finales, junto con telemetría reducida de audio únicamente mientras la captura está activa y no pausada. Conectar a casos de uso compartidos las acciones de restaurar, detener/finalizar guardando, pausar y reanudar; cancelar grabación con eliminación confirmada; y cancelar procesamiento conservando una sesión reintentable. El contrato debe ser estable y comprobable sin depender de animaciones, logs o una captura duplicada.

### Fase 6 — Robustecer dispositivos y grabaciones largas

Cerrar las brechas detectadas por validación real: apertura y pérdida de dispositivos, ausencia de señal, desalineación entre fuentes, presión de colas, memoria y disco, fallos de red/proveedor y recuperación después de errores. Ejecutar una matriz de sesiones largas y escenarios de interrupción, con artefactos que permitan diagnosticar cada resultado.

### Fase 7 — Empaquetado mínimo para Windows

Producir un paquete instalable que incluya runtime y dependencias necesarias, cree una entrada reconocible en Inicio de Windows y permita abrir y desinstalar la aplicación sin preparar manualmente un entorno Python. Verificar rutas de datos, credenciales, permisos, recursos, comportamiento del overlay y diagnóstico en una instalación limpia. La elección concreta de empaquetador e instalador se toma en esta fase con una prueba corta, no se fija en este documento.

### Fase 8 — Validación final y preparación de cierre

Ejecutar la suite completa, pruebas de casos de uso, pruebas de UI relevantes y una matriz de aceptación en Windows instalado. Confirmar recorridos felices, fallos recuperables, cancelación, cierre durante grabación, continuidad con la ventana principal oculta, reapertura y detención con guardado desde el overlay, salida completa y sesiones largas. Verificar que la integración visual no afecte la captura; el pulido de la animación conserva sus propios criterios en `LIVE_LOGO_PLAN.md`. Revisar los artefactos resultantes y la documentación antes de declarar terminado el hito.

## Decisión acordada y decisiones pendientes

### Decisiones acordadas

#### Ciclo de vida al cerrar

Mientras se está grabando o procesando, cerrar la ventana principal la oculta y no detiene el trabajo ni el proceso de `here`. El overlay permanece disponible según el estado y un clic restaura la ventana. Cuando no existe grabación ni procesamiento en curso, cerrar la ventana termina la aplicación por completo: `here` no queda residente esperando una futura grabación. Si el trabajo termina con la ventana ya oculta, se muestra el estado terminal transitorio definido en `LIVE_LOGO_PLAN.md` y el proceso finaliza al desaparecer ese indicador. Todo sucede dentro del mismo proceso de usuario, sin servicio de Windows ni proceso auxiliar.

### Cancelación durante grabación

Cancelar una grabación exige confirmación explícita con un aviso claro de que el audio se perderá. Si el usuario confirma, se descarta definitivamente todo lo capturado, se eliminan sus temporales y no se crea una sesión recuperable. Esta conducta es distinta de cancelar procesamiento, que también pide confirmación pero siempre conserva el audio y una sesión reintentable.

### Salida completa con trabajo activo

“Salir” mientras se graba o procesa ejecuta **detener y guardar**: finaliza la captura o el procesamiento en curso, persiste la sesión recuperable y luego termina el proceso. No ofrece cancelación destructiva desde esa salida.

### Marca y señal visual

El menú de clic derecho durante grabación contiene detener y guardar/finalizar, pausar o reanudar, y cancelar, con iconos distintos. La reacción usa una única señal agregada de la captura o mezcla; no separa micrófono y audio del sistema en formas o colores distintos. El diseño final de la marca central se resuelve en el plan compañero. La primera integración debe preservar el contrato funcional sin fijar todavía el sistema visual definitivo.

### Elección inicial de dispositivos

Se usan **siempre los dispositivos predeterminados** del sistema, mostrando cuáles son. No hay selección explícita en la primera UI. Debe existir una prueba de señal y una respuesta clara ante cambios entre el arranque y la captura. La selección manual queda como mejora posterior si la matriz de hardware lo justifica.

### Política de conservación del audio

**El `audio.wav` se guarda siempre** al completar o fallar una sesión. Por ahora no hay retención automática ni purga: la política de plazos y borrado se define antes del empaquetado / Hito 5.

### Configuración de la credencial

La API key se configura mediante **`.env`** (formato actual). No hay UI de credenciales en este hito.

### Plataformas soportadas

Las **versiones recientes de Windows 11**. El listado formal de versiones y arquitecturas se cierra al final del desarrollo, antes del empaquetado.

### Timestamps y diarización ausentes

Cuando el modelo no proporciona hablantes o timestamps, la UI y los artefactos **no los inventan**: se omiten o se representan explícitamente como ausentes. La validación con modelos reales forma parte de la matriz de aceptación.

### Diferencia visible entre acciones

Detener, cancelar y recuperarse de un fallo deben distinguirse en la UI y en los mensajes: detener guarda y finaliza; cancelar (con confirmación) destruye o preserva según el estado; un fallo deja el estado y el audio recuperable a la vista.

### Otros puntos que conviene cerrar

- criterio medible de “reunión larga” y conjunto de equipos para aceptación.

## Criterios verificables de terminado

El hito se considera terminado cuando se cumplen conjuntamente estos criterios:

- existe un instalador validado en una instalación limpia de una versión de Windows declarada como soportada;
- la aplicación aparece en Inicio, abre sin consola y explica cualquier configuración necesaria;
- desde la ventana se puede iniciar y detener una captura combinada sin bloquear la UI;
- con grabación o procesamiento en curso, cerrar la ventana principal la oculta sin interrumpir el trabajo, y un clic en el overlay la restaura;
- durante la captura activa, el overlay permanece por encima de otras aplicaciones, carece de marco y fondo aparente, muestra únicamente el logo y puede arrastrarse por la pantalla; su continuidad visual posterior se valida mediante `LIVE_LOGO_PLAN.md`;
- durante grabación, el clic derecho ofrece tres acciones con iconos propios y significado distinto: detener y guardar/finalizar, pausar o reanudar según el estado, y cancelar;
- pausar detiene la captura de audio, registra eventos de pausa/reanudación con timestamps de pared y duración total pausada, cambia a un estado observable no reactivo y permite reanudar sin iniciar otra sesión; audio y transcript mantienen el timeline relativo al material grabado, sin silencio inventado;
- cancelar durante grabación solicita confirmación explícita indicando que se perderá el audio y, si se confirma, elimina lo capturado y sus temporales sin crear sesión recuperable;
- durante procesamiento, el menú contextual ofrece “Cancelar procesamiento” y ejecuta el mismo caso de uso que la ventana principal;
- cancelar procesamiento solicita confirmación, detiene el trabajo en curso sin eliminar el audio ni los metadatos necesarios, deja una sesión recuperable claramente identificada y permite reintentar después;
- la capa de aplicación publica estados observables de grabación, pausa, procesamiento, éxito, cancelación y error, además de telemetría de audio durante captura no pausada, sin duplicar la captura ni depender de logs;
- la UI muestra estados y errores coherentes, impide acciones incompatibles y comunica si hay audio recuperable;
- una reunión larga definida por la matriz de aceptación completa captura y procesamiento sin pérdida no explicada de bloques ni crecimiento no acotado de recursos;
- el resultado contiene audio recuperable, transcripción, Markdown, metadatos, chunks y errores cuando corresponda;
- timestamps y hablantes se preservan cuando el modelo los proporciona, y su ausencia se representa sin inventarlos;
- los fallos ensayados de red, proveedor, dispositivo y procesamiento terminan en recuperación o en un estado final comprensible, sin destruir el audio útil;
- en ausencia de grabación o procesamiento, cerrar la ventana termina por completo la aplicación y no deja un proceso residente;
- si el trabajo termina con la ventana principal oculta, el proceso finaliza después del indicador terminal transitorio, conservando la sesión y cualquier error para la próxima apertura;
- ocultar con trabajo activo, reabrir, detener, cancelar y salir tienen comportamientos decididos, probados y documentados;
- la CLI sigue operativa como interfaz secundaria y ejecuta los mismos casos de uso que la UI;
- la suite automatizada completa pasa desde un entorno limpio y la matriz manual de Windows queda registrada;
- instalación, uso, arquitectura real, decisiones y cambios relevantes están documentados según la disciplina siguiente.

## Disciplina documental durante la implementación

La documentación se actualiza junto con cada cambio relevante, no como una reconstrucción al final. El cierre del hito incluye revisar y actualizar, cuando corresponda:

- `README.md`, para reflejar instalación, configuración y uso actuales;
- `ARCHITECTURE.md`, para describir las fronteras y el flujo que realmente quedaron implementados;
- `docs/MASTER_PLAN.md` y este plan, para registrar estado, decisiones y cualquier cambio de alcance;
- `whats-new/`, con una nota breve cuando se libere una funcionalidad relevante.

Una nota de What's New no es un changelog exhaustivo: debe explicar qué cambió, por qué importa y cómo afecta al proyecto, como indica `whats-new/README.md`. No corresponde crearla durante la planificación; se escribe cuando una capacidad relevante queda implementada y lista para comunicar o liberar.

## Fuera de alcance

- convertir los artefactos actuales en la memoria estructurada de producto del hito 2;
- listar, abrir, filtrar o buscar el historial como producto navegable;
- RAG, preguntas entre reuniones, resúmenes inteligentes, decisiones, tareas o bloqueos derivados;
- una UI principal visualmente pulida o un sistema de marca completo; el logo vivo se diseña ahora, pero se gobierna mediante su plan paralelo;
- inicio automático y sistema completo de actualización automática;
- servicio tradicional de Windows o proceso persistente independiente, salvo una decisión posterior explícita;
- sincronización en la nube, colaboración, cuentas y uso multidispositivo;
- clientes para macOS, Linux, web o móvil;
- integraciones con calendarios, videollamadas u otras aplicaciones.

La sesión recuperable de este hito es una base técnica para la memoria futura, no la implementación anticipada de los hitos de memoria, navegación o inteligencia.
