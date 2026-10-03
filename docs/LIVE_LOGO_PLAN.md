# Plan del logo vivo

## Propósito

Diseñar e integrar el recordatorio visual de que `here` está grabando y su continuidad de estado hasta que una sesión termina. La experiencia debe ser visible sin ocupar una ventana convencional, permitir volver rápidamente a la aplicación y comunicar el paso de captura a procesamiento y resultado.

Este plan acompaña al [plan del hito 1](MILESTONE_1_PLAN.md) y puede avanzar en paralelo. No redefine la confiabilidad de captura: usa los estados, la telemetría y las acciones que entrega el hito para construir una experiencia visual coherente.

## Frontera con el hito 1

### Requisito funcional del hito 1

- disponer de estados y eventos estables para grabación, pausa, procesamiento, finalización, cancelación y error;
- emitir telemetría de nivel desde el audio ya capturado, sin duplicar la captura;
- mostrar el overlay durante una grabación y mantener la aplicación activa si la ventana principal se oculta mientras hay grabación o procesamiento;
- restaurar la ventana principal con clic primario;
- durante grabación, ofrecer desde el clic derecho detener y guardar/finalizar, pausar o reanudar, y cancelar, con un icono propio para cada acción;
- ofrecer “Cancelar procesamiento” desde el clic derecho mientras se procesa, conservando audio y sesión recuperable;
- mantener captura, transcripción y persistencia fuera del hilo de UI.

### Refinamiento de esta iniciativa paralela

- diseñar el lenguaje gráfico del logo y sus transiciones;
- convertir el nivel de audio en una reacción visual legible;
- resolver los morphs entre grabación, procesamiento, éxito y error;
- ajustar ritmo, interpolación, tiempos y pulido visual;
- validar que la experiencia resulte notoria sin distraer ni afectar al motor.

El hito puede validar su contrato con representaciones mínimas. La calidad final de los morphs no debe ocultar una falla de captura ni bloquear el trabajo de robustez, pero esta iniciativa se trabaja ahora y no se difiere al hito 5.

## Experiencia confirmada

La experiencia es una pequeña ventana flotante que muestra únicamente el logo:

- transparente y sin marco o fondo aparente;
- siempre visible por encima de las demás aplicaciones;
- arrastrable libremente por el espacio de pantalla;
- ubicado al comenzar cada grabación en la esquina inferior derecha, dentro del área útil del monitor activo o principal correspondiente y con margen suficiente sobre la barra de tareas; el movimiento del usuario vale solo para esa sesión y no se persiste para la grabación siguiente;
- sin controles permanentes ni apariencia de mini reproductor;
- con un único color configurable que se aplica tanto a la onda/perímetro de grabación como a la espiral de procesamiento; los colores finales de éxito y error son semánticos y fijos.

Una bandeja del sistema no es la experiencia principal. Podría existir como detalle técnico secundario si la integración con Windows lo necesitara, pero el recordatorio de grabación y las interacciones definidas aquí pertenecen al overlay.

## Secuencia visual

| Estado | Comportamiento del overlay | Fuente de información |
|---|---|---|
| Grabando | El perímetro u onda del logo usa el color configurado, reacciona a una única señal combinada y recuerda de forma inequívoca que la captura está activa | Nivel agregado de la captura o mezcla emitido por el núcleo |
| Pausado | El logo permanece visible en una variante inequívoca, estática y no reactiva; no simula nivel de audio ni confunde pausa con silencio | Estado de pausa y eventos temporales explícitos de la sesión |
| Procesando | El mismo logo evoluciona hacia una indicación circular o espiral de carga con el mismo color configurado; deja de reaccionar al audio | Estado de aplicación y, si existe, progreso honesto |
| Completado | La forma evoluciona a un check o tic verde, permanece visible unos segundos y desaparece | Evento de finalización correcta |
| Error | La forma evoluciona a una cruz roja, permanece visible unos segundos y desaparece; el error real sigue registrado y visible en la aplicación | Evento y resultado de error recuperable o final |

La secuencia debe sentirse como la evolución de un mismo objeto, no como widgets independientes que aparecen y desaparecen sin relación. La onda de grabación y la espiral de procesamiento comparten el color elegido por el usuario; esa preferencia no altera el verde fijo del check ni el rojo fijo de la cruz. Esto es una dirección de diseño, no una definición de la forma final de la marca.

La desaparición de la cruz es solo visual. El estado fallido, su detalle y la existencia de audio recuperable deben permanecer registrados en la sesión y visibles al restaurar o reabrir la aplicación.

## Interacciones confirmadas

- **Clic principal:** abre o restaura la ventana principal.
- **Arrastre:** mueve el logo a otra posición de la pantalla sin activar la acción principal. La nueva posición dura hasta que termina esa sesión y no modifica el punto de inicio de futuras grabaciones.
- **Clic derecho durante grabación:** abre un menú pequeño con tres acciones claramente diferenciadas y un icono propio para cada una: **Detener y guardar/finalizar**, **Pausar** o **Reanudar** según el estado, y **Cancelar**. Cancelar pide confirmación y advierte claramente que se perderá el audio; al confirmar elimina la captura y sus temporales sin crear una sesión.
- **Clic derecho durante procesamiento:** ofrece **Cancelar procesamiento** y pide confirmación. La acción detiene el trabajo, conserva el audio capturado y deja una sesión recuperable para reintentar; no elimina datos.
- **Pausa:** deja de capturar audio, registra eventos de pausa y reanudación con timestamps de pared y duración total pausada, y cambia el logo a una representación estática y no reactiva. Reanudar continúa la misma sesión; el audio y el transcript mantienen su timeline relativo al material capturado, sin insertar silencio inventado.

El gesto de arrastre y el clic deben distinguirse de forma estable para evitar aperturas accidentales. Todas las acciones deben invocar los casos de uso comunes de la aplicación; el overlay no implementa rutas paralelas. Toda acción de cancelar pide confirmación, pero su consecuencia depende del estado: durante grabación es destructiva y no crea sesión; durante procesamiento es recuperable y conserva audio y metadatos.

## Contrato de integración

El overlay consume una interfaz pequeña y explícita:

- estado actual de la ejecución;
- evento de transición entre estados;
- un único nivel de audio combinado, agregado y limitado en frecuencia solo mientras se graba;
- eventos de pausa/reanudación con timestamps de pared y duración total pausada, manteniendo separado el timeline del material capturado;
- resultado de finalización o error;
- resultado de cancelación recuperable;
- comandos para restaurar la ventana, detener/finalizar guardando, pausar, reanudar, cancelar grabación y cancelar procesamiento de forma recuperable;
- una preferencia de color compartida por la onda de grabación y la espiral de procesamiento.

No debe leer logs, inspeccionar archivos ni abrir su propio dispositivo de audio. La telemetría visual se calcula a partir del flujo que el núcleo ya captura y se reduce antes de llegar a la UI. La ausencia de progreso medible durante procesamiento se representa como actividad indeterminada, no con un porcentaje inventado.

## Dirección técnica

La implementación prevista vive en el mismo proceso PySide6/Qt Widgets que la ventana principal. El overlay es una superficie de presentación separada, mientras el coordinador, la captura y la transcripción permanecen en trabajadores de fondo.

El proceso continúa sin ventana principal visible únicamente mientras hay grabación o procesamiento. Si el trabajo termina con la ventana oculta, el indicador terminal se muestra durante su intervalo transitorio y, al desaparecer, `here` finaliza. La sesión completada, fallida o cancelada permanece guardada para la próxima apertura. Cuando no existe trabajo activo, cerrar la ventana principal termina `here` por completo; el overlay no convierte la aplicación en un residente permanente.

La frecuencia de render debe estar desacoplada de la frecuencia de captura. El procesamiento de la señal para visualización será acotado y descartable: nunca debe introducir presión sobre la cola de audio ni comprometer la escritura recuperable. El comportamiento siempre visible, el arrastre y la transición entre pantallas deben validarse en Windows real.

## Fases de la iniciativa

### 1. Prototipo de interacción

Probar una superficie transparente, sin marco y siempre visible con estados sintéticos. Validar posición inicial, arrastre limitado a la sesión, clic principal, clic derecho y restauración de la ventana antes de conectarla al audio.

### 2. Contrato con la aplicación

Consumir los estados, eventos, comandos y telemetría definidos por el hito 1. Verificar que el overlay puede probarse con una fuente de eventos simulada y que no depende de logs ni del adaptador de captura concreto.

### 3. Estado de grabación

Explorar y validar la respuesta del perímetro u onda a un único nivel agregado de audio, la configuración de color, la legibilidad como recordatorio y la variante inequívoca de pausa. No se usarán anillos, sectores ni colores separados para micrófono y audio del sistema. Durante pausa, la representación será estática y no habrá reacción de audio simulada; el detalle gráfico se resuelve por criterio de diseño. Ajustar la frecuencia sin convertir esta fase en el diseño definitivo de la marca.

### 4. Procesamiento y finalización

Construir la transición no reactiva hacia carga circular o espiral, habilitar la cancelación recuperable desde su menú contextual, y crear el morph de éxito hacia check/tic verde y el morph de fallo hacia cruz roja. Validar que ambos estados finales permanezcan unos segundos y desaparezcan sin borrar el resultado persistido. El intervalo aproximado de 2–4 segundos sirve como punto de partida; la duración exacta queda pendiente de validación.

### 5. Pulido integral

Probar la secuencia completa —incluidos check verde y cruz roja— en Windows, su convivencia con distintas aplicaciones y que el costo visual no degrade la captura o el procesamiento. Confirmar que el error persiste en la sesión y la aplicación después de desaparecer el indicador.

## Criterios de validación visual y de interacción

- durante grabación, el overlay contiene únicamente el logo, permanece visible y reacciona al audio real;
- el usuario puede moverlo por la pantalla y su posición no impide restaurar la ventana con un clic posterior;
- cada grabación comienza en la esquina inferior derecha del área útil del monitor correspondiente, sin superponerse a la barra de tareas, aunque el usuario haya movido el logo en la sesión anterior;
- durante grabación, el menú de clic derecho contiene detener y guardar/finalizar, pausar o reanudar según el estado, y cancelar, cada acción con su propio icono;
- al pausar, la captura de audio se detiene, el overlay cambia a un estado inequívoco, estático y no reactivo, y los metadatos registran eventos de pausa/reanudación con tiempo de pared y duración total pausada; al reanudar vuelve la reacción dentro de la misma sesión sin insertar silencio en el timeline del audio o transcript;
- cancelar una grabación siempre pide confirmación, explica que se perderá el audio y, al confirmarse, elimina la captura y temporales sin crear una sesión recuperable;
- durante procesamiento, el menú de clic derecho contiene “Cancelar procesamiento”;
- cancelar procesamiento siempre pide confirmación, detiene el trabajo sin eliminar el audio, conserva una sesión recuperable y permite reintentar desde la aplicación;
- al detener, cesa la reacción al audio y el logo pasa a un estado de procesamiento no reactivo;
- al completar, la evolución termina en un check/tic verde fijo, visible durante unos segundos antes de desaparecer;
- al fallar la grabación o el procesamiento, la evolución termina en una cruz roja fija, visible durante unos segundos antes de desaparecer;
- la desaparición de la cruz no borra ni oculta el error persistido: la sesión y la aplicación conservan el estado y el detalle;
- el mismo color personalizable se usa en la onda de grabación y la espiral de procesamiento, sin modificar los colores semánticos finales;
- la reacción durante grabación usa una única señal combinada, sin separar micrófono y audio del sistema por anillos, sectores o colores;
- ninguna transición depende de analizar logs o de realizar una segunda captura;
- la animación sigue siendo legible y su trabajo está limitado para no afectar el pipeline;
- la duración exacta de los indicadores finales queda validada antes de cerrar la iniciativa.

## Parámetros menores delegados

La posición inicial queda fijada en la esquina inferior derecha del área útil del monitor activo o principal correspondiente, con margen sobre la barra de tareas. El margen exacto, el tamaño inicial, la frecuencia de animación, la duración concreta de los estados terminales dentro del rango aproximado de 2–4 segundos y las curvas de easing pueden resolverse por criterio de diseño e implementación.

Estas decisiones no requieren consulta mientras preserven legibilidad, rendimiento y el comportamiento acordado. Deben elevarse al usuario solo si cambian el producto o introducen un tradeoff material.

## Decisiones pendientes

### Representación de una cancelación

Definir la transición visual posterior a una cancelación solicitada por el usuario. No debe presentarse como éxito ni borrar la distinción entre cancelación y error. También debe respetar la diferencia de datos: cancelar grabación no deja sesión, mientras cancelar procesamiento sí deja una sesión recuperable.

### Forma final de la marca

Diseñar la marca central y las geometrías de onda, carga y check como un sistema reconocible. Este plan fija la secuencia y el comportamiento, no el logo definitivo.

## Fuera de alcance de esta iniciativa

- controles visibles de mini reproductor;
- eliminación destructiva al cancelar procesamiento;
- convertir la bandeja del sistema en la superficie principal;
- capturar, transcribir o persistir audio dentro de la capa visual;
- servicio tradicional de Windows o proceso separado;
- rediseño completo de la ventana principal.
