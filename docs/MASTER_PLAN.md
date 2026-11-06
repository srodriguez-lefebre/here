# Plan maestro de producto — here

## Visión

Convertir las reuniones en una memoria confiable, consultable y útil. `here` debe poder acompañar una conversación extensa desde su captura hasta su consulta posterior, conservando siempre un vínculo verificable con lo que ocurrió originalmente.

Este documento describe los grandes hitos del producto. Deliberadamente no fija fechas, tecnologías ni tareas de implementación. Cada hito tendrá después su propio plan específico, con alcance, decisiones, riesgos y criterios de validación detallados.

## Principios generales

- La confiabilidad de la memoria depende primero de no perder la conversación original.
- Toda información derivada debe poder rastrearse hasta la reunión y el fragmento que la sustenta.
- La privacidad y el control sobre los datos forman parte del producto, no son un agregado posterior.
- La memoria estructurada y la memoria navegable se mantienen como hitos separados: una se ocupa de representar y preservar correctamente la información; la otra, de hacerla accesible para una persona. Tienen riesgos y criterios de éxito diferentes.

## Hito 1 — Captura confiable

### Propósito

Crear una base de información fiel incluso durante reuniones largas o ante fallos parciales. El sistema debe capturar el micrófono y el audio del sistema, transcribir sin pérdida relevante y conservar el contexto temporal y de hablantes necesario para entender lo ocurrido. Este hito incluye una aplicación mínima instalable para Windows como interfaz principal y un logo flotante, reactivo y siempre visible durante la captura, manteniendo la CLI como interfaz secundaria sobre el mismo núcleo.

### Definición de terminado

El hito está terminado cuando una reunión larga puede grabarse y transcribirse de extremo a extremo con ambas fuentes de audio desde una aplicación mínima de Windows; con captura o procesamiento en curso, cerrar la ventana permite continuar y volver mediante el logo flotante, mientras que en reposo cierra la aplicación por completo; una interrupción o fallo no obliga a perder la sesión; y el resultado conserva audio recuperable, timestamps y atribución de hablantes con una calidad suficiente para volver al momento original.

## Hito 2 — Memoria estructurada

### Propósito

Transformar cada captura en una entidad durable y coherente del producto. La memoria debe representar reuniones, participantes, fechas, audio, transcripción, segmentos, estados y ciclo de vida sin depender de archivos aislados o convenciones frágiles.

### Definición de terminado

El hito está terminado cuando cada reunión y sus componentes pueden crearse, actualizarse, finalizarse, recuperarse y conservarse con identidad y relaciones claras; sus estados son consistentes; y la evolución o eliminación de una reunión puede gestionarse sin romper la integridad de la memoria.

## Hito 3 — Memoria navegable

### Propósito

Permitir que una persona encuentre y recorra la memoria acumulada sin conocer cómo está almacenada. Debe ser posible pasar de una necesidad concreta a la reunión relevante y, desde allí, al fragmento original.

### Definición de terminado

El hito está terminado cuando las reuniones se pueden listar, abrir, filtrar y buscar de forma útil; los resultados ofrecen suficiente contexto para elegir; y cada coincidencia permite regresar con precisión al audio o segmento original que la contiene.

Este hito no se considera resuelto solo porque la información esté bien estructurada. Su éxito se mide por la capacidad real de encontrar y comprender contenido; el hito anterior se mide por la integridad y durabilidad de la representación.

## Hito 4 — Inteligencia sobre la memoria

### Propósito

Convertir el historial de reuniones en conocimiento accionable. El sistema debe responder preguntas, relacionar información de varias reuniones y extraer decisiones, tareas y bloqueos sin ocultar la evidencia de origen.

### Definición de terminado

El hito está terminado cuando el usuario puede consultar una o varias reuniones y obtener respuestas y extracciones útiles, diferenciando con claridad hechos, síntesis e incertidumbre; y cuando cada afirmación relevante incluye citas verificables que conducen a los fragmentos originales.

## Hito 5 — Producto cotidiano

### Propósito

Completar la productización de las capacidades anteriores para que funcionen de manera habitual, con poca fricción y confianza. La UI y el recordatorio flotante de grabación nacen en el primer hito; aquí se maduran su diseño, la operación avanzada en segundo plano y el ecosistema del producto.

### Definición de terminado

El hito está terminado cuando `here` puede ejecutarse en segundo plano, ofrece una interacción simple y estados comprensibles, protege la privacidad y el control de los datos, y dispone de una experiencia sostenible de instalación y actualización. Las integraciones necesarias para el uso cotidiano están disponibles cuando aportan valor y respetan esos mismos principios.

## Desarrollo posterior

Estos hitos marcan la dirección, no sustituyen los planes de trabajo. Antes de implementar cada uno se elaborará un plan específico que precise su alcance, decisiones pendientes, riesgos, validación y secuencia interna, manteniendo la definición de terminado de este documento como referencia de producto.

La ejecución autónoma de los cinco hitos comenzó el 2026-10-03 por delegación
explícita del usuario. Los planes específicos, decisiones y PRs se registran en
[`development/DELIVERY.md`](development/DELIVERY.md); el diseño de los hitos 2–5
está en [`development/PRODUCT_DESIGN.md`](development/PRODUCT_DESIGN.md). La matriz
de [`development/ACCEPTANCE.md`](development/ACCEPTANCE.md) distingue código,
verificación automatizada y aceptación con hardware/instalador/proveedor. Cada hito
permanece abierto hasta demostrar conjuntamente su definición de terminado.

El 2026-10-03 el usuario acotó la ejecución actual: cerrar el hito 1 en Windows
y terminar ahí. Los hitos 2–5 conservan su lugar en este plan maestro, pero su
implementación queda fuera de esta ejecución. El workflow manual de Ubuntu se eliminó el 2026-11-04 (ruling 11); Linux no es
objetivo de producto ni condiciona ningún hito.


### Estado al 2026-11-04

El hito 1 quedó cerrado bajo el alcance reducido de abajo (PR #14, v0.2.0). El
2026-11-04 el usuario resolvió doce decisiones abiertas, registradas como rulings en
[`development/DELIVERY.md`](development/DELIVERY.md). El 2026-11-05 quedó escrito el
plan completo de los hitos 2 a 5, cada uno con criterios verificables y con el SPEC y
PLAN de cada grupo de PR: [`MILESTONE_2_PLAN.md`](MILESTONE_2_PLAN.md),
[`MILESTONE_3_PLAN.md`](MILESTONE_3_PLAN.md), [`MILESTONE_4_PLAN.md`](MILESTONE_4_PLAN.md)
y [`MILESTONE_5_PLAN.md`](MILESTONE_5_PLAN.md). Ninguno está en ejecución (ruling 15).
La hoja de arranque para agentes es `CLAUDE.md` en la raíz y el estado corto es
`STATUS.md`.

### Alcance final autorizado del hito 1 — 2026-10-03

Por indicación explícita del usuario, se omiten la prueba de captura de
al menos dos horas con pausas, recuperación y medición de recursos, y la
transcripción larga con audio generado, timestamps y diarización. No se
ejecutan ni se presentan como aprobadas. El cierre de M1 se evalúa con
el alcance restante en Windows: correcciones, distribución instalable,
verificaciones de runtime/instalación, avisos, checksums y PR revisadas
y mergeadas. Al cerrar M1 termina esta ejecución; M2-M5 quedan pendientes.

### Ejecución vigente al 2026-11-05

La instrucción explícita del usuario registrada como ruling 20 reemplaza la pausa
de implementación del ruling 15 y el límite histórico de terminar en M1. Se
implementan M2–M5 en los nueve grupos de PR que enumera `STATUS.md`, conservando
los principios y las definiciones de terminado de este plan. El grupo actual es
M2 A: la base de catálogo de su tarea 1 (`9a80060`) tiene review independiente
aceptada y verificación local; `M2A-T1-FINAL-20261105` registra 778 pasan/7 skips
en `ACCEPTANCE.md`, sin sustituir CI del grupo.
La tarea 2 conserva raw y segmentos en todos los recorridos. La tarea 3 integró
las rondas 2/3 en `8087e25`: reviews acotadas marcan originales F1–F7, NB1 y NB2
ADDRESSED. El full histórico `1f1bc88` dio cinco fallos/920 pasan/7 skips; sus
correcciones tienen cobertura; su full histórico pasa 940 tests/7 skips en 325,09 s.
La review única de todo A terminó con veredicto fuente/docs changes requested por
GR1–GR3 Important: replay tras metadata, aislamiento de recibos inválidos y
divergencia de catálogo al cancelar captura. La tanda única de fixes R38 está
integrada en `7bfd0cb`, equivalente a Green `ec55b95`, con foco final de grupo y
recovery de 26 pasan/34,46 s. Marker 3 sella evidencia final antes de metadata;
recibos 2/3 y schema 2 se conservan. La re-review final independiente APPROVED
fuente7bfd/fixtureef70/docs8845877 marca GR1–GR3/GD1/GD2 ADDRESSED. El full940/7
corresponde al histórico `8087e25`; faltan los gates de GitHub. Full 3 de `7bfd0cb` dio ocho fallos/
945 pasan/siete skips en 364,23 s, por dos wrappers de tests de captura que rechazan
el callback aditivo. Sus assertions/cortes de crash se conservan al reenviarlo.
La corrección test-only `ef70f78` pasó foco13/24,82 s y full 4:953 pasan/7 skips/
358,94 s, ambos smokes wheel pasan con el mismo source/probe no editable exacto.
Los fallos históricos permanecen. Tarea 3 aceptada técnicamente en local;
grupo A e hito siguen abiertos hasta sus gates y alcance restantes.
El ciclo
de vida/borrado y el cierre de M2 siguen pendientes; no cambia la versión a 0.3.0
hasta cerrar el grupo B. Su PLAN ajustado se entrega antes en un PR de docs según
ruling 14, sin sustituir los gates del grupo A.

La nueva instrucción del ruling 35 ordena cerrar y mergear A primero y completar
los PRs docs de contratos. Después root crea dos chats separados del proyecto para
M2 B y M3 A, cada uno con rama/worktree y brief sobre la SHA mergeada real; IDs,
bases y responsables futuros siguen pendientes en STATUS. Root conserva contratos,
leases, integración y merges. Cada chat ejecuta su grupo con RED/GREEN/review/docs,
sin reiniciar ni duplicar trabajo. Tras M2 B puede adelantarse M5 B con claims
pendientes hasta M4 B; se mantienen las demás dependencias y los mismos criterios.

Cada merge exige review independiente, Copilot recibido y respondido sobre el head
actual y CI de Windows verde en ese head; el ruling 20 no permite omitir una
condición por demora. La evidencia automática local no certifica hardware,
proveedor ni instalación limpia. La prueba de dos horas permanece omitida por
ruling 5, sin presentarla como aprobada.
