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
implementación queda fuera de esta ejecución. La validación de Ubuntu se conserva
aparte y no se ejecuta ni condiciona el cierre del hito 1.
