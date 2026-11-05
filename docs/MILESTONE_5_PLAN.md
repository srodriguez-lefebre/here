# Hito 5 — Producto cotidiano

Estado: plan preparado el 2026-11-05. **No está en ejecución (ruling 15).** Arranca
solo con una instrucción explícita del usuario, después de cerrar M4. El diseño
detallado está en [`development/PRODUCT_DESIGN.md`](development/PRODUCT_DESIGN.md),
secciones 7, 8 y 9, que es decisión tomada (ruling 3). Este plan no lo repite: fija
alcance, orden, criterios verificables y lo que queda fuera.

## Propósito

Que `here` pueda usarse todos los días sin fricción ni sorpresas: abrir sin credencial,
saber dónde están los datos y qué sale a la red, llevarse la memoria a otra máquina y
traerla de vuelta, dejar la aplicación en la bandeja si se quiere, instalar una versión
nueva sin perder nada, y borrar lo que se decida borrar sin que quede rastro en lo que
`here` administra. Es la productización de M1 a M4, no una capacidad nueva de memoria.

## Qué dejan M1 a M4 y qué falta

Lo que ya existe y M5 consume tal cual:

- De M1, real y cerrado: rutas por usuario en `%LOCALAPPDATA%/here` con `HERE_DATA_DIR`
  (`src/config/paths.py`), credencial opcional por `.env` (`src/config/settings.py`),
  preferencia de color en `QSettings` (`src/ui/preferences.py`), overlay con sus
  semánticas de cierre, acción `Salir` que detiene y guarda, mutex de aplicación para
  el instalador (`src/distribution.py`), entrada `--smoke-check`, pipeline PyInstaller
  onedir + Inno Setup por usuario (`packaging/`, `scripts/`,
  `.github/workflows/windows-package.yml`) con avisos, inventario y checksums.
- De M2, según su plan: catálogo SQLite con revisiones inmutables, operaciones
  recuperables en `pending_operations`, borrado íntegro y backups administrados.
- De M3, según el diseño: biblioteca, búsqueda FTS5 y reproductor con seek.
- De M4, según el diseño: respuestas y extracciones persistidas con citas verificadas.

Lo que falta y M5 construye:

- Ajustes no secretos versionados en archivo, escritos de forma atómica; hoy solo
  existe el color en `QSettings` y el resto son variables de entorno.
- Primera ejecución que explique dónde se guardan los datos y qué usa red.
- Logs rotativos y redactados; hoy la GUI no configura ningún sink de `loguru`.
- Bandeja opcional, instancia única, lock de captura entre CLI y GUI, inicio opt-in.
- Exportación e importación portables con validación hostil.
- Build que incluya los plugins multimedia que M3 necesita (el `windows.spec` de M1
  los excluye a propósito), pantalla Acerca de, prueba de actualización N→N+1 y la
  matriz final de aceptación en Windows.

## Alcance

Entra en M5:

- Archivo de ajustes versionado con `data_dir`, `accent`, `close_to_tray`,
  `launch_at_login`, `intelligence_mode`, `intelligence_model`, nivel de log y política
  de retención manual; compatible con el `.env` existente; nunca copia credenciales.
- Pantalla de primera ejecución y pantalla de ajustes con uso de disco y borrado de
  reuniones seleccionadas por el flujo íntegro de M2.
- Cambio de raíz de datos como caso de uso explícito: detiene trabajos, copia,
  verifica, cambia la configuración y conserva rollback.
- Logs rotativos (10 MiB por archivo, tres archivos) sin transcript, preguntas,
  prompts, nombres de participantes, encabezados ni secretos; exportación de
  diagnóstico bajo acción explícita que enumera lo que incluye.
- Bandeja opcional, instancia única de GUI, lock de captura por usuario y raíz, inicio
  al iniciar sesión opt-in y reversible, salida segura con trabajo activo.
- Formato `.here.zip` versionado, exportación con snapshot consistente y progreso,
  importación validada en staging publicada por las operaciones recuperables de M2.
- Build empaquetado con multimedia y sonda de FTS5, pantalla Acerca de, procedimiento
  de actualización documentado, prueba N→N+1, pruebas Qt, accesibilidad y matriz final.
- Versión 1.0.0 al cerrar (ruling 9).

No entra en M5:

- Sincronización en nube, cuentas, telemetría remota, actualizador automático,
  descarga o ejecución silenciosa de binarios.
- Cifrado propio de la biblioteca; se documenta la protección del usuario de Windows
  y la compatibilidad con el cifrado de disco del sistema.
- Integraciones con calendario, correo o videollamadas.
- Retención o purga automáticas; la retención es indefinida con borrado manual.
- Servicio de Windows o proceso residente distinto de la bandeja opt-in.
- Rediseño del overlay o de la marca (ruling 8).

## Grupos de PR

Tres grupos, cada uno con su SPEC y PLAN en `development/`, revisables por separado.

| Grupo | Tareas del diseño | Documentos | Estado |
|---|---|---|---|
| A. Ajustes y runtime | 1 ajustes, rutas, primera ejecución, diagnóstico · 2 bandeja, instancia única, inicio opt-in, salida segura | `M5_SETTINGS_SPEC.md`, `M5_SETTINGS_PLAN.md` | plan escrito, sin ejecutar |
| B. Portabilidad y privacidad | 3 exportación e importación validadas, con y sin audio | `M5_PORTABILITY_SPEC.md`, `M5_PORTABILITY_PLAN.md` | plan escrito, sin ejecutar |
| C. Distribución y aceptación | 4 build e instalador con actualización real · 5 pruebas Qt, inspección visual, accesibilidad y matriz Windows | `M5_RELEASE_SPEC.md`, `M5_RELEASE_PLAN.md` | plan escrito, sin ejecutar |

Orden: A, después B, después C. C es el último porque certifica el paquete con todo
lo anterior adentro; B depende de los ajustes de raíz de datos de A.

## Decisiones acordadas

- Overlay y marca no cambian (ruling 8). La bandeja no reemplaza al overlay: el
  overlay sigue siendo el indicador de trabajo activo.
- Hay máquina Windows limpia y credencial real a pedido; la reunión de dos horas con
  medición de recursos sigue omitida y así se registra en la matriz (ruling 5).
- Versión 1.0.0 al cerrar M5 (ruling 9). Un tag no publicado se renumera sin costo.
- Retención indefinida con borrado manual. Sin purga programada; si algún día se
  agrega, es opt-in y reutiliza el borrado íntegro de M2.
- Sin cifrado propio. Sin nube ni cuentas. Bandeja e inicio al iniciar sesión son
  opt-in, apagados por defecto, reversibles y sin privilegios de administrador.
- Sin actualizador automático. Acerca de muestra versión, procedencia del build y un
  enlace al canal oficial verificado; si no existe canal, muestra el procedimiento de
  instalación manual y lo dice.
- El instalador de una versión nueva sigue rechazando reemplazar una aplicación en
  ejecución, como en M1. "Cerrar de forma segura" se cumple con `Salir` desde la
  aplicación, que detiene y guarda; el mensaje del instalador lo indica.
- La prueba N→N+1 usa como N el instalador 0.2.0 de M1. Si el artefacto de CI
  expiró, N se reconstruye desde `beeedac` con el mismo pipeline y se registra su hash
  como reconstrucción, no como el binario original.
- Los servicios de M5 (`memory/portability.py`, ajustes, diagnóstico) no importan Qt
  ni Typer; la instancia única vive en la UI y el lock de captura en la capa de
  aplicación.
- Idioma: este plan y los criterios en español; SPEC y PLAN en inglés (ruling 6).

## Criterios verificables de terminado

El hito se considera terminado cuando se cumplen conjuntamente:

1. La aplicación abre sin `OPENAI_API_KEY`, con la biblioteca navegable y la
   recuperación local disponibles; la primera ejecución muestra dónde se guardan los
   datos, qué operaciones usan red y cómo configurar la credencial, sin revelar su valor.
2. Los ajustes no secretos se leen desde un único archivo versionado bajo la raíz de
   datos, se escriben con temporal + flush + replace, y un archivo corrupto o de
   versión futura produce un error legible sin perder la copia anterior.
3. Un `.env` existente sigue funcionando; ninguna credencial aparece en el archivo de
   ajustes, en `QSettings`, en exportaciones, en diagnósticos ni en logs, verificado
   con un valor canario.
4. Cambiar la raíz de datos detiene trabajos, copia y verifica hashes, cambia la
   configuración y, ante un fallo a mitad, deja la raíz anterior intacta y vigente.
5. Los logs rotan a 10 MiB por archivo con tres archivos, y un recorrido completo
   (grabación sintética, búsqueda, pregunta, exportación, borrado) con frases canario
   en transcript, títulos, nombres y pregunta no deja ninguna frase canario en los logs.
6. El diagnóstico exportado enumera su contenido, se genera solo por acción explícita,
   y no incluye sesiones, `.env`, credenciales ni frases canario.
7. Con `close_to_tray=false` (default) cerrar en reposo sale y cerrar con trabajo
   oculta; con `close_to_tray=true` cerrar deja la bandeja con abrir, nueva grabación
   y salir; `Salir` con trabajo activo detiene y guarda antes de terminar.
8. Una segunda instancia de la GUI restaura la ventana de la primera y termina; CLI y
   GUI no pueden grabar a la vez sobre la misma raíz, y los comandos de consulta sí
   pueden coexistir con la GUI.
9. `launch_at_login=true` crea una única entrada en `HKCU\...\Run` con la ruta del
   ejecutable entre comillas, solo en build frozen; `false` la elimina; en modo
   desarrollo la opción se rechaza con un mensaje.
10. Exportar reuniones seleccionadas produce un `.here.zip` cuyo manifiesto lista
    versión, IDs, hashes y tamaños, con o sin audio, sin DB cruda, rutas absolutas,
    `.env`, credenciales ni logs; el archivo aparece solo completo (temporal + rename).
11. Importar ese paquete en una raíz nueva conserva IDs de reunión, revisiones,
    segmentos, tiempos, participantes, asignaciones, citas y hash de audio; reimportar
    el mismo paquete es no-op; un paquete con el mismo UUID y contenido distinto
    reporta conflicto sin sobrescribir.
12. Un corpus de paquetes hostiles (rutas absolutas, `..`, UNC, prefijos de unidad,
    ADS, nombres de dispositivo, duplicados normalizados, enlaces, manifiesto futuro,
    ratio de compresión excesivo, exceso de archivos) se rechaza sin escribir nada
    fuera de `.staging/` y sin ejecutar contenido.
13. Borrar una reunión desde ajustes deja cero rastro de su transcript en DB, FTS,
    respuestas de M4, caches, temporales de exportación y backups administrados,
    verificado buscando frases canario después de reiniciar.
14. El bundle incluye los plugins multimedia de Qt y pasa la sonda de FTS5; el
    `--smoke-check` empaquetado lo reporta; ningún plugin excluido en M1 vuelve sin
    justificación en el inventario.
15. El build es reproducible desde el lockfile con versiones de herramientas
    registradas, y `SHA256SUMS.txt` cubre instalador, ZIP, wheel e inventario.
16. En la máquina limpia disponible, la aplicación instalada abre desde Inicio, graba
    ambas fuentes por defecto, reproduce un WAV con seek y responde una pregunta en
    modo local, sin repo ni entorno de desarrollo en el PATH.
17. Instalar 1.0.0 sobre el 0.2.0 real conserva la biblioteca y los datos sintéticos,
    migra con backup y abre; desinstalar 1.0.0 conserva `%LOCALAPPDATA%\here`; una
    prueba aparte borra los datos solo por la acción explícita.
18. Acerca de muestra versión y procedencia del build; el enlace al canal oficial
    existe y responde, o la pantalla dice que no hay canal y cómo instalar a mano; no
    se afirma firma Authenticode sin certificado.
19. Las pantallas nuevas pasan recorrido por teclado, nombres accesibles y contraste
    en pruebas Qt offscreen; la inspección visual queda registrada con capturas.
20. La matriz de `ACCEPTANCE.md` para M5 nombra la corrida de cada fila; la reunión
    de dos horas figura como omitida por el ruling 5, no como aprobada; `README.md`,
    `ARCHITECTURE.md`, este plan y `WHATS_NEW.md` reflejan lo implementado.

## Lo que queda pendiente de decidir

- Si `accent` migra de `QSettings` al archivo de ajustes o convive. Default: migra
  una sola vez al primer arranque de M5 y `QSettings` deja de usarse.
- Dónde vive el lock de captura. Default: `<data_dir>/.locks/capture.lock` con lock
  de archivo del sistema operativo en la capa de aplicación, sin Qt.
- Qué canal oficial enlaza Acerca de. Default: la página de releases del repositorio
  declarada en `[project.urls]` de `pyproject.toml`; si no se publica ninguna release
  con binario, la pantalla lo dice.
- Si el tamaño máximo de un paquete importable es configurable. Default: límites
  fijos (10.000 archivos, 20 GiB descomprimido, ratio 1:100) con error explicativo.
- Si la exportación de texto/Markdown desde el detalle incluye citas de M4. Default:
  incluye solo transcript y metadatos de la reunión; las respuestas se exportan en el
  paquete `.here.zip`.

Estos defaults no requieren consulta; se registran como ruling si se adoptan.

## Fuera de alcance explícito

- Actualización automática, detección de versiones sin opt-in, descarga de binarios.
- Cifrado propio de datos en reposo.
- Cuentas, nube, colaboración, sincronización, telemetría.
- Integraciones con calendario, correo, videollamadas u otras aplicaciones.
- Clientes para macOS, Linux, web o móvil.
- Firma Authenticode sin certificado de publicación configurado.
- Cambios en captura, overlay, chunking, catálogo, búsqueda o inteligencia que no sean
  necesarios para empaquetar, configurar o portar lo que ya existe.
