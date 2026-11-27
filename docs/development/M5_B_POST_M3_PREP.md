# Preparación de M5 B después de M3 B

Mapa de coordinación del 27 de noviembre de 2026. El aporte adelantado de M5 B
incluye paquetes ZIP, recuperación y borrado local, adaptadores CLI y Qt y
Markdown de detalle. La integración parte del merge aceptado de M3 B. Claims y
answers requieren el traspaso real de M4 B. Aplican los contratos del
[SPEC](M5_PORTABILITY_SPEC.md), el [PLAN](M5_PORTABILITY_PLAN.md) y
[CLAUDE §11](../../CLAUDE.md).

## Bases y secuencia de integración

| Referencia | SHA y uso |
| --- | --- |
| Checkpoint M5 B | `5f6324f4e58d760c0ce8b032769d6f7a4babd41e` en `codex/m5-portability-adapter-integration`. Árbol `bff3fb57b51359198b782baabeb56cb38826462b`. |
| Consumidor provisional usado por M5 | `ea61d6a8d77bd670492489083efcf99ba8710fc2`, primer padre de `5705a5d4e70ea4ae58d852bd3084eee6934dcf0f`. El segundo padre es el core M5 verificado `41150f5c8658f5d274166acf2b6d766acff6f9f8`. |
| Candidato de lectura de M3 B | `063d974f66ac42da85a82474cfc900a1335e54f3` sobre `8d0b03d`. Árbol `9bae7572944dc495867906c00f64563a78ec508f`. El orquestador controla la aceptación y el SHA de destino. |
| Ancestro Git común | `0c160657ad357f8ab9593978e0e045149c29afe9`, anterior al cierre de M3 A. `ea61d6a` ya quedó fuera de la ascendencia del candidato rebasado. |

El aporte neto de M5 respecto de su consumidor provisional abarca 34 archivos.
El diff desde el ancestro común arrastra cambios de M3 y presenta 44 archivos
en común. La comparación por la base funcional identifica cuatro. Estas
consultas reproducen ambas comprobaciones y la equivalencia del árbol usado
para el foco Qt.

```powershell
$commonBase = git merge-base 5f6324f 063d974
$m3FromAncestor = @(git diff --name-only $commonBase 063d974)
@(git diff --name-only $commonBase 5f6324f |
  Compare-Object -ReferenceObject $m3FromAncestor -IncludeEqual -ExcludeDifferent).Count
git show -s --format='%H %P' 5705a5d
git diff --stat ea61d6a 5f6324f
$m3Paths = @(git diff --name-only ea61d6a 063d974)
git diff --name-only ea61d6a 5f6324f |
  Compare-Object -ReferenceObject $m3Paths -IncludeEqual -ExcludeDifferent
git log --reverse --first-parent --format='%h %s' 5705a5d..5f6324f
git rev-parse '28a5ba7^{tree}' '5f6324f^{tree}'
```

1. El orquestador fija el SHA del merge real de M3 B y abre una rama de
   integración sobre ese SHA. Recalcula la intersección usando el destino real.
2. Traslada el delta de `5705a5d` relativo a su primer padre, que contiene el
   core, exportación e inspección y la resolución original de `service.py`.
   Sobre ese resultado, aplica en orden los 30 commits de primer padre de
   `5705a5d..5f6324f`. Usa `git diff ea61d6a 5f6324f` como referencia del
   alcance. Conserva el checkpoint como árbol de comparación y deja fuera los
   cambios heredados de M3. Ajusta la documentación sobre su versión canónica.
3. El orquestador asigna cada archivo en común a un escritor. Un revisor
   independiente comprueba los contratos en el nuevo head con las evidencias
   RED y GREEN. El orquestador administra la ventana Windows y la integración.

| Archivo en común | Resolución y dueño |
| --- | --- |
| `src/memory/service.py` | Orquestador o titular exclusivo designado. Conservar el preflight R50/R76 de M3 al tomar el snapshot bajo el lock. El helper `_catalog_snapshot_locked` de M5 llama hoy a `_deletion_registry` sin `preflight_under_lock=True`. Revisar las lecturas R50 y la captura/revalidación de exportación, y auditar el preflight de importación por separado según su contrato de escritor. Usar la transacción existente, con supresión R42 y aliases exactos antes de exposición. Preservar R51 para borrado, R52 para el orden de propietarios nativo/catálogo, R62 para reintentos nativos y la prueba local del grupo portable. La publicación ZIP y Markdown revalida bajo lock, termina la transacción de lectura antes del rename y preserva el destino ante carrera. |
| `src/output/paths.py` | Orquestador o titular exclusivo designado. Reutilizar el `validate_artifact_filename` de M3 con sus comprobaciones de archivo regular, identidad y redirecciones. M5 usa ese símbolo para sus basenames. |
| `STATUS.md` y `docs/development/DELIVERY.md` | Orquestador. Conservar el estado actual y los rulings R68, R74, R75, R76 y R77. Incorporar evidencia nueva sobre los documentos canónicos. |

M3 conserva el contrato de citas, búsqueda v2 y paridad stock/native, sus dos
encoders, QtMultimedia, los reintentos y el checkout exacto del workflow.
Los módulos de paquete y los adaptadores se trasladan con su comportamiento
M5. Validar el preflight que bloquea nuevas inserciones si hay conflicto de lote
y el orden del manifiesto en los informes. Preservar reuniones ya confirmadas
si falla una publicación posterior. En CLI el conflicto da 3 y un error de
integridad o E/S da 4 con precedencia. Qt conserva selección, progreso y
cancelación en los trabajos existentes.

## Pruebas y evidencia

Esta preparación añade un mapa de fuentes y recibos. La revisión estática es
nivel 1 y las corridas previas conservan su SHA y nivel. Los siguientes XML de
nivel 2 existen bajo `C:/Users/savar/AppData/Local/Temp`, salvo indicación
explícita. El recuento y el tiempo son los del JUnit.

| SHA ejecutado | Resultado | Recibo y alcance |
| --- | --- | --- |
| `41150f5` | 4/4 en 14,903 s y 89/89 en 130,937 s | `m5-portability-import-service/here/.superpowers/m5-ir-green-41150f5-65afc7c2.xml` y `m5-ir-integrated-green-41150f5-69a1a9e8.xml`. Recuperación, borrado y lote del core. RED causal previo `m5-ir-red-0986b7d-068933d7.xml`. |
| `d35f31a`, `0a3f0ea` | 46/46 y 22/22 | `m5-import-partial-report-green-d35f31a-20261127.xml` y `m5-import-partial-order-green-0a3f0ea-20261127.xml`. Informe parcial, orden y adaptadores. |
| `1498f4c` | 140/140 en 122,871 s | `m5-portability-adapters-1498f4c-20261127.xml`. Core, CLI, Qt y Markdown. |
| `b04d30f` | 142/142 en 107,946 s | `m5-portability-verified-b04d30f-20261127.xml`. Añade cobertura de advertencias de limpieza. |
| `56166d8` | 12/12 en 5,837 s | `m5-markdown-typed-56166d8-20261127.xml`. Contrato tipado y foco Markdown. |
| `28a5ba7` | 1/1 en 3,642 s | `m5-portability-qt-core-28a5ba7-20261127.xml`. Árbol íntegro idéntico a `5f6324f`. Servicio y Qt reales con diálogo simulado. Importa sin audio en un catálogo configurado distinto y exporta Markdown desde el detalle. |

En M3 B, el recibo R76 `m3-b-r76-product-gates-063d974` registró p95 de
395,38 ms y heartbeat máximo de 13,2683 ms en ocho consultas, nivel 2. El seek
real histórico de `912f63f` es nivel 3 en Windows Media Foundation, con
2,913 ms de error absoluto máximo y 36,245 ms al sumar incertidumbre. El nuevo
head integrado necesita su propia corrida. El fixture sintético de seek ya
está bajo `C:/Users/savar/.codex/evidence/here/m3-b-synthetic-seek-fixture-20261127`
con `receipt.json`, `stage-marker-package.ps1`, marcas a 0, 1200, 2800 y
4700 ms y hash de `Probe.dll`.

| Gate pendiente | Pruebas y fixture | Dueño y condición de aceptación |
| --- | --- | --- |
| Regresión M5 sin claims | Reusar `tests/portability_fixtures.py`, `tests/test_portability_*.py`, `tests/test_portable_json_budget.py`, `tests/test_cli_meetings_{export,import}.py` y `tests/test_ui_library_{export,import,markdown,portable_integration}.py`. Mantener procesos de crash/rebuild reales y sus escenarios de staging y rutas largas. Conservar el destino ante carrera/cancelación, la evidencia local ante pérdida de catálogo, la privacidad y los límites R68. | M5 B propone el selector y el orquestador habilita la corrida de nivel 2. Anotar cobertura efectiva de NTFS, espacio insuficiente, enlaces/redirecciones y publicación a otro volumen si es accesible. Si falta la condición del entorno, registrar la prueba como pendiente. |
| Importación y R45/R76 | Componer dos raíces temporales y texto único en revisión activa e histórica. Capturar antes de importar un cursor de búsqueda y uno de listado. Comprobar que una importación con alta real cambia el corpus, que el cursor de búsqueda devuelve `corpus_changed` y que el listado conserva su keyset. Obtener cursores nuevos y probar que la reimportación idéntica no cambia el corpus ni invalida esos cursores. Reusar `test_memory_search_generation_scale.py`, `test_memory_search_native_active_v2.py`, `test_memory_search_row_representation.py`, `test_m3_library_cli.py` y las continuaciones/cancelaciones Qt de `test_ui_library.py`. | M3 entrega sus fuentes y DLL aceptadas. M5 B añade los cruces faltantes. Comparar paridad stock/native y cursores v2/legado. Medir p95 < 500 ms y heartbeat máximo < 50 ms en el head compuesto. |
| Citas y WAV legado importado | RED nuevo a partir de `tests/test_memory_citation.py` y `tests/test_ui_playback_navigation.py::test_navigation_routes_resolve_and_select_the_same_evidence`. Usar `make_session` y `write_marked_wav` de `tests/memory_fixtures.py`, con `session.json` v1 en el origen, `source_file` no estándar y dos revisiones con enlaces exactos. Exportar con audio, importar en otra raíz y retirar el origen y el ZIP sintéticos. Reabrir y verificar que los destinos carecen del `session.json` del origen y que cada cita activa e histórica resuelve el WAV canónico, su digest y el tiempo exacto. Incluir no-audio, tiempo desconocido, WAV alterado, grupo sin completar, enlace a otra revisión y supresión antes de cualquier lectura de WAV. | Tras el merge de M3 B, RED y escritor de la adaptación distintos. Nivel 2 para resolver y navegar desde una búsqueda real al detalle y a la cita. Nivel 3 para comprobar el seek del WAV importado en el paquete Windows exacto con las marcas detectadas por la sonda. |
| Composición y cambio de raíz | Añadir al test del catálogo configurado la operación real de M5 A con raíces temporales anterior/nueva, sesiones nativas registradas explícitamente e IDs con canarios de contenido distintos. Probar éxito, trabajo activo, cancelación y fallo del cambio. Verificar en CLI/Qt qué raíz usa la fachada y qué árbol queda activo en cada resultado. Mantener el preview sintético. | M5 A provee el cambio real de raíz, quiescencia, copia verificada y rollback. M5 B prueba exportación/importación/Markdown sobre esa composición. |

El RED legado verifica una diferencia concreta de autoridad. El resolver
`resolve_citation → _citation_audio` comprueba hoy `source_file` como rol
nativo. En un import ese campo es etiqueta y el WAV es un artefacto administrado
`audio_<uuidhex>.wav`. La adaptación futura debe verificar el vínculo exacto
revisión/asset/ordinal y el receipt del grupo local, con prueba de completitud,
roles y hash/tamaño. Auditar `_portable_receipt_for_asset` junto con
`read_portable_completion` y `validate_portable_group_members`. El ZIP y la
ruta del origen pueden desaparecer tras el commit. R42 antecede la lectura de
audio. El orquestador asigna `service.py` y `artifacts.py` cuando el RED confirme
el cruce. Una regresión de seek se compara con el presupuesto de 250 ms incluido
el error de medición.

Para los focos y la suite completa usar `pwsh`, `QT_QPA_PLATFORM=offscreen`,
`OPENAI_API_KEY` sintética, `HERE_ENV_FILE` inexistente, `HERE_DATA_DIR` y
`TRANSCRIPTIONS_DIR` temporales, y el runner Windows de la rama exacta. Los
gates finales de M5 C incluyen CI Windows, wheel/paquete del checkout exacto,
instalación limpia y actualización soportada con evidencia propia. El ruling 5
omite la prueba de dos horas.

## Traspaso de claims y answers

El checkpoint exporta `answer_ids=()` y `omitted=()`. El import actual rechaza
inventarios con answers. El inicio del adaptador requiere el contrato aceptado
de M4 B con migración/esquema, IDs, relaciones, operaciones de lectura y
publicación, verificador de citas y semántica de historial y caché. M4 A entrega
su contrato de recuperación/verificación. M4 B conserva la autoridad de
persistencia. M5 B enlaza de forma exclusiva el adaptador portable tras ese
traspaso y el orquestador asigna los archivos compartidos.

| Área | Contrato que debe entregar M4 B y aceptación futura |
| --- | --- |
| Exportación | Proyección detached bajo R50 con campos permitidos y verificación de IDs, revisión/segmento, cita Unicode, offsets y tiempos originales. Incluir el answer completo sólo si todas sus reuniones citadas están seleccionadas. Registrar cada exclusión en `omitted`. Aplicar R42 antes de exponer datos derivados. |
| Importación y recuperación | Validar las relaciones y citas con los contratos aceptados y publicar cada answer sólo cuando todas sus reuniones sean durables, nuevas o idénticas. Acordar con M4 B el límite de transacción, prueba local, idempotencia y reconstrucción para cancelación, fallo parcial o commit incierto. Verificar referencias cruzadas incluso si el manifest y hashes del ZIP se recalcularon. |
| Borrado | Usar el purger R51 de SQL de confianza en la misma conexión y transacción final. Borrar todos los answers que citen la reunión y sus claims/citas, y auditar exposición por caché e historial. Ante error, rollback de la transacción y borrado pendiente. Mantener la supresión R42 durante los reintentos. |
| Fixture y límites | Tres reuniones sintéticas, dos seleccionadas. Un answer cita ambas con evidencia Unicode exacta, revisión activa e histórica y un segmento sin tiempo. Otro cita la tercera y queda en `omitted`. Mutaciones de vínculo, revisión, cita y offsets, duplicado, reimportación, pérdida de catálogo y canario de privacidad. Reusar los fixtures ilustrativos de M4 A en `0c35b6b` y `a2ce1e1` donde apliquen. El fixture persistido se deriva del contrato entregado. Repetir R68 con manifest de hasta 8 MiB y JSON total de hasta 32 MiB incluido el manifest, y respetar 10.000 entradas, 20 GiB declarados y ratio 100:1. |

El adelanto §11 disponible en este checkpoint queda cubierto por la
transferencia del core y este mapa reproducible. La composición de fixtures
sintéticos puede prepararse con una asignación de pruebas separada. Las
próximas ediciones de producto dependen del traspaso de M3 B, M4 B o M5 A y
de su archivo exclusivo. La próxima ejecución requiere la ventana Windows
del orquestador.
