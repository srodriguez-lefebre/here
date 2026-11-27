# Estado de here

Actualizado: 2026-11-27. Leé CLAUDE.md primero; después este archivo y
`docs/MASTER_PLAN.md`. Historial y rulings: `docs/development/DELIVERY.md`.

## Reanudación — 2026-11-27

Se reactivaron los chats de M3 A y M5 B con GPT‑6.2 Sol. M2 sigue cerrado en
0.3.0. Copilot sigue omitido por ruling 64 y dos horas, **omitida por ruling 5**.

**M3 A mergeado.** [PR #28](https://github.com/srodriguez-lefebre/here/pull/28)
entró a `main` en `a1a83629cdfab64dd249f877cdd443e7f7d96b08` con el head
revisado `3a9cc11`. Windows CI y el paquete pasaron en ese head, nivel 2.
Se conservan los cursores v2 y los gates de rendimiento. El
[PR documental #29](https://github.com/srodriguez-lefebre/here/pull/29)
fijó el traspaso a M3 B en `8d0b03d` con revisión independiente, CI y
paquete Windows verdes sobre `d57a43d`. Nivel 2. M3 B rebasó sus 34
commits sobre ese merge y verifica el candidato `063d974`.

### Frentes activos del ruling 75

| Frente | Rama | Worktree bajo `C:/Users/savar/.codex/worktrees/` | Base y responsable |
| --- | --- | --- | --- |
| M3 A, extensión y equivalencia v2, chat `01a1af04-e779-7363-be61-c840651b0b18` | `codex/m3-search-native-v2` | `m3-search-native-v2/here` | `ed571ed`, chat M3 A |
| M3 A, RED de distribución | `codex/m3-search-native-windows-red` | `m3-search-native-windows-red/here` | `6acd485`, `native_windows_distribution_red_sol` |
| M3 A, build Windows e integración del paquete | `codex/m3-search-native-windows-green` | `m3-search-native-windows-green/here` | `c8195de`, orquestador |
| M3 A, documentación del contrato R75 | `codex/m3-search-native-docs` | `m3-search-native-docs/here` | `fdf4ca6`, `m3_nov27_evidence_sol` |
| M3 A, candidato FTS integrado | `codex/m3-a-r75-integration` | `m3-a-r75-integration/here` | `4ba043e`, orquestador |
| M3 A, candidato R76 publicado en PR #28 | `codex/m3-a-r75-integration` → `codex/m3-a-product` remoto | `m3-a-r75-integration/here` | `3a9cc11`, orquestador |
| M3 A, revisión independiente integrada | HEAD separado | `m3-a-r75-exact-review/here` | `3a9cc11`, `m5_portability_package_io_review_sol` |
| M3 A, RED de rechazo de carga | `codex/m3-a-r75-load-refusal-red` | `m3-a-r75-load-refusal-red/here` | `a0806ed`, `native_windows_distribution_red_sol` |
| M3 A, perfil de etapas | HEAD separado | `m3-a-r75-phase-diagnostic/here` | `4ba043e`, orquestador. Conserva DLL y manifest generados para reproducir |
| M3 A, encoding ACTIVE, chat `01a1af04-e779-7363-be61-c840651b0b18` | `codex/m3-search-native-active-v2` | `m3-search-native-active-v2/here` | `44c1bc1`, chat M3 A. Lease devuelto |
| M3 A, RED de encoding ACTIVE | `codex/m3-search-native-active-v2-red` | `m3-search-native-active-v2-red/here` | `ad9763d`, subagente RED del chat M3 A |
| M3 A, RED de smoke con segmento sintético | `codex/m3-a-r76-active-smoke-red` | `m3-a-r76-active-smoke-red/here` | `91417c5`, `native_windows_distribution_red_sol` |
| M3 A, suite completa del candidato R76 | HEAD separado | `m3-a-r76-full-verification/here` | `7b04f76`, orquestador |
| M3 A, aislamiento de sonda FTS stock | `codex/m3-a-r76-stock-proof` | `m3-a-r76-stock-proof/here` | `c64b22d`, `native_windows_distribution_red_sol` |
| M3 A, evidencia R75/R76 | `codex/m3-a-r75-evidence` | `m3-a-r75-evidence/here` | `e101914`, `m3_nov27_evidence_sol` |
| M3 B, integración | `codex/m3-library` | `m3-library-group/here` | `063d974` sobre `8d0b03d`, orquestador |
| M3 B, suite completa integrada | HEAD separado | `m3-b-r76-full-verification/here` | `063d974`, orquestador y ventana Windows local |
| M3 B, SPEC y PLAN de traspaso tras R75 | `codex/m3-b-native-handoff-contracts` | `m3-b-native-handoff-contracts/here` | `ad815db`, `m3_b_native_handoff_docs_sol` |
| M3 B, PR documental #29 mergeado | `codex/m3-b-accepted-native-handoff` | `m3-b-accepted-native-handoff/here` | `d57a43d` → `8d0b03d`, `m3_b_native_handoff_docs_sol` y orquestador |
| M4 A, preparación de SPEC y PLAN | `codex/m4-grounded-candidate-handoff` | `m4-grounded-candidate-handoff/here` | `2cf53f5`, `m3_b_native_handoff_docs_sol`. Vincular al merge final de B |
| M4 A, fixtures de respuestas sintéticas | `codex/m4-grounded-verifier-fixtures` | `m4-grounded-verifier-fixtures/here` | `0c35b6b`, `m3_b_native_handoff_docs_sol`. Transferir al RED después del cierre de M3 |
| M5 B, adelanto de portabilidad, chat `01a1af04-b158-75a2-8933-98783880f3e8` | `codex/m5-portability-adapter-integration` | `m5-portability-adapter-integration/here` | `5f6324f`, chat M5 B. Claims esperan M4 B |

El orquestador administra la ventana Windows exclusiva para candidato,
benchmarks y builds pesados. RED de distribución **nivel 2** en `6acd485`:
dos errores de setup esperados al verificar el ZIP real de la corrida
`46015025767`, que carece de la DLL nueva. Recibo `package-red.xml` en
`C:/Users/savar/.codex/evidence/here/m3-native-windows-distribution-20261127/`.
El RED de dispatch del chat falló por ausencia del loader. El recorrido corregido
en `ed571ed` pasó 160 controles con una omisión prevista, nivel 2. El smoke
local del módulo en un paquete de prueba verificó dos batches y bytes v2 de 257 filas
dentro de R50 con carga SQL vedada. Recibos en
`C:/Users/savar/.codex/evidence/here/m3-search-native-v2-equivalence-01/`.
`c8195de` pasó revisión independiente estática del empaque después de
aislar el entorno de Zig. Nivel 2 en `build-2ee8ac6.json`: la extensión
compiló con el entorno controlado aun con rutas Zig y Clang inválidas
heredadas por el proceso padre. La revisión integrada pidió cubrir el
rechazo de carga de SQLite y conservar la ruta canónica. RED `7f78f1d`:
un fallo esperado y dos pases. GREEN: tres pases con el fixture de orden
físico corregido. La revisión independiente aprobó `4ba043e`.

**El p95 de nivel 2 falló en `4ba043e`.** `r75-public-p95-4ba043e` completó
5+40 sobre 1.000 reuniones y 100.000 segmentos y dio 530,14 ms p95, con
el límite de 500 ms intacto. El perfil 2+8 `phases-4ba043e.json`
observó ocho cargas y recorridos FTS nativos completos. Medianas
246,94 ms en ACTIVE, 116,78 ms en FTS y 2,78 ms en carga. El chat M3 A
retomó para extender el encoding v2 a ACTIVE por ruling 76.
En esa etapa el PR #28 seguía abierto en `0c16065` y su descripción
conservó la evidencia de diagnóstico.
El [paquete diagnóstico Windows](https://github.com/srodriguez-lefebre/here/actions/runs/46028557240)
de `4ba043e` quedó verde, nivel 2. Se verificaron seis hashes, 88 controles
sin fallos ni omisiones y seis smokes de activación nativa en GUI y CLI,
desde el bundle y tras instalar y reinstalar. Recibo
`package-46028557240-receipt.json` en la carpeta de evidencia R75.

**Los dos gates locales de nivel 2 pasaron en `7b04f76`.** El foco R76
de `44c1bc1` pasó 175 controles con una omisión prevista. El candidato
integrado pasó `r76-product-gates-7b04f76` con 397,40 ms p95 en 5+40 y
9,462 ms de máximo en el heartbeat de ocho consultas, 988 ticks y
workers drenados. El smoke ahora usa un segmento sintético de un
catálogo temporal y exige ese resultado y la activación de ambos
encoders. El RED `91417c5` falló por cero hits y el foco GREEN pasó.
El full local con DLL tuvo 1.699 pases, un fallo y 23 omisiones en
`r76-full-7b04f76`, nivel 2. El fallo fue la sonda stock, que esperaba su
traza SQL con el encoder nativo activo. El foco corregido de `2286bd7`
pasó 13 controles stock y nativos. En el head `3a9cc11`, cinco controles
de carga y smoke pasaron. La anotación del lector verificable se acotó a
`BufferedReader` y la comparación mypy de los módulos tocados mantuvo
los cuatro diagnósticos heredados sin altas. `c7695c7` tuvo
[CI verde](https://github.com/srodriguez-lefebre/here/actions/runs/46031147690)
y [paquete verde](https://github.com/srodriguez-lefebre/here/actions/runs/46031147574).
La revisión aprobó `3a9cc11` exacto. [PR #28](https://github.com/srodriguez-lefebre/here/pull/28)
lo publica con [CI Windows](https://github.com/srodriguez-lefebre/here/actions/runs/46032934133)
y [workflow de paquete](https://github.com/srodriguez-lefebre/here/actions/runs/46032934126)
sobre ese SHA. **Paquete y wheel verificados a nivel 2.** La corrida
`46032934126` pasó 110 controles nativos y seis smokes congelados.
Se verificaron los seis hashes externos, las 736 entradas del ZIP y
la procedencia limpia del head. El wheel de ese artefacto se instaló en
un entorno no editable separado, encontró el hit sintético por la ruta
canónica y el lote `r76-wheel-3a9cc11` pasó cinco. Recibos
`package-46032934126-verified.json`, `r76-wheel-search-3a9cc11.json` y
`r76-wheel-3a9cc11.xml` bajo evidencia R75. CI Windows `46032934133`
pasó 1.619 casos y omitió 104. El merge protegido usó
`--merge --admin --match-head-commit` sobre `3a9cc11`. M3 B aún debe
integrar el reproductor y empaquetado y cerrar la versión 0.4.0.
El rebase de B resolvió conflictos en `src/memory/service.py`,
`packaging/windows.spec` y `.github/workflows/windows-package.yml`,
conservando R50, ambas rutas nativas y QtMultimedia. La rama de respaldo
`codex/m3-library-before-native-rebase` conserva `b36162a`. La revisión
independiente aprobó `063d974`. Los gates integrados pasaron a nivel 2 en
`m3-b-r76-product-gates-063d974`, con p95 de 395,38 ms y heartbeat de
13,2683 ms máximo en ocho consultas. La suite local con DLL
`m3-b-r76-full-063d974` sigue en ejecución en su worktree separado.
El [paquete integrado diagnóstico](https://github.com/srodriguez-lefebre/here/actions/runs/46036939106)
se construye desde `063d974` en `codex/m3-b-r76-verification`. El seek
real se mide cuando el paquete verificado esté disponible y la máquina
local quede libre. El bump a 0.4.0 queda para después de esos criterios.
El fixture de seek ya generado y la identidad de `Probe.dll` se verificaron
a nivel 2 en `C:/Users/savar/.codex/evidence/here/m3-b-synthetic-seek-fixture-20261127/receipt.json`.
Conserva los siete insumos sintéticos de la toma anterior para preparar una
raíz nueva con el paquete integrado de B. El mismo directorio contiene
`stage-marker-package.ps1` para verificar y preparar esa raíz antes del
smoke, con el recibo del paquete exacto y los tests de QtMultimedia.
Los workflows fijan el checkout del head exacto por ruling 77. El PLAN
preparado de M3 B conserva la ruta nativa cuando incorpora QtMultimedia.
El orquestador lo vinculará al merge de A en el PR de documentos.

- M3 A recuperó `2710/here` en `e7f84b0`. Nivel 2 diagnóstico de FTS: la
  mediana de la fase bajó de 404,98 a 320,11 ms. Las ocho muestras conservaron
  el mismo SHA sobre 98.901 filas y 387 frames. El RED `ae82ec8` produjo siete
  pases, un fallo esperado de la ruta y un fallo del arnés corregido en
  `f04a5f3` y quedó un fallo esperado en la corrida corregida. `7e3bd56`
  pasó 133 controles y omitió dos. Su 5+40 público dio 966,13 ms p95.
  [PR #28](https://github.com/srodriguez-lefebre/here/pull/28) permanece abierto
  por el p95 de 500 ms. Nivel 2: `7e3bd56` pasó el heartbeat con ocho
  consultas, un máximo de 18,65 ms y tres sellos iguales. El prototipo R66
  en `cf80ea2`, `codex/m3-native-current-probe`, dio 570,24 ms p95 en 5+40
  con CPython 3.13.15 y SQLite 3.53.1. Sus cursores v2 y los empates de
  valores numéricos siguen abiertos. El chat M3 A entregó `7087d13`
  en `codex/m3-native-fts-order-probe`, worktree del mismo nombre. Nivel 2
  diagnóstico R66: pasó 25 controles de orden y 126 compartidos. El par
  5+40 dio 437,93 ms p95 con el agregado nativo canónico y 333,78 ms con
  el ordenado. Hubo 50 intentos y cero fallbacks. El chat dejó el diseño
  condicional `33192ac` en el mismo worktree. El PR #28 sigue abierto con
  la evidencia y conserva el head `0c16065`.
- M3 B reprodujo en `a31daca` la selección de un WAV normalizado vacío con
  una fuente cruda válida. Nivel 2 focalizado: 1 fallo y 6 pases. `b2dc640`
  pasó 35 casos de reintento y geometría. La revisión independiente detectó
  que también inhabilita un `audio.wav` v1 declarado como fuente de captura.
  El RED `e7f9807` en `m3-library-retry-declared-source-red/here` confirmó
  el fallo de compatibilidad y el rechazo de audio sin declarar (1 fallo y
  1 pase, nivel 2). `m3_retry_audio_resume_green` entregó `5edb571` en
  `codex/m3-library-retry-declared-source-green`, worktree
  `m3-library-retry-declared-source-green/here`. La re-review de
  `m3_retry_audio_resume_review` señaló archivos distintos con nombres que
  Windows `Path` iguala. `m3_retry_declared_source_red` cubrió ese caso y los
  aliases del mismo archivo en `d12d483`, rama
  `codex/m3-library-retry-file-identity-red`, worktree
  `m3-library-retry-file-identity-red/here`. RED nivel 2: un fallo esperado
  y un pase. GREEN `cffa2de` en `codex/m3-library-retry-file-identity-green`,
  worktree `m3-library-retry-file-identity-green/here`, pasó 247 casos.
  `m3_retry_audio_resume_review` aprobó ese SHA. El árbol integrado en
  `912f63f`, `codex/m3-library`, es idéntico y tiene un
  [paquete diagnóstico Windows exitoso](https://github.com/srodriguez-lefebre/here/actions/runs/46015025767).
  Se verificaron seis hashes, 772 archivos y el inventario incrustado.
- Sonda de seek M3 B: build local y calibración PCM sintética pasaron, nivel 2.
  El artefacto Windows de `f894923`, corrida `44781729845`, coincidió con el
  checksum del ZIP y las 772 entradas del inventario. El CLI del bundle ingirió
  el WAV sintético con hash `eac58e90b86607135ad0a30562a36f68b2914b95dc81e77a3553b7b664b4cabc`.
  La GUI cargó la marca cero. La sonda falló antes de armar con `0x88890021`.
  `1dfa531` corrigió las flags de WASAPI, compiló, pasó la calibración y
  armó la captura. Antes de reproducir, devolvió una discontinuidad o un
  timestamp inválido. Hubo cero mediciones. `m3_audio_capture_integrity_red_sol`
  dejó `7af6935` en `m3-audio-capture-quality-red/here`. La sonda `8620e8e`
  pasó calibración. La toma diagnóstica `7472090` sobre `912f63f` observó
  601 paquetes audibles, todos con posición de dispositivo cero, y 600
  pares QPC con diferencia exacta según los frames. El RED `cc0c0fd` está
  en `m3-audio-capture-boundary-red/here` y falló al faltar los helpers de
  reloj. `62f0404`, `codex/m3-audio-qpc-green`, `m3-audio-qpc-green/here`,
  verificó la continuidad QPC y el límite entre paquetes. Build,
  calibración y revisión independiente aprobados. Nivel 3 para el seek
  sobre `912f63f`: `m3-b-seek-912f63f-20261127` pasó las cuatro marcas en
  reproducción real de Media Foundation. Máximo |error| 2,913 ms y
  máximo |error| más incertidumbre 36,245 ms frente a 250 ms. Recibos en
  `C:/Users/savar/.codex/evidence/here/m3-b-seek-912f63f-20261127/`.
  M3 B espera rebase y suite en su head definitivo después de M3 A.
  `a82c14e`, `codex/m3-library-playback-accessibility`, pasó tres controles
  de teclado, accesibilidad y contraste. Revisión independiente aprobada.
  Root integró los tests en `b36162a`, sin cambios a la fuente del paquete.
- M5 B nivel 2 RED: `b66d15a` dio 16 fallos esperados en CLI. `7d7c04d` dio
  14 fallos esperados en Qt. La base común `bbc1717` en
  `codex/m5-portability-adapter-integration`, `m5-portability-adapter-integration/here`,
  integra el servicio `41150f5` y las pruebas. El chat de M5 B coordina dos
  escritores desde las ramas `codex/m5-portability-adapter-cli-green` y
  `codex/m5-portability-adapter-qt-green`, con worktrees del mismo nombre
  bajo `C:/Users/savar/.codex/worktrees/`. Root es dueño de la integración.
  Nivel 2 focalizado: CLI `ee4506c` pasó 16 casos y fue aprobado. Qt `e0fee3e`
  pasó 14 casos al corregir la firma del doble de PySide6, luego de 13 fallos
  de arnés en `1523399` y obtuvo revisión independiente. El RED adicional
  de acceso ZIP reprodujo dos errores de clasificación en `e1b55e9` y el del
  parser de ZIP en `e27ec20`. `9dd0422`, rama `codex/m5-portability-adapter-zip-parser-io-green`,
  pasó 61 casos CLI y ZIP hostil y obtuvo review independiente. Root integró
  los adaptadores e I/O en `769bc28`. El chat prepara Markdown desde el
  detalle. `974f39a`, `codex/m5-portability-markdown-core-red`, reprodujo
  10 fallos esperados en Qt y la fachada de exportación. El renderizador
  `412bb49` pasó revisión estática. Fachada `ca4b652`, guard de destino
  `ad02a06` y Qt `3713784` se integraron en `1498f4c`, rama
  `codex/m5-portability-adapter-integration`. Pasaron 140 pruebas Windows.
  La revisión detectó un aviso perdido si falla la limpieza después de
  guardar. `9eee8f9` reprodujo los dos fallos esperados. El arreglo pasó
  142 casos en `b04d30f`; el ajuste de tipo de test `56166d8` pasó el foco
  de 12 casos y fue aprobado. `28a5ba7` pasó la integración Qt con catálogo
  y servicios reales en raíces temporales, con picker simulado. El grupo
  local está en `5f6324f` (árbol idéntico a `28a5ba7`).
  `d35f31a` pasó 46 casos de reporte parcial y recuperación. El guard
  del orden de manifiesto `0a3f0ea` pasó 22, obtuvo revisión independiente
  y está integrado en `eb63749` con árbol idéntico. Todas las ramas usan
  worktrees del mismo nombre. Claims espera M4 B.
- La preparación de M4 A conserva `99f1ec8`. `m4_grounded_fixture_sol`
  entregó el corpus de 50 reuniones y 2.000 coincidencias en `a2ce1e1`,
  rama `codex/m4-grounded-fixture`, worktree `m4-grounded-fixture/here`.
  El handoff documental `02ca1bb` está en `codex/m4-grounded-candidate-handoff`,
  con worktree del mismo nombre. La implementación espera el cierre de M3
  y el PR de contratos.

## Pausa solicitada — 2026-11-20

Pausa general solicitada por el usuario. Ambos chats y los subagentes dejaron
de trabajar. Se verificó que no quedaron procesos locales de runtime. Los
chats y agentes activos en esta reanudación usaron GPT‑6.2 Sol. M2 sigue cerrado
en 0.3.0. Copilot sigue omitido por ruling 64 y dos horas queda **omitida por
ruling 5**.

| Frente | Rama y worktree bajo `C:/Users/savar/.codex/worktrees/` | Commit y próximo paso al reanudar |
| --- | --- | --- |
| M3 A, chat `01a1af04-e779-7363-be61-c840651b0b18` | `codex/m3-search`, `2710/here` | `e7f84b0`. Perfilar ordenamiento FTS con el guard R50 y luego medir 5+40 en el candidato exacto. |
| M3 A, heartbeat | `codex/m3-heartbeat-harness`, `m3-heartbeat-harness/here` | `45f35ae`. Diagnosticar el gap de 55,349 ms en la octava búsqueda. |
| M3 A, controles R50 | `codex/m3-r50-preflight-controls`, `m3-r50-preflight-controls/here` | `54c5458`. Reaplicar los negativos y el control de fingerprint al próximo candidato. |
| M3 B, integración | `codex/m3-library`, `m3-library-group/here` | `ea61d6a`. Correr primero el RED de selección de fuente y seguir con la corrección y suite del candidato exacto. |
| M3 B, RED de WAV normalizado vacío con fuente cruda válida | `codex/m3-library-retry-source-red`, `m3-library-retry-source-red/here` | `a31daca`. Corrida pendiente. |
| M3 B, fixture de seek y sonda de audio | `codex/m3-library-marker-fixture`, `m3-library-marker-fixture/here` | `42d01db`. Compilar y calibrar la sonda del proceso aislado antes de medir reproducción real. El fixture integrable es `3d96608`. |
| M4 A, preparación documental | `codex/m4-grounded-preparation`, `m4-grounded-preparation/here` | `99f1ec8`. Esperar cierre de M3 y contrato de recuperación. |
| M5 B, chat `01a1af04-b158-75a2-8933-98783880f3e8` | `codex/m5-portability-import-service`, `m5-portability-import-service/here` | `41150f5`. Base del servicio para integrar adaptadores. |
| M5 B, RED CLI | `codex/m5-portability-adapter-cli-red`, `m5-portability-adapter-cli-red/here` | `fcfc973`. Completar RED de importación y verificar ambos contra el SPEC. |
| M5 B, RED Qt | `codex/m5-portability-adapter-qt-red`, `m5-portability-adapter-qt-red/here` | `ea61d6a`. Preparar las pruebas nuevas de exportación e importación. |

**Nivel 2.** M3 A verificó 67 casos del guard y un omitido. Los dos negativos
R50 fallaron en la base y pasaron con el guard. El heartbeat completó ocho
búsquedas con tres sellos de corpus idénticos y falló el presupuesto de 50 ms.
La mejor muestra diagnóstica de búsqueda fue 605,65 ms. El p95 público de
1.431,6706 ms sigue como último gate medido y [PR #28](https://github.com/srodriguez-lefebre/here/pull/28)
permanece abierto. Los recibos del chat están en `.superpowers/sdd/M3_SEARCH_PLAN`.

**Nivel 2.** M3 B conserva los 465 pases y el paquete Windows diagnóstico
`44781729845`, con multimedia y smoke de instalación, en `f894923`. El head
local posterior `ea61d6a` queda pendiente de verificación integrada. M5 B pasó
los cuatro casos focales de recuperación y los 89 de continuidad en `41150f5`.
Sus recibos están en `.superpowers` del worktree del servicio. Claims sigue
pendiente de M4 B. Quedan pendientes la reproducción real de M3 y la
certificación de instalación y upgrade de M5, nivel 3.

## Reanudación — 2026-11-20

M2 cerrado en 0.3.0. El usuario reanudó M3–M5 y eligió GPT‑6.2 Sol para
los frentes activos. El orquestador conserva las ventanas de ejecución en serie.
Copilot sigue omitido por ruling 64 y dos horas queda **omitida por ruling 5**.

| Frente | Rama y worktree bajo `C:/Users/savar/.codex/worktrees/` | Head o candidato y responsable |
| --- | --- | --- |
| M3 A, chat `01a1af04-e779-7363-be61-c840651b0b18` | `codex/m3-search`, `2710/here` | `1f71d08`, chat M3 A reactivado con GPT‑6.2 Sol. Tiene el mismo árbol que `0c16065`. |
| M3 A, PR #28 | `codex/m3-a-product`, `m3-a-product/here` | `0c16065`, orquestador. Mantener abierto por p95. |
| M3 B, integración | `codex/m3-library`, `m3-library-group/here` | `ea61d6a`, orquestador. Review por `m3_b_integration_review_sol` y `m3_multimedia_packaging_review_sol`. Base verificada `f894923`. |
| M3 B, citas | `codex/m3-library-resolver-green`, `m3-library-resolver-green/here` | GREEN `7323613`, orquestador, 28 pases. RED de roles `4cabf16`, `m3_citation_roles_red_sol`. |
| M3 B, reproducción | `codex/m3-library-playback-green`, `m3-library-playback-green/here`; RED de revisión en `codex/m3-library-playback-edge-red`, `m3-library-playback-edge-red/here` | GREEN `2745ec7`, `m3_b_playback_green_sol`. RED `6cb1d72`, `m3_b_playback_edge_red_sol`. Re-review pendiente. |
| M3 B, navegación | `codex/m3-library-navigation-green`, `m3-library-navigation-green/here`; correcciones `codex/m3-library-nav-review-green`, `m3-library-nav-review-green/here` | `820c38b` pasó 30. Review reproductor y navegación en worktree aislado. `e9a3e90`, `m3_b_navigation_green_sol`, incorpora correcciones con runtime integrado pendiente. |
| M3 B, CLI | `codex/m3-library-cli-green`, `m3-library-cli-green/here`; scope `codex/m3-library-scope-green`, `m3-library-scope-green/here` | `a5c71d3` de `m3_b_cli_green_sol`; `30a8ea3` de `m3_b_scope_green_sol`; diagnóstico de playback `beebd8a` pasó cuatro casos. |
| M3 B, paquete y fixture WAV | `codex/m3-library-multimedia-green`, `m3-library-multimedia-green/here`; `codex/m3-library-marker-fixture`, `m3-library-marker-fixture/here` | `1fb797c`, `m3_multimedia_packaging_green_sol`; fixture sintético `3d96608`, `m3_marker_fixture_preparation_sol`. |
| M5 B, chat `01a1af04-b158-75a2-8933-98783880f3e8` (antes M2 B) | `codex/m5-portability-import-service`, `m5-portability-import-service/here` | `41150f5`, chat reactivado con GPT‑6.2 Sol. El orquestador controla leases, contratos y merges. |
| M5 B, arnés Windows | `codex/m5-portability-recovery-harness`, `m5-portability-recovery-harness/here` | `deccde1`, `m5_recovery_harness_sol`. Corrección de staging `d0449fc` incluida en `dcb3660`. |
| M5 B, presupuesto JSON | `codex/m5-portability-json-cap-red`, `m5-portability-json-cap-red/here` y `codex/m5-portability-json-cap-green`, `m5-portability-json-cap-green/here` | RED `05ba0ee`, `m5_json_cap_red_sol`. GREEN `fc36f91` revisado por `citation_json_independent_review_sol` e incluido en `dcb3660`. |
| M4 A, preparación del PLAN | `codex/m4-grounded-preparation`, `m4-grounded-preparation/here` | `99f1ec8`, `m4_next_plan_preparation_sol`. Pendientes cierre de M3 y contrato de recuperación para múltiples reuniones. |

[PR #28](https://github.com/srodriguez-lefebre/here/pull/28). Nivel 2 para CI
Windows `43477815724` y paquete `43477815761`, ambos verdes en `0c16065`. Se
restauró el checkout del chat. Parte de los recibos locales de la pausa que
solo estaban en ese checkout sigue sin localizarse. En el árbol exacto `97e2929`,
`m3-a-resume-20261120-benchmark-1` dio 1.431,6706 ms p95 con 40 muestras válidas.
`m3-a-resume-20261120-heartbeat-1` agotó la espera de arranque antes de medir gaps.
Ambos fallaron con evidencia de nivel 2. La descripción del PR registra esos
resultados. El chat M3 A prepara el diagnóstico y el handoff del scope para M3 B.
El diagnóstico del arranque `m3-a-startup-phases-20261120-1` registró ocho
`VACUUM` de reconciliación en 25,56 s antes del timer. La serie instrumentada
`m3-a-search-phases-20261120-1` conservó página y cursor y ubicó trabajo en el
preflight R50 y en las lecturas y la codificación del digest. Son diagnósticos
de nivel 2. Tras aislar el arranque según ruling 71, el heartbeat sobre el corpus
intacto completó ocho consultas y drenó workers. Falló con 55,349 ms de gap
máximo frente a 50 ms. Siguen los diagnósticos de optimización.

Nivel 2 focalizado en M3 B. El RED de citas obtuvo 25 fallos por la ausencia del
resolver en `f987ec6`. El candidato dio 24 pases y un fallo del fixture de fecha,
corregido en `ac493c9` con rerun pendiente. El RED Qt obtuvo 13 errores de setup
por módulo ausente y el controlador `1027d22` obtuvo 13 pases. Las correcciones
pedidas por la revisión reprodujeron dos fallos y luego pasaron 17 casos en
`2745ec7`. Citas `19c06b3` dio 25 pases. Su review pidió verificar roles de audio,
reproducido en tres RED de `4cabf16`. `7323613` pasó los 28 casos. El RED CLI
obtuvo 21 fallos por grupo o scope ausente y el RED de navegación, 13 fallos.
La primera corrida integrada de `4635bd1` abortó con el backend Qt real cuando el
fake del test CLI no interceptó un módulo ya cargado. Se corrigió el arnés en
`9413b6c`. La corrida selectiva `m3-b-review-red-4635bd1` dio 16 pases, 12 fallos y
1 caso deselectado. `m3-b-focused-f894923` pasó 465 casos. Su review encontró
una pérdida de autoridad del reintento al entrar al worker y la excepción de
SQLite al conectar sin tipar. En `589a832` fallaron los tres RED que lo reproducen.
`m3-retry-storage-green` pasó 41 y omitió un symlink por privilegios. `8fd64f1`
los integra. La segunda review pidió cubrir pérdida de audio elegible antes del
proveedor. `c11a9e3` obtuvo 2 fallos y 4 pases, incluida una revisión nueva
publicada por un retry válido con transcriptor sintético. `ea61d6a` integra el
fix. La re-review requiere alinear la elegibilidad con el WAV normalizado
que consume el worker. El RED `a31daca` quedó preparado para la próxima ventana.

El RED de distribución dio un fallo en `m3-qt-multimedia-bundle-red-c4045a5`
sobre el ZIP real del PR #28, SHA-256 `ac5f4c5b45796f7cb31e56e7a532412650fcc6f45a06fc04f3282fa332db8e9d`.
Faltaban el módulo y la DLL de QtMultimedia. `1fb797c` prepara un paquete con el
backend nativo de Qt 6.11.2 y sus avisos. Pasó review estática, build diagnóstico
`44781729845` sobre `f894923`, prueba de presencia y smoke real de instalador
con datos sintéticos, nivel 2. El WAV `3d96608` y la sonda en
`m3-library-marker-fixture/here/.superpowers/m3-audio-probe` preparan el control
de seek sin leer audio personal. La sonda tiene revisión estática pendiente de
compilación y calibración. Falta nivel 3.

Nivel 2 diagnóstico en M5 B. La recuperación de importación obtuvo 11 fallos en
`fada4fc` y 1 pase con 10 fallos en `8083b91`. La corrida mostró el fallo de
staging con rutas largas de Windows y un problema de supervisión del subproceso.
Se conserva el repro original. Separar el rerun de recuperación con ruta corta
del control de rutas largas con el arnés corregido. El RED de ruta corta dio
9 pases y 2 fallos. El candidato `d0449fc` pasó los 11 casos con ruta normal.
El RED JSON `05ba0ee` dio 3 pases y 3 fallos esperados. `fc36f91` pasó los seis
casos y la review independiente. El lote `m5-import-integration-dcb3660` dio
89 pases, nivel 2 focalizado. Claims sigue pendiente de M4 B. Falta certificación
de nivel 3 para M3–M5.

El RED de recuperación `0986b7d` dio 4 fallos esperados, entre ellos pérdida de
SQLite durante borrado, rollback interrumpido, temporal de recibo y raíz Windows
profunda. `m5-ir-green-41150f5` pasó los cuatro y el lote de continuidad exacto
pasó 89. M5 B prepara RED de los adaptadores CLI y Qt sobre el contrato actual
de M3 B, con solo pruebas nuevas. M3 A tiene la ventana exclusiva Windows y el
escritor de `service.py` y `output/paths.py` para R50. El orquestador integra
y controla merges.

## Pausa solicitada — 2026-11-13

Se pausaron los frentes activos por pedido del usuario. Retomar desde estos
commits locales y sus worktrees. M2 sigue cerrado en 0.3.0. Copilot sigue omitido
por ruling 64 y la prueba de dos horas, **omitida por ruling 5**.

| Frente | Rama | Worktree bajo `C:/Users/savar/.codex/worktrees/` | Head / responsable |
| --- | --- | --- | --- |
| M3 A, chat `01a1af04-e779-7363-be61-c840651b0b18` | `codex/m3-search` | `2710/here` | `07785b6`, chat M3 A |
| M3 A, PR abierto | `codex/m3-a-product` | `m3-a-product/here` | `0c16065`, root |
| M3 A, prototipo diagnóstico | `codex/m3-native-digest-probe` | `m3-native-digest-probe/here` | `ca80f49`, `search_budget_design` y chat M3 A |
| M3 B, integración | `codex/m3-library` | `m3-library-group/here` | `0c16065`, root |
| M3 B, RED de citas | `codex/m3-library-resolver-red` | `m3-library-resolver-red/here` | `f987ec6`, `portability_export_red` |
| M3 B, RED Qt | `codex/m3-library-playback-red` | `m3-library-playback-red/here` | `9fbd450`, `portability_contract_review` |
| M3 B, RED CLI | `codex/m3-library-cli-red` | `m3-library-cli-red/here` | `0c16065`, `portability_contracts`, solo preparación |
| M5 B, exportación e inspección | `codex/m5-portability` | `m5-portability-core/here` | `c202604`, root |
| M5 B, integración de importación | `codex/m5-portability-import-service` | `m5-portability-import-service/here` | `8083b91`, root |
| M5 B, RED de recuperación de importación | `codex/m5-portability-import-recovery-red` | `m5-portability-import-recovery-red/here` | `fada4fc`, `portability_export_red` |

[PR #28](https://github.com/srodriguez-lefebre/here/pull/28) queda abierto. La
revisión independiente del head exacto aprobó el alcance funcional y la entrega
de contratos a M3 B. Package Windows pasó en `43477815761`, CI `43477815724`
seguía en curso al pausar. La suite local de ese head se interrumpió al 59 % y
el heartbeat quedó sin ejecutar. Se conserva el log parcial y el recibo
`m3-a-pause-checkpoint-1.json` en el scratch del chat. El prototipo aislado
midió 621,37 ms p95, nivel 2 diagnóstico, y falló el requisito de 500 ms. Sin
resultado de aceptación nuevo para búsqueda o playback.

M5 B verificó 19 casos de exportación/recuperación en `29d5642` y 48 de
inspección/exportación en `27c86d1`, nivel 2 focalizado. Importación en
`52c7c5d` dio 10 pases y un fallo. La corrección `cf8e39f`, las correcciones de
recuperación posteriores y los 11 casos de proceso de `fada4fc` quedan sin
corrida nueva. El ruling 68 registra el límite JSON autorizado de 32 MiB,
pendiente de implementación y pruebas. La certificación Windows de nivel 3
queda pendiente para la entrega final. M4 y los demás grupos mantienen sus
dependencias.

También se preservaron los cambios remanentes de los worktrees de M2 en
`e6bef92` (`b5-w3-red`), `04b31f9` (`m2-b-lifecycle-resume`), `3b6375e`
(`m2-b-r62-regressions`) y `e701020` (`m2-b-release-docs-prep`). Quedan como
commits locales de pausa en sus ramas. El log de instalación de `uv` quedó
fuera de Git en `m2-ingestion/here`.

## Checkpoint anterior — 2026-11-13

M2 cerrado. Versión 0.3.0 mergeada por [PR #25](https://github.com/srodriguez-lefebre/here/pull/25)
en `b4b3d1b`, head revisado `55d68e2`. Nivel 2. CI Windows `43461764790` pasó
1273 pruebas y omitió los dos smokes de wheel, que pasaron en la corrida local
no editable. Package `43461764783` verificó el instalador estable y los seis
hashes del artefacto. El detalle está en ACCEPTANCE y DELIVERY.

M3 A sigue en chat `01a1af04-e779-7363-be61-c840651b0b18`, rama
`codex/m3-search`, worktree `C:/Users/savar/.codex/worktrees/2710/here`.
El consumidor R52 pasó dos casos focalizados y review independiente en
`2ece845`. El ACK de borrado pasó los dos casos de no resurrección y review
independiente en `d8fb75b`, nivel 2. Candidato integrado `3148869` en
`codex/m3-final-integration`, worktree
`C:/Users/savar/.codex/worktrees/m3-final-integration/here`, a cargo del chat M3 A.
La corrida `m3-final-functional-1` terminó con 1538 pases, 21 omisiones y 19
fallos en 769,32 s, nivel 2. Se investigan fallos de composición, binding del
oráculo R61 y contaminación de entorno del runner. La review independiente
marcó además el cursor largo y la recuperación de la instancia inyectada.
El p95 medido de 879,48 ms frente a 500 ms sigue fallido. Ruling 66 autoriza
el prototipo diagnóstico aislado en
`codex/m3-native-digest-probe`, worktree
`C:/Users/savar/.codex/worktrees/m3-native-digest-probe/here`, base `ca80f49`.
`search_budget_design` escribe el agregado nativo, el chat M3 A coordina el
adaptador y root coordina compilación y runtime. Compilación, smoke nativo y
268 controles por cada plan SQL pasaron, nivel 2 de diagnóstico. La medición
`m3-native-subquery-p95-2` dio 621,37 ms p95 en 40 muestras y falló el gate de
500 ms. Queda aislado y conserva pendiente la equivalencia de empates tipados.
La preparación contractual de
M3 B está en el [PR #27](https://github.com/srodriguez-lefebre/here/pull/27), rama
`codex/m3-playback-handoff-docs` y worktree
`C:/Users/savar/.codex/worktrees/m3-playback-handoff-docs/here`, head `dbc9d16`.
`portability_contracts` escribió y `portability_contract_review` aprobó. CI y
package Windows pasaron y el PR se mergeó en `94e175f`. M3 B espera el PR abierto
de A y su handoff.

El PR de contratos de portabilidad [#26](https://github.com/srodriguez-lefebre/here/pull/26)
se mergeó en `5286dd4`. La rama de grupo M5 B `codex/m5-portability` queda en
`C:/Users/savar/.codex/worktrees/m5-portability-core/here`. La sonda Windows
pasó ocho controles de capacidad en el host de desarrollo, sin symlinks.
El RED `2c427d5` tuvo seis fallos esperados por falta de módulo y en `a69a490`
pasaron los seis. La corrida de recuperación `fd57c23` pasó nueve controles y
reprodujo cuatro fallos de publicación y recuperación. La revisión pidió cambios
y aprobó el núcleo corregido `29d5642`. La ejecución en ese head pasó los 19
casos en 21,81 s, nivel 2 focalizado en NTFS local. El RED de inspección de ZIP
`a326ccf` reprodujo 38 fallos esperados por ausencia de `inspect`, nivel 2.
Worktrees bajo `C:/Users/savar/.codex/worktrees/`, cada uno con sufijo `/here`:

| Directorio y rama `codex/<directorio>` | Responsable | Fuente |
| --- | --- | --- |
| `m5-portability-export-red` | `portability_export_red` | `2c427d5` |
| `m5-portability-export-recovery-red` | `portability_export_red` | nuevo RED de proceso y recuperación sobre `2c427d5` |
| `m5-portability-export-green` | `portability_export_green` | fachada, repositorio y nombres `cde01fd` |
| `m5-portability-export-native` | `portability_windows_binding` | handle de Windows `701e992` y `49982b5` |
| `m5-portability-export-service` | root | integración y correcciones de export en `29d5642` |
| `m5-portability-export-evidence` | `portability_contracts` | evidencia RED y capacidad `ea57f69` |
| `m5-portability-import-red` | `portability_export_red` | RED de inspección `a326ccf` |
| `m5-portability-import-inspect-green` | `portability_export_green` | validación del directorio central, base `a326ccf` |
| `m5-portability-import-stage-red` | `portability_export_red` | RED de importación y staging `4cb591d` |

Root coordina el handoff, las ventanas de prueba y los archivos compartidos.
Claims y respuestas esperan M4 B.
M4 A/B y M5 A/C conservan sus dependencias. Sin certificación nueva de nivel 3.
Dos horas sigue **omitida por ruling 5**. Ruling 64 sigue vigente.

## Candidato 0.3.0 y reanudación — 2026-11-13

M2 A está mergeado en PR21. M2 B prepara 0.3.0 en `codex/m2-lifecycle-release`,
worktree `C:/Users/savar/.codex/worktrees/m2-lifecycle-release/here`, responsable
root. La fuente `d00a1d4` pasó 110 casos y review independiente. El full de
`350761a` terminó con 1251 pases, 18 omisiones y seis fallos. Se corrigieron
la separación de raíces de cinco fixtures y la traducción de errores SQLite
en la vista R50. El foco de 19 casos pasó en `f626933`. La revisión aprobó
también `ec57204`, con Ruff y formato limpios en 27 archivos Python y cero
diagnósticos nuevos de mypy en los archivos modificados.

El full corregido `b5-full2-ec572046c277-38b8014c` pasó 1257 pruebas y omitió
18 en `ec57204`, nivel 2. Son 16 omisiones por privilegio local de symlinks y
dos del wheel, que tienen su corrida aparte. La rama de entrega integra solo
documentación después de ese commit. El wheel 0.3.0 no editable pasó los cuatro
smokes de distribución y la ayuda CLI se comprobó desde esa instalación. Nivel 2.
El paquete Windows, la CI final y el merge siguen pendientes.
El equipo de desarrollo tiene una instalación con el AppId y el grupo Inicio
de here. La prueba de ese instalador estable usa el runner Windows del workflow.

M3 A sigue en el chat `01a1af04-e779-7363-be61-c840651b0b18`, rama
`codex/m3-search`, worktree `C:/Users/savar/.codex/worktrees/2710/here`.
R63 tiene verificación funcional de nivel 2 y el benchmark sigue fallido en
879,48 ms p95 frente a 500 ms. Los dos tests RED de R52 quedan preparados para
el handoff de B. Root coordina las ventanas de runtime y la integración.

M5 B tiene el borrador revisado de contratos en `codex/m5-portability-contracts`,
worktree `C:/Users/savar/.codex/worktrees/m5-portability-contracts/here`, commit
`2c25a54`. Root coteja los bindings y publica el PR de contratos después de
mergear M2 B. Los claims esperan M4 B. Evidencia estática solamente.

Ruling 64 omite la review de Copilot. Siguen la revisión independiente, el PR
con evidencia, CI Windows verde en el head final, merge protegido y registro.
Dos horas permanece **omitida por ruling 5**. Sin bloqueo humano en este punto.

## Historial de reanudación — 2026-11-13

El usuario reanudó los hitos 2 a 5. Los chats originales de M2 B y M3 A
retoman desde los commits de pausa. Al verificar la reanudación, `origin/main`
seguía en `3e76a20` y Git conservaba ambas ramas de trabajo. Faltaban los
checkouts `faab`, `2710`, `b5-w2-red` y `m2-native-owner-red`. Cada frente
recupera sus archivos desde los commits registrados y confirma su nueva ruta.

M2 B recuperó `codex/m2-lifecycle` en
`C:/Users/savar/.codex/worktrees/m2-b-lifecycle-resume/here` con `89305c1`.
El autor RED trabaja en `C:/Users/savar/.codex/worktrees/m2-b-r62-red-resume/here`
sobre `ae00b44`. M3 A recuperó el checkout `2710/here` en `49baac0`.

M2 B reprodujo R62 en `7455561` con siete fallos esperados y dos controles verdes.
Los mismos nueve casos pasaron en `r62-green-1-25354948`, nivel 2 focalizado,
con los deltas de productor `F027EA6D` y consumidor `07FEABE6` ensamblados en
`C:/Users/savar/.codex/worktrees/m2-b-r62-green-check/here`. La review de
`7994ebc` encontró cuatro casos abiertos. `b5_w3_red` los reprodujo en
`r62-review-red-1-009b46ae` con 4 fallos esperados en `c8747aa`, worktree
`C:/Users/savar/.codex/worktrees/m2-b-r62-rollback-red/here`. Corresponden a
compensación con lease retenida, recibo exacto con una tumba previa en la ruta,
recibo discordante y retry WAV-only tras reconstruir el catálogo. El control
adicional de token ambiguo también falló en RED. La corrección `b365fa9` pasó
los 14 casos en `r62-green-2-51dfbfff`, nivel 2 focalizado. El lote
`r62-regressions-1-ceef8825` dio 82 pases y 13 fallos, incluidos errores de
configuración del runner y un caso pendiente de aviso por SQLite ocupado.
Siguen la segunda review, la verificación de esos fallos y la integración.
La preparación de integración de cuatro archivos de tests quedó en `f85e366`,
rama `codex/m2-b-r62-product-prep`, worktree
`C:/Users/savar/.codex/worktrees/m2-b-r62-product-prep/here`.
`b5_docs` entregó el borrador de 0.3.0 sobre `89305c1` desde
`C:/Users/savar/.codex/worktrees/m2-b-release-docs-prep/here`.
El integrado `d00a1d4` pasó los 110 casos de `r62-green-3-61a5088c` y la
revisión independiente de fuente. Root recibió el worktree `m2-b-r62-product-prep`,
integró la documentación y subió la versión del candidato en `350761a`.
El full `b5-full2-350761adfc76-37061fb5` terminó con seis fallos.
Cinco fixtures de captura incluyen el lock del catálogo en su árbol de sesiones.
El sexto requiere recuperar la traducción SQLite de R50 en la vista de lectura.
Se prepararon el arreglo de fixtures `7b9f532` y la corrección de fuente
`f626933` en worktrees separados.
M3 A verificó automáticamente los focos de R63
en `ca80f49`: Ruff, 8 comparaciones con el oráculo v2, 10 controles de
generación y 240 regresiones de búsqueda. La review independiente aprobó ese
alcance funcional. `r63-query-benchmark-1` midió p95 de 879,4833999891125 ms
frente al requisito de menos de 500 ms. La sonda posterior midió 539,90 ms en
generación y 222,99 ms en la entrada al guard en una llamada instrumentada.
M3 A descartó la proyección JSON1 tras medir 3.446,71 ms frente a 782,22 ms
del control en la prueba de concepto. El benchmark sigue fallido.
Root coordina las ventanas de ejecución, la integración, los PR y los merges.

Preparación documental M5 B en `C:/Users/savar/.codex/worktrees/m2-segments/here`,
rama `codex/m2-segments`, base `1085a6e`, borrador `e20c095`.
`portability_contracts` ajustó el SPEC y el PLAN. `portability_contract_review`
aprobó la preparación estática. Quedan las comprobaciones contra el merge real
de M2 B y el PR de contratos antes de implementar.

El usuario indicó que no se pida review de Copilot por falta de cuota
(ruling 64). Para los PR pendientes siguen vigentes la revisión independiente,
CI de Windows sobre el head final, el merge protegido y el registro de evidencia.

## Pausa histórica — 2026-11-06

El usuario pidió guardar todo y cortar; luego precisó el mensaje en inglés:
`paused: checkpoint work in progress`. Todos
los grants de fuente/runtime están revocados; solo se permite cerrar procesos y
guardar el checkpoint. No reanudar automáticamente: esperar nueva instrucción.
Los apartados operativos siguientes son historia anterior a esta pausa.

Punto de retoma: M2 B conserva W1/W2 y reader R59 aprobados en sus scopes;
W3 parcial pasó7P/11,33s sobre5926/C234. El writer atómico R62 sigue pendiente
de RED/GREEN/review y bloquea SOURCE COMPLETE. Faltan regresiones finales,
tipos/full2, integración, versión0.3.0, distribución y gates del PR.
M3 A conserva UI R58 verificada automáticamente y R61 funcional; p95 real
848,67978ms sigue fallido frente a500ms. R63 es trabajo preparado sin verificación
nueva; R52 UI sigue pendiente. No hay aceptación ni certificación nueva de hito.
Versión vigente0.2.0. M5 B early core no empezó; claims esperan M4 B.

Los reportes/logs/manifests ignorados permanecen en sus worktrees y no se agregan
al repositorio. Los commits de pausa preservan WIP, no aprueban código ni planes.
Las filas de evidencia y omisiones conservan su alcance; dos horas sigue
**omitida por ruling 5**.

Checkpoints locales de pausa; carpetas bajo `C:/Users/savar/.codex/worktrees/`,
cada una termina en `/here`. Los diez WTs de los dos frentes quedaron limpios,
sin procesos propios activos. No hubo tests/builds nuevos durante la pausa.

| Worktree | Rama | Commit de pausa |
|---|---|---|
| faab | codex/m2-lifecycle | 89305c1 |
| b5-w2-red | codex/m2-b5-r62-red | ae00b44 |
| b5-w3-red | codex/m2-b5-w3-green | 3d97ed3 |
| m2-native-owner-red | codex/m2-b5-w2-red | 90fc7e2 |
| 2710 | codex/m3-search | 49baac0 |
| m3-search-evidence | codex/m3-search-evidence | f0db68d |
| m3-search-green | codex/m3-search-green | ca80f49 |
| m3-search-red | codex/m3-search-red | 8f37a9a |
| m3-ui-implementation | codex/m3-ui-green | 1ffa174 |
| m3-ui-red | codex/m3-ui-red | 348c567 |
| m2-preparation | codex/m2-preparation | 8a0aae7 |
| m2-segments | codex/m2-segments | 1085a6e |
| m4-grounded-preparation | codex/m4-grounded-preparation | fa6fa7b |
| m2-copilot-red | codex/m2-copilot-red-3 | d36e800 |
| five-milestones | codex/m1-desktop-recovery | 9d63d80 |
| ruling19-review | codex/pausado-ruling19-review | 7faf8e7 |

Orquestación y evidencia global: `m2-catalog/here`, rama
`codex/m3-library-contracts`, commit de pausa que contiene este registro. Los
borradores históricos/fixtures preservados no sustituyen los contratos vigentes;
no integrar ramas de preparación completas por el solo hecho de estar commiteadas.
`m2-ingestion/here` conserva únicamente `uv-setup.log` como artefacto local ajeno
al código. El checkout original no tenía cambios antes de registrar la pausa.

## Objetivo y checkpoint vigente

Implementar y entregar M2–M5 completos bajo ruling 20, sin cambiar definiciones
de terminado. **M2 A cerrado y mergeado** por PR [#21](https://github.com/srodriguez-lefebre/here/pull/21).
Nivel: implementado y verificado automáticamente con datos sintéticos; no aporta
certificación nueva de hardware, proveedor o instalación limpia. M2 sigue abierto:
ciclo de vida/borrado y versión 0.3.0 corresponden a B/cierre del hito.
Versión vigente 0.2.0; M5 exige distribución Windows 1.0.0 y aceptación real.

Merge A: `53712c5beedafc2d40e2abf10c13362a1c3c9f78`, 2026-11-06T03:22:06Z.
Head revisado/publicado: `8a498e207812592ff84a0acc5ba4e9f392e02373`.
Fuente final `c59ae331ebf124400ef3e476e5f0fcc029e83999`, src tree
`5c66169c0a93a8f0606a011bd9570bbdd9137903`, tests tree
`fb958d021eaf7d149c00a3a1be174e36fbef2a1c`, iguales al merge.

Los cinco pasos de §7 se cumplieron en orden: PR no borrador con evidencia y
review independiente; Copilot solicitado sobre 8a y recibido como review
`5795325438` COMMENTED a03:20:34Z, sin issues abiertos y C5 resuelto; cinco
observaciones respondidas en sus hilos; CI exacto verde; merge commit con
`--merge --admin --match-head-commit 8a498e207812592ff84a0acc5ba4e9f392e02373`;
ledger en DELIVERY. C5 reply `4488660276`, hilo `PRRT_kwDORsQx4M60W6_-` resuelto.
Suite Windows `42155415514`: **980 pasan/2 skips wheel/307,46 s**.
Package Windows `42155415501`: success. Los dos smokes wheel omitidos en CI
pasan por separado en el wheel fresco exacto: distribución4/7,51 s.

## Tarea en curso: dos frentes separados en ejecución

El PR de contratos [#22](https://github.com/srodriguez-lefebre/here/pull/22) se
mergeó en `73d3cb384bd422de6899d8c7b761f3f3159ea3f1` a2026-11-06T03:59:54Z.
Head revisado `96f76dbbcb5ab4ba92866d9acf73c2673e140ee8`; review independiente
APPROVED, PC1–PC3/R45 ADDRESSED. Copilot5795547739 recibido03:57:00Z sin issues
abiertos; observación BM25 respondida4488851997/hilo resuelto. CI Windows exacto
42159238848 pasó980/2wheel skips/304,66s y package42159238847 success. Merge
con `--merge --admin --match-head-commit 96f76dbbcb5ab4ba92866d9acf73c2673e140ee8`.
Esto acepta documentación; no implementa ni certifica búsqueda/ciclo de vida.

PR de registros/contratos [#23](https://github.com/srodriguez-lefebre/here/pull/23)
mergeado c4d79b2759cb857094f5fa22adf5b1ad2865f69b a2026-11-06T09:47:58Z.
Head13f23b569b9375cb88c8a686e1ca9f31afa8dbcd, review independiente APPROVED;
Copilot5798363749 recibido09:43:03Z sin issues abiertos, única observación
tipográfica respondida4491036067/hilo resuelto. CI42196684370:980P/2skips wheel/
277,06s; package42196683781 success sobre13f. Merge guardado, ledger root1f3d0ec.
Sin source/tests/config/version nuevos; no valida B5/M3 por arrastre.

PR documental [#24](https://github.com/srodriguez-lefebre/here/pull/24) mergeado
3e76a207cf828b50b8e07cbd8136830b38fea80b a2026-11-06T12:52:24Z. Head3e2f6bff768de0a36b7b6f6704f5b36a9637ae90,
review independiente APPROVED/R1–R2 ADDRESSED. Copilot5800249014 recibido12:45:47Z
tras solicitud12:43:20Z, sin observaciones. CI42219811270:980P/2skips wheel/326,90s;
package42219811401 success, ambos head exacto. Merge commit guardado y ledger
en DELIVERY. SPEC/PLAN M3 B preparados y R53–56 publicados; no GO de ejecución.
Los checks validan fuente A sin cambios, no el producto B/M3 aún no mergeado.

Root creó los dos chats pedidos en el mismo proyecto, cada uno con worktree y
rama propios. Ambos se inicializaron clean en contratos73d3cb3 con A53712c5 como
ancestro; hoy contienen el trabajo detallado debajo. No recrearlos ni reiniciarlos.

Proyecto `local-e307e59e09194b354122137b16ef6d77`, host `local`; raíz original
`C:/Users/savar/Documents/proyects/here` preservada. Root orquesta desde
`C:/Users/savar/.codex/worktrees/m2-catalog/here`, rama
`codex/m3-library-contracts`, con base de contratos73d3cb3 y main3e76a20 integrado.
La rama anterior `codex/m2b-m3a-orchestration` preserva sus checkpoints hasta883e3cd.
Esta rama publicó el PR24 de contratos de M3 B; conserva ahora su anotación posterior
al merge. No inicia la implementación B.

| Frente / responsable ejecutor | Chat ID | Rama | Worktree | Base |
|---|---|---|---|---|
| M2·B: ciclo de vida y borrado | 01a1af04-b158-75a2-8933-98783880f3e8 | codex/m2-lifecycle | C:/Users/savar/.codex/worktrees/faab/here | 73d3cb384bd422de6899d8c7b761f3f3159ea3f1 |
| M3·A: búsqueda y acciones | 01a1af04-e779-7363-be61-c840651b0b18 | codex/m3-search | C:/Users/savar/.codex/worktrees/2710/here | 73d3cb384bd422de6899d8c7b761f3f3159ea3f1 |

Briefs completos de cada frente en `.superpowers/m2b-front-brief.md` y
`m3a-front-brief.md`: objetivo, SPEC/PLAN, bases reales, archivos/leases,
dependencias, límites y DoD. Metadatos completos y GO entregados a ambos chats;
cada ejecutor usa subagentes RED/GREEN/review independiente/docs por tarea.
Root conserva contratos, integración, global docs y todos los merges. No reiniciar
trabajo previo ni duplicar fixtures. Monitorear los dos chats con wait_threads;
pedidos de leases/handoffs se registran en su progreso/reporte local.

Leases actuales: B posee models/repository/database/service, artifacts solo para
estado2/compatibilidad1 y CLI. El principal conserva service.py para W2/R59;
recovery.py fue liberado en todos sus WTs y transferido al consumidor distinto
en `b5-w3-red/here`, rama `codex/m2-b5-w3-green`. DTOs R46/errores canónicos y B4 completos
ya se integraron en M3; esto no concede un escritor libre de core al chat M3.
M3 posee search.py/tests y UI enumerada. B no tiene writer UI; la futura línea
app.py R52 espera API B verificada y RED consumidor, en la misma instancia/customroot.
Controller1278bdb fue escrito/verificado/revisado separadamente en el WT nativo.
Root autorizó integrarlo mecánicamente en B tras confirmar release del autor;
entonces controller vuelve exclusivamente al GREEN principal para registro R52.
Hasta esa transferencia no hay otro writer. Processing continúa sin concesión.
Cualquier handoff adicional se concede explícitamente. Migraciones/pyproject/uv.lock/versiones
siguen seriales y requieren permiso de root. Global docs son root-only.

Runtime del checkpoint: productor R59 liberó53479 y M3 benchmark R61 liberó92885
antes de reportar. B completó cuatro controles R60 en m2-native-owner-red:
sourceC20E/test814BF580, runner17F5E82D/freeze84B926A2,4P/11,55s y XML11,520,
0F/0E/0S;71 pins sin delta. Reporte actual confirma sesión28939 y dos hijos cerrados,
GLOBAL RELEASE antes de análisis. Root concedió ahora W3 PARCIAL exclusivo:
WTb5-w3-red/here, source5926/C234, siete casos originales; runnerCBABF69E/bindingA939943C,
151hashes/149tracked verificados por coordinador. Terminó7P/11,33s, XML11,266,
0F/0E/0S; GLOBAL RELEASE10689, sin timeout y procesos/pipes cerrados. Review W3
acotada pendiente; SOURCE COMPLETE/R62 siguen abiertos. Host libre, sin otro grant.
No trasladar4P deC20E a5926/C234/R62.
W1 terminó con release completo:
RED3F/1P/16,63s → GREEN4P/19,91s y regresión delete79P/5symlink skips/182,92s.
Fuente42a29fbfaef83c30b895ed6e75af95562313623a, src01400e51c9ac19d66309aaa6bf91976a95c91992;
misma re-review W1 ADDRESSED/SPEC y calidad del alcance APPROVED, sin nuevos C/I.
Los RED W2/W3 se ejecutaron separadamente sobre624be55; sus fallos conservan esa
base y no se atribuyen al fix posterior. Briefs y autores RED distintos:
b5_w2_red en
`C:/Users/savar/.codex/worktrees/b5-w2-red/here` (detached624 al crear),
b5_w3_red en `C:/Users/savar/.codex/worktrees/b5-w3-red/here`, rama
`codex/m2-b5-w3-red`. El ledger B registra bindings y ramas posteriores.
W2/R60 pasó5P/0F/0E/0S/35,41s, XML35,359 sobre service DIRTY
`C20E1FA450E33B940713F876383954BFC59F552CDDC22159EEAC60E94D00C38F`, patch
`27F620F6664F3AF98300263EE24856BDF3A660A383982AB03DFFDF0E6872D53F`.
HEADd480317988a8f355f64034a782637a63f97f1ae4 es test-only sobre42a; no es un commit
del source ejecutado. La review posterior `task-5-whole-w2-re-review-report.md`
declara W2 ADDRESSED/SPEC-R60 y calidad APPROVED solo para C20E/27F, sin nuevos
hallazgos. Regresiones/finaltypes/full2 y Task5/grupo B/M2 siguen abiertos.
RED59 real:40 collected,38F/2P/0E/0S/14,57s, XML14,528;29 fallos por método
ausente y7 por módulo ausente, dos regresiones de identidad de excepción del
caller y dos controles Busy que pasan. Runner real en `b5-w2-red/here`, rama
`codex/m2-b5-r59-red`, HEAD05c9fed2449cab24d16f96a9997bcdd86cb461bc/base624,
serviceA987; testAC636ACC3E2003E67B261EB32829A297F1739350B37BDBE05944FDDEDF154C36.
No es corrida sobre C20E ni prueba de los cuerpos de guard ausentes.
Checkpoint B-R59-PRODUCER-20261106: GREEN2 real40P/13,85s/XML13,818 sobre
service DIRTY5926969BA9466288AC1F1A96EACE91CC7ED677A944A0128CC6736D44BA8038E4 y
testF70551A820D2568196DB0A88996AFF08D559341F11D75DF948F7C72367DB04E6;71pins intactos.
RunnerWT b5-w2-red recibió65 fuentes MAIN, no source Git624 como si fuera5926.
Apéndice final task-5-r59-producer-review-report.md APPROVED productor/SPEC/calidad,
0hallazgos; su headerNOT_RUN es histórico. ConsumerC234/W3 sigue parcial: edge
marker unmapped confirmado. R62/dd59c82 fija scope WRITER/revalidación antes de
material/marker sobre mismo servicio, sin inventarID/padre; nuevo RED asignado.
Contrato R62 APPROVED estático en análisis raíz; no GREEN de ese writer.
Reader40P no prueba ese writer ni SOURCE COMPLETE/aceptación del consumidor.
Productor service y consumidor recovery conservan sus writers/WT separados.

M3 terminó10 controles R55 y240 de regresión, sin fallos; benchmark40 válido
p95=902,413285ms FALLA500. Perfil residual2 exit0 liberado: control5207ms/perfil2744ms,
diagnóstico frío no comparable al gate warm. Observer R55 aporta diagnóstico de
contención, sin aceptación UI nueva. Pytest eliminó su catálogo temporal205; R57 materializó
un input sintético dedicado con el helper aceptado intacto, nueva procedencia y
sin afirmar igualdad binaria. Fixture READY y diagnóstico CPU terminaron exit0:
cinco warmups y dos llamadas públicas, estado DIAGNOSTIC_COMPLETED_NOT_BENCHMARK;
no pytest ni p95 nuevo, no causa general del host inferida. Fuentee198/tests a2ac/
config7143 invariantes en la corrida9406c72; docs posteriores31e9 no son su HEAD.
Checkpoint M3-R58-UI-2b66246-20261106: en WT propio `m3-ui-implementation/here`,
commit2b66246198965eaca7a3df5d326837f4851b8942,
src12774a1e6702ab943934280d05d0de6897e8378d: solo library.py/bridge.py, sin integración
en primary al reporte. Dos corridas separadas: r58-ui-focus-1 pasó44P/16,36s/
XML16,294 y r58-ui-heartbeat-1 pasó1P/23,56s/XML22,725. Gate original de ocho
callbacks IDs3–10/render/drain completo:1788 ticks/1789 gaps/8941,6032ms,
máximo8,9216ms<50ms, errors=[], drained=True;83 pins intactos. Verificado
automáticamente solo ese gate/contexto; review Task2 posterior APPROVED acotada
en r58-ui-review-report.md sobre el mismo2b/source12774, sin findings. Coordina auxiliares
con páginas; page/page/título/crossview/externos conservan límites. No p95<500ms,
full/grupo/certificación. Reporte r58-ui-verification-report.md; hashes en ACCEPTANCE.
R61commit62a5197 autoriza query source-only: representación local del cursor y
equivalencia de bytesv2. Checkpoint M3-R61-2e69-20261106: ownWT m3-search-green,
HEAD2e69c117c66f2ed1763b4e507c76947729dd9647/src04f3fd9802b378efc588df393d92e238915f203f,
tests1ebd4f67002a13300f454e18dcb49580d9d416e0/config7143. Existentes10P/4,95s +
240P/52,48s enab837 con misma producción04f3; captura old7821/e198/f303 real1P/
1,78s/XML1,656, oráculo11E2CF6B data-only. Comparación actual8P/1,96s/XML1,912,
capture DESELECTED/cero hidden skips,84bindings. Benchmark original cinco warmups+
40 correctas FALLA p95=848,6797799909255ms frente a<500,1F/49,08s/XML48,730,
0E/0S y siete hashes invariantes; muestras retenidas, sin rerun ni causa inventada.
R61 no integrado al reporte; no pase combinado con R58. Propuesta query siguiente
solo texto autorizada; R52 UI RED preparado sin handoff aceptado. Reports/hashes
en ACCEPTANCE; no full/tipos/Task1/grupo/certificación.
Handback posterior: M3 integró mecánicamente R58 como
`38cff90dba7bd2914e44491ec60e8cf58771a2f4`; src12774/tests a2ac/config7143
son idénticos a2b (diff de source/tests/config vacío). No es otra corrida.
B liberó full1
con1F/310P/110,55s, detenido25% por -x: el override global TRANSCRIPTIONS_DIR del
runner interfirió con test_config_paths. No editar producto/test para ese fallo;
corregir runner después de conservar intento. Mypy fresco B4→624 válido/estable:
65 módulos,79→79 errores/115→115 diagnósticos, multiset delta vacío, NO type-clean.
Review whole Task5 original CHANGES REQUESTED, Critical0/Important3: W1/P2 pierde la prueba
del WAV v1 durante purge; W2/P1 puede completar con staging publicado identificado
todavía presente; W3/P2 discovery de journal nativo omite supresión antes de audio/
candidato. Root leyó informe completo y acepta los tres como defectos del contrato
existente. Sus continuaciones acotadas cerraron W1/W2 y aceptaron solo el reader
R59; W3/C5 y handoff writer R62 siguen abiertos.
service.py conserva GREEN principal para W2/R59; recovery.py ya está transferido
al consumidor distinto, con W3 parcial7P sobre5926/C234 y R62 aún abierto. El lote v2 sobre
624be55 terminó sin fallos: native31P/4skips, CORECOMPAT-A98P/5skips,
CORECOMPAT-B47P, SERVICE-FULL13P y guard/privacy20P. Son cinco corridas separadas,
no un full agregado. R54-R1 ADDRESSED/SPEC y calidad APPROVED por la misma review
independiente. M3 conserva R57 READY; R58 verificó sus dos corridas y liberó runtime.
Sin full/build/uv/mypy/benchmark concurrente. Consultar SIEMPRE
los dos ledgers antes de reasignar: este owner es snapshot, no permiso permanente.
READERS7 terminó7P/5,10s, incluido primer uso con raíz vacía. M3 verificó44P/14,42s
con título/Conflict/FTS B4 real, source75be486/srcd1bddf68, test-format ed8e5aa.
Mypy79→78, cero diagnósticos nuevos; primer79→87 retenido. Foco legado33:30P/3F/
10,44s por mensajes vacíos; fix5db071a y foco7P/1,92s/Ruff/formato/scoped review
aprobados. No trasladar44/mypy al nuevo sourceE3 como nueva corrida.
La UI42P/review R1–R3 conservan su alcance previo; ningún grupo está cerrado.
La validación corta de integridad del fixture M4 también terminó y liberó host:
exit0/0,326s; sin here/proveedor/pytest. Faltan full/build/benchmark y mypy final
M3 sobre fuente integrada posterior a E3; el mypy79→78 previo corresponde a D1.
Cada coordinador libera inmediatamente al terminar los comandos, antes de reportes
o commits; root rota el host con esa confirmación. Escritura autorizada, docs y
reviews de lectura siguen en paralelo, nunca dos tests/builds/mypy pesados.

Última verificación B4: fuente4e1b2cead486563516b69e09fb45543ee85b53aa,
src4ca55bbe7e88ff59f79dee47e8740fef2253718e y testsbf4f471a87ad84df8ac41f8da375f8fd1025f42e.
Review R1 ADDRESSED/SPEC/quality aprobados sin hallazgos; mypy79→79 sin delta.
FULL2 pasó1043P/9S/390,03s, XML1052/0F/0E/9S/interno389,952s en
`task-4-full-2.*`; HEAD/src/tests clean y sin mutaciones. Dos skips wheel sin
entorno y siete privilegios symlink. Task4 ACEPTADA en local tras docs distintos
96e8aed255a442acb14bbdff5c32877ec201c628, source/tests invariantes. Root autorizó a
M3 consumir SOLO ese frozen B4, preservando DTOs/error canónico y sin lease libre
de core. Edición real/FTS M3 ya tiene44P; integración B5 guard/reindex pendiente.
B5 core y adapter nativo tienen RED real; implementación final aún no aceptada.

B5 RED100 observado92F/3P/5S/47,39s, XML0E. Dos pases son controles válidos; el
tercero era falso positivo de --yes por comando ausente. Un fallo era fixture sin
speaker. Tres reparaciones test-only075e280d28805b87012c059a74b5d9f95b54aeb7,
1654 líneas/3 archivos, se verificaron luego en M3 test-mergef3f06bc con B4 real
intacto:3F/0E/3,44s, fallos correctos de B5 ausente; no repetir100 ni sumar runs.
Primer GREEN B5:94P/1F/5S/82,78s sobre fuente candidata dirty documentada, no4ca55
como si fuera ese código. Fallo real de lectura que migraba bajo SQL Busy; fix
serviceSHA29ACF3565A6E554C944194DB85673E9CDC6B378830375897B0C60CD10695A845,
patchE6978215EE65CC29E268212D109BE6ED591A2B5416ABF0690343A035C384A18C.
Caso único1P/2,89s/XML0F0E0S, source/tests congelados y host liberado. No llamarlo
100 verde. READERS7 pasó7P/5,10s y foco guard/reindex34 pasó33P/1symlink skip/
16,57s; Ruff/formato propios pasan, mypy fresco79→79 sin delta. No full/interop B5.
Candidato técnico c9f64c3a65bf640196934e9c515ca158e0759ebf, cinco archivos fuente:
review independiente original CHANGES REQUESTED, H1/H2 Important/P2. H1 probe FTS
enmascara OperationalError de storage/Busy como FTS ausente; H2 registry I/O dentro
de BEGIN IMMEDIATE en edición/status. READ BEGIN R50 sí es intencional. Mismos
RED/GREEN/reviewer completaron fix acotado; c9 sin fix no se transfiere solo.
Reporte task-5-handoff-review-report.md. RED H1/H2 real7F/2P/6,63s, sin errores/skips,
test-commit225144d; storage/Busy y cinco lecturas bajo WRITE alcanzaron los asserts.
Los dos controles pasan. Fix fuente fbe00e6583dc51bf78976ddd9834cda1f8581e1a:
9P/6,91s y Ruff/formato propios; SAME re-review H1/H2 ADDRESSED, SPEC/quality
APPROVED sin nuevos C/I. Root leyó reportes completos y autoriza SOLO ese handoff
técnico a M3 para R50/reindex/query; no R52 posterior ni delete como feature aprobada.
Srcf564ba13a9564bd3526f598419c3879eb8a1d405/tests42b9653094ffdc4762fd00a926a9bf756f56fd46.
Mypy/compat/full de fbe aún pendientes:79→79 y33P/1S son históricos de c9.

RED nativo R52 tiene su worktree separado verificado:
`C:/Users/savar/.codex/worktrees/m2-native-owner-red/here`, rama
`codex/m2-native-owner-red`, tests-only5667553213202655227d55f0e5fb6437e078018a.
Corrida real34:27F/4P/3S/0E/20,75s; tests SHA4EA0AC6AD6160B679189ED2A2CD3CACD8AAD82C87B7E75E8B728890CEFA12EFA.
Incluye tail existente que libera ownership canónico antes del cleanup nativo y
dos guards destructivos reales; APIs ausentes/controles se clasifican aparte en
task-5-native-red-report.md. No son34 fallos ni prueba de cleanup por API ausente.
RED terminó/congeló archivo y pasó a H1/H2 en faab. Nuevo GREEN b5_tail_green
solo controller.py en ese WT; principal conserva servicio/contexto/CLI/recovery
en faab. Tail fuente1278bdbf8843a315c75c1feb8976447a566bffe1 tiene review SPEC/quality
APPROVED acotada y foco2P/higiene; contexto R52 completo aún no aprobado. TAIL-M1
control live-finally pasó1P/2,15s tras un primer fallo de limpieza del test1F/2,39s;
ambos retenidos, misma review ADDRESSED y test-only integradoedf7d48. API mínima
R52:4P/4,08s antes del registro StartRequest; no aprobación R52 completa.
Native35 histórico:30P/1F/4S/24,35s (skips privilegios), fallo de reporte por no incluir
operación preparada retenida aunque los archivos quedan protegidos. Compat2:
1P/1F/2,63s, before-stage alcanza exit73 pero deja .publication.json; assertions
A intactas. Fixes acotados de reporte/staging pasaron foco7/6,73s sobre aa16784.
La review independiente encontró R54-R1 Important: referencia léxica equivalente
por dot/case no protegía el staging. Nuevo RED real4F/3,31s; el primer intento
4F/3,03s era arnés sin import json y se conserva separado. Fix solo service
624be55f2101450e458824739de8c8bfb80d0ba5, srcd4575d1a70a2fb2579acb669012cfb8510caa1d5,
testsc0a23fdec146fdeb9eeb1eb6d5ffe6e1d1672d0b: GREEN9/5,49s/XML5,422, controles
completos de bytes, owner, row/outcome y orphan genuino. Misma re-review acotada
R54-R1 ADDRESSED/SPEC y calidad APPROVED, sin nuevos hallazgos; los cinco XML v2
confirman el lote amplio anterior sin fallos/errores. Los tipos frescos no agregan
diagnósticos; el full de esta fuente todavía no pasó. Review whole Task5/grupo y docs distintos en
curso; W1 y W2 corregidos/revisados en sus bindings propios, W3/R59 abiertos y
fallo de runner se registran separados.
No transferir R52 ni cerrar Task5. Warning Ruff heredado queda para root/config
serial. Cero writer UI B.

M3 checkpoint primario docs d7009c724da4fb5eca01d8e28fcec56f6529bf3d, source
e3ede58c32f5d848988b94dd8c4a5319512546c3, tests35efb958f45c9b107f3639fe018c6f0df42b3b79.
Preview1 funcional pasa, visual no aprobada por glifos cuadrados. Probe sin here
aisló QFontDatabase offscreen vacío. Preview2 con Segoe UI existente registrado
solo en arnés pasa función y texto legible en capturas533x452/1100x800; root vio
ambas. Conserva scroll/clipping de encabezado y combos, sin aprobación general
de layout ni cambio de producto/config/font global. Scoped reviewer aprobó solo
esa evidencia; docs distintos ya integrados en00cf0ea97bc97a6f8b02d242e26d179f407f8003,
sourceE3/tests35 sin cambios. Tres corridas Qt separadas,
no pytest/benchmark/DB/audio/red/proveedor. P95 real1000/100000 con guard+heartbeat,
full/mypy/review integrada y gates hospedados de M3 siguen pendientes. Root autorizó
merge mecánico fbe y RED de consumo real; fuente R52 posterior excluida. M3 no
mergea B5 a main por arrastre antes del PR B aceptado/mergeado.

Consumo R50 ya observado: RED51F/9,10s en4df, GREEN51P/12,30s en965+formato,
commit del árbol ejecutado d6a3b474dcf77a0ca0026110b6df7c74711e30e8, srcbb4abfd7/
tests4a21a4b. Misma review Task1 SPEC/quality APPROVED acotada, sin findings; reindex
real y segundo no-op alcanzados. Escala/p95/regresión177/full/tipos aún pendientes.
R53 ruling cbe2fa7: callback opcional al mismo MemoryService, una delegación/sin
recursos propios; reads/fakes intactos. RED11:10F/1P/2,40s, API ausente real. GREEN
search y UI separados, combinadod8ec92b3d6aca9ec872b8f3d77d446f7b686f91a,
src0bea8879682c59f3aa5b7dbede9eba1ec78760f0. GREEN11 R53 pasó y ambas reviews
acotadas aprobaron; regresión separada228P/1benchmark skip/38,53s. Control real
de220 IDs con límite SQLite64 pasó1P/4,09s; review aprobó ese
alcance, sin inventar RED ni p95. Docs distintos integraron23 corridas pytest y
3 previews Qt separados, preservando la evidencia anterior.

Primera medición real1000 reuniones/100000 segmentos/98901 docs FTS,220 registros
y221 supresiones: cinco warmups y40 muestras públicas correctas, p95
1303,482595ms, FALLA presupuesto500ms. XML72,625s; todas las muestras conservadas.
UI real1F/16,17s: Busy en requests3/4 library-page, cero respuestas de búsqueda y
objetivo de ocho consultas no alcanzado.115 ticks/578,164ms/max11,2069ms son intervalo diagnóstico
parcial, NO heartbeat aprobado. Workers drenados. Fuente0bea; árbol ejecutado
89c628f569eeef03236ce29c5194f0c46d722815, tests13c5f8c06be88b4260b03151a18be3dd416e4855;
commit posterior equivalente ec800c271b6a5dafd2ab269dd7cc28edba42db5d solo formato.
Reporte guarded-performance-first-run-report.md y dos logs/XML mantienen fallos
y límites. Perfil causal completado exit0/4,605s: control público1266,9802ms,
perfil1878,714ms, generación1549,1202ms con199918 dumps; overhead material, no p95.
DB/registro y79 archivos invariantes. R55 registra encoding completo por lotes256;
fuente preparada e783255ca23b4da6288f3ac45bf61bc01f08085a, search.py14+/3-,
src e198e954658eefa15222476adfdfc06db797b8a8, integrado en4cbc7f3, sin velocidad aprobada.
Nuevo control público independiente99fb803, integrado enb9f16b02740fb436c9ff69d7db21e2b90fe70568,
tests04249753f743b7835cfcdcfb2cfde072e90d3027 antes de higiene. Controles10P/4,79s y
regresión240P/46,02s sobre R55; query nuevo40 muestras válidas falla p95
902,413285ms, XML51,151s, setup11,5409553s separado. `guarded-query-perf-r55-1.xml`
conserva todas las muestras; no es un pass por mejorar el resultado anterior.
Mismos RED/reviewer cubren cambios públicos de corpus más allá del primer lote;
sin test espejo de codec ni caché. Timeline Busy pendiente. No relajar budgets,
corpus, guards o flujo. Todo sigue en evidencia automática
sintética, sin cierre de grupos ni certificación real.

Historia de corridas, no estado vigente ni resultados sumables:

- B4:111 casos,47 fallan/64 pasan/0 errores/0 skips/76,66s; XML interno75,352s.
  59 controles A/M1 y5 controles de marker pasan; fallan APIs/efectos B4 ausentes.
  Prueba inicial43F/63P se conserva; corrección de un supuesto de formato JSON,
  sin cambiar intención/autoridad, previa a implementar. Test-only9d311a9.
- M3 Task1 público:111 fallan por módulo ausente/1 benchmark omitido/66 casos
  privados diferidos/6,91s, fuente test-only3d6a6ac; XML112/111F/0E/1S.
  UI:2 fallan/8 pasan/3,31s, fuente3becc79;16 casos de widgets pendientes de DTOs.
  No presentar preparaciones/deselecciones como tests aprobados.

Checkpoints posteriores: B4 selección111/111 GREEN76,83s, foco afectado294 pasan/
4 deselected/162,35s y mypy79→79 sin delta (previos al suplemento R48).
R48 RED11F/39deselected/5,26s, commit1f4d22d; GREEN122P/78,72s y probe aislado
verificados. Handoff e0d083ae2aebb0ef1b1d428f6b3beba42ac9030f,125 líneas en tres
archivos, autorizado a M3 preservando sus DTOs. API real ya disponible técnicamente,
sin aceptación global B4. M3 core e4d2469:177P/1benchmarkskip/26,79s, los66 controles
estructurales ejecutados; review APPROVED CORE con B5/performance/full pendientes.
UI post-DTO histórico:2F/8P/26 errores por LibraryView ausente en el baseline.
Después focos13F/25P,9F/29P,9F/29P conservaron causas/harness; el foco35P/3F
alcanzó assertions R48 reales. Correcciones UI y foco final38/38P/12,57s tras
formato/Ruff verifican roster, speakers scoped, targets pasivos y teclado, con DB
sintética/customroot real para dos casos. No edición real B4/guard B5/perf/full aún.
R1 compatibilidad129:128P/1F/86,28s y callers40P/30,10s. El único fallo era conteo
global del journal: RED preservó publicación exacta, agregó persist_state complete
y ausencia global de prepared; caso único corregido1P/3,15s, testf244018. No sumar
estos pases como full129; full corregido pendiente sobre fuente4e1b2ce.

B4 FULL1 en9e10c9cb1e0d385feea8cd73c8912b9ebd8a3929 pasó1036/9skips/391,45s
(1045 casos,0F/E;2wheel ausente+7privilegios symlink). REVIEW necesita fixes por
B4-R1 Important: snapshot postcommit de transición/evento no queda journalizado
si falla el write, próximo reconcile no lo repara. R49 y una sola tanda RED/fix/
re-review por los mismos roles; Task4 y B5 siguen abiertos pese al full verde.
Fuente full congelada src284c24aa/tests01db3dd; logs/review en ledger B4.

Leer reportes de progreso canónicos para pedidos de lease y resultados (además
de wait_threads): `faab/here/.superpowers/sdd/M2_LIFECYCLE_PLAN/progress.md` y
`2710/here/.superpowers/sdd/M3_SEARCH_PLAN/progress.md`, bajo la raíz de worktrees.
R46 fija los defaults de DTOs/errores compartidos; R47 define empty-search→MeetingPage
con filtros intactos. Antes de UI GREEN completar RED real con los DTOs. El marker
slow se registrará por root en el próximo cambio serial de metadatos/versionado.

M3 consulta el catálogo A; título real espera contrato B4 aceptado y reindex/
supresión real espera B5. R45 exige generación de búsqueda coherente, rechazo
estructurado antes de usar ranks viejos y reinicio por requests existentes;
el listado fecha/ID conserva continuidad ante inserts. No contar fakes como
integración. Root facilita los contratos/DTOs compartidos con un solo writer.

## Secuencia y concurrencia autorizada

Los nueve grupos: M2 A `M2_CATALOG` tareas1–3 (**mergeado**); M2 B
`M2_LIFECYCLE`4–5; M3 A `M3_SEARCH`1–2; M3 B `M3_LIBRARY`3–4;
M4 A `M4_GROUNDED`1–3; M4 B `M4_PRODUCT`4–5; M5 A `M5_SETTINGS`1–2;
M5 B `M5_PORTABILITY`3; M5 C `M5_RELEASE`4–5. Versiones al cerrar hitos:
0.3.0, 0.4.0, 0.5.0 y1.0.0, en serie y solo con sus criterios verificados.

M2 B y M3 A tienen los gates de contratos cumplidos y avanzan tras sus GO. Después
de M2 B puede adelantarse M5 B en lo permitido por §11, con claims pendientes
hasta M4 B. El resto conserva las dependencias de los planes. Preparación del
PLAN siguiente, fixtures sintéticos y respuestas a Copilot pueden solaparse.

Un escritor por archivo, incluso entre worktrees. Migraciones, pyproject/uv.lock y
versiones nunca en paralelo. Suites de tiempos/deadlines, benchmarks, tests y
builds pesados usan lease exclusivo del host; sin otro test/build/mypy pesado.
Cada tarea: RED independiente → GREEN → review por otro autor → docs/evidencia.
Cada PR conserva los cinco pasos de §7 y Copilot recibido/respondido sobre el head
actual, nueva review tras cambios sustantivos, CI exacto y merge guardado. Sin
excepción por silencio, sin tocar ruleset/permisos.

## Evidencia y recuperación sin repetir trabajo

Fuente y aceptación A: `docs/development/M2_CATALOG_SPEC.md`, `_PLAN.md` y
`M2_CATALOG_EVIDENCE.json`; matriz/corridas en `ACCEPTANCE.md`, ledger en DELIVERY.
La implementación ya resuelve F1–F7, NB1/NB2, GR1–GR3 y C1–C5; no reabrir sus
reviews ni repetir suites históricas sin un cambio/fallo nuevo que lo justifique.
Reports locales bajo `.superpowers/sdd/M2_CATALOG_PLAN/`:
`progress.md`, `task-1-fix-review-report.md`, `task-2-review-report.md`,
`task-3-fix-1-review-report.md`, `task-3-fix-2-review-report.md`,
`task-3-fix-3-review-report.md`, `group-review-report.md`,
`group-fix-review-report.md`, `capture-fixture-forwarding-report.md`,
`copilot-1-review-report.md`, `copilot-2-review-report.md`,
`copilot-3-red-report.md`, `copilot-3-implementation-report.md`,
`copilot-3-review-report.md`, `copilot-3-docs-report.md` y `m2a-merge-docs-report.md`.
CI actual: `.superpowers/m2a-ci-42155415514.log`; wheel exacto:
`.superpowers/m2a-wheel-catalog-probe-c59ae33.json` y
`.superpowers/m2a-copilot-3-wheel-smokes.log`/`.xml`.
Los fulls fallidos1/3, diagnósticos, timeout no explicado y pases históricos
conservan alcance en DELIVERY/ACCEPTANCE/evidence y sus logs; no se relabelan.

Preparación futura ya existente: `next-contracts-preparation-report.md`,
`parallel-contracts-review-report.md`, `lifecycle-plan-preparation-report.md`,
`m3-plan-preparation-report.md`; briefs `.superpowers/m2b-front-brief.md` y
`m3a-front-brief.md`, ya completos y entregados a los chats.
Fixtures de memoria/búsqueda/playback/ZIPs siguen preservados en sus worktrees;
su generación no certifica benchmark, seek o importador.

M5 B preparación SPEC/PLAN congelada en `m2-segments`, blobs7904905/c77a3c6;
8 cláusulas/14 bloques preservados y review independiente estática READY FOR
PREPARATION, cero hallazgos, reporte `m5-portability-preparation-review-report.md`.
Core permite avanzar tras merge real M2 B, PR docs/handoff B5 y GO;
claims/answers esperan M4 B y adapters sus contratos reales. Sin código/runtime
portabilidad; reporte `m5-portability-contract-preparation-report.md` en ese worktree.

M3 B preparación SPEC/PLAN lista en `m2-segments`, blobs6ce254b/305a1f9 tras deltas
aditivos R46/R47/R50,76 líneas R51/R52 y23 líneas R53. Preflight y scoped reviews independientes READY
FOR PREPARATION, cero hallazgos,11 cláusulas/14 bloques completos preservados.
Reportes `m3-library-preparation-review-report.md` y
`m3-library-binding-delta-review-report.md`, `m3-library-r51-r52-binding-review-report.md`
y `m3-library-r53-binding-review-report.md`; R56 aclara publicación documental
anticipada, con todos los bindings/handoffs obligatorios antes del dispatch;
Citation/offsets/Unicode, empty MeetingPage/InvalidInput.reason, guard READ,
ACK/leases/fase durable/contexto explícito fijados; adaptadores
reales B5/UI y PR docs final todavía pendientes. B puede arrancar sobre A revisado con PR abierto, tras contratos
necesarios, y rebasear tras merge A: no agregar un bloqueo de merge más estricto.

Fixture real de actualización 0.2.0 ya descargado una vez desde el workflow M1
42102224714/artifact13450512413: `.superpowers/m5-upgrade-baseline-0.2.0/`.
Se verificaron seis checksums y 730 entradas de inventario; reporte
`.superpowers/m5-upgrade-baseline-preparation-report.md`. No se ejecutó ni instaló:
es preparación verificada del fixture, no certificación de actualización.

No se leyó el `.env` real ni audio/biblioteca privados. Smokes opt-in futuros usan
solo HERE_PROVIDER_SMOKE_KEY y texto/audio sintéticos según ruling19, sin imprimir
ni persistir la credencial. Máquina Windows limpia: pedir al necesitar la fila,
con protocolo escrito. Prueba de dos horas: **omitida por ruling 5**, nunca aprobada.
Contrato B5/M3 R50 fijado tras review independiente estática, reporte
`b5-m3-guard-contract-review.md`: B posee lock y única read transaction;
snapshot de UUIDs y aliases exactos antes de LIMIT, fingerprint incluye IDs
resueltos; caller abre/cierra. Parser único, sin nuevas migraciones o p95 relajado.
No es implementación/aceptación B5; ambos frentes reciben API antes de integrarla.
R51 ya fija DTO DeletionResult/ReindexReport, categorías/fase durable, ACK fuera
del lock/SQL y revalidación, purger privado en transacción y roles de temporales.
Reporte `b5-result-hook-contract-review.md`; la interoperabilidad real se prueba
en M3 integrado con fuente B5 exacta, sin duplicar búsqueda en B. Source B5 aún
no aceptado; las nueve fronteras R42 y todos los gates siguen obligatorios.
R52 fija contexto nativo explícito y lifetime canónico; review de callpaths reales
`b5-native-owner-contract-review.md`. No registry durable de roots, ni autoridad
por origins. RED suplementario observado; código adapter espera GREEN/review, sin cambiar deadlines
ni ubicación/retención/privacidad. Posible línea GUI es del writer M3 y espera API
B verificada/RED consumidor; ninguna segunda instancia por defecto.
R54 aclara que R52 conserva la autoridad R23/R30 para staging privado pre-journal
bajo la misma .memory.lock y guards propios A. El análisis independiente
`r52-owned-staging-contract-analysis.md` y el RED real before-stage motivan el fix;
no permite fuentes nativas/committed/unknowns ni relaja assertions/retención.

Mapa de cierre M2 listo: `m2-milestone-closure-readiness.md` coteja16 criterios y
DoD, evidencia A y faltantes B/0.3, nueve fronteras R42 y protocolo build/smoke.
No exige máquina limpia antes de M5 ni convierte full B4 con R1 abierto en cierre.
Fixture sintético M4 preparado por catalog_group_fixes, único archivo nuevo
`m2-preparation/here/tests/grounded_fixtures.py`, blob7d74c8221c219779099d5a3d4654951692f2997e,
SHA256318E31615073D5DDC9B73937DA10A1DA7BA291A59F74497C1983533F6094D265.
4 reuniones/5 revisiones/13 pasajes/8 anchors, estándar y declarativo. Validación
independiente de integridad exit0/0,326s, dos construcciones frescas, IDs/referencias/
8 spans Unicode y hash inalterados; sin here/proveedor/pytest. Dos exit1 previos
del arnés (whitelist tuple y stdout cp1252) conservados en
`m4-synthetic-corpus-validation-report.md`; ninguna aceptación semántica M4.
Futuro RED inspecciona/consume al recibirlo UNA vez por root.
Reporte `m4-synthetic-corpus-preparation-report.md` en ese worktree; corpus previos intactos.

Preparación documental M4 A en paralelo: responsable catalog_group_fixes,
`C:/Users/savar/.codex/worktrees/m4-grounded-preparation/here`, rama
`codex/m4-grounded-preparation`, base73d3cb384bd422de6899d8c7b761f3f3159ea3f1.
Preparación y review distinta terminadas READY WITH EXPLICIT HANDOFF DEPENDENCIES,
sin findings. SPEC1a31bbbc/PLAN23704cb;12 criterios/15 bloques completos preservados.
Brief explícito entregado; solo M4_GROUNDED_SPEC/PLAN y reportes ignored, sin
código/fixtures/runtime/GO M4. Conserva dependencia cierre M3. El actual SearchQuery
no provee selección meeting_ids ni OR del retrieval M4: handoff interno pendiente,
sin asumir MemoryService.search, SQL directo de intelligence o filtro post-LIMIT.
Root resolverá ese contrato; no es bloqueo humano ni aceptación de un adapter.

No hay bloqueo humano actual. Próximo: recibo de controles R60, RED writer
R62 y productor/consumidor reales; regresiones/finaltypes/full2 B. R58 conserva su
handoff aprobado; R61 mantiene p95 FALLIDO848,6798ms y próxima propuesta solo texto.
R52 UI espera handoff aceptado. W3 parcial7P liberó10689; review acotada pendiente;
R57 sigue siendo diagnóstico, no aceptación.
PR documental24 ya mergeado; M3 B todavía espera GO y handoffs reales.
Alternar host tras release confirmado.
Consultar ledgers antes de rotar
host o asignar archivos; los planes/fixtures/reviews ya preparados no se reinician.
Reanudar desde estos IDs/reportes sin recrear los chats ni repetir el catálogo A.
