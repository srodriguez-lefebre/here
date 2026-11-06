# Estado de here

Actualizado: 2026-11-06. Leé CLAUDE.md primero. Historial y rulings:
`docs/development/DELIVERY.md`.

## Objetivo vigente

Implementar y entregar M2–M5 completos por instrucción explícita del usuario
(ruling 20). Primero cerrar M2 A con todos sus gates. El ruling 35 ordena después
los dos frentes de proyecto M2 B y M3 A; todavía no se creó ninguno. Se mantienen
las definiciones de terminado y los tres niveles de evidencia. Versión vigente
0.2.0; 0.3.0 solo al cerrar B/M2. M5 exige distribución Windows 1.0.0 y aceptación real.

## Código y secuencia

- Base histórica: `878e7d4`, v0.2.0; PR #20/ruling 19 mergeado en `763a891` e
  integrado. Grupo actual: `codex/m2-catalog`, worktree
  `C:/Users/savar/.codex/worktrees/m2-catalog/here`.
- Fuente histórica previa a Copilot: `7bfd0cbf477a42aacc44b83b3d2da1e438801756`, equivalente
  a Green `ec55b9564d3af01247d3367675a340a361f6b6ef` tras la tanda única GR1–GR3.
  Src tree `2b4036f7deb905efb0a8836f378d267a066564b4`, tests tree
  `a4310416bd10dc922b49e010d8495a98bfdfd585`; source/config/scripts/workflows
  sin diferencias con Green. Integración no equivale a aceptación ni merge de A.
- Head histórico de full4 `ef70f78cfc9c8f2910e77b2b2818ce2e6b4d71ae` añade solo el forwarding
  de dos wrappers de `tests/test_capture_recovery.py`, desde autor
  `c3a3ccbcf2928502cbf0d0881fdfee3d7f86d84a`. Fuente idéntica a `7bfd0cb`; tests
  tree `30c88848b3cf42c9ce4443328e3c8e0100e2af44`, 114 assertions/cortes intactos.
- Fuente histórica Copilot1 `c3cdf085c03718ca6a9d5db87ea0de28fa9652b0`, desde Green
  `61fb8e2815887c19a1d6dc64572e5f469ec5cee9`: src tree
  `b465d3198840f96d5d15a85eeb45e4312e4c4dec`, tests tree
  `293bc5038bae501dee07631704899e5f385e9025`, iguales al worker. C1–C3 implementados
  bajo R40, full5 verde961/7/371,80 s. Copilot 5794829768 sobre D23 confirmó C1–C3
  resueltos; respuestas4488243766/4488243915/4488244052, hilos resueltos.
- Fuente histórica C4 56af5253a8eec498c594bd6f1e1e456ad69f9453 desde Green
  0a28842e9ac0a9f670a5e800077bf0fce8ae9e6d: src a17ec107c212d62e1984bb1cede094b579db5137,
  tests c6a29f96af9ffc1e0c488ca73a430ef9a99b86b6, iguales al worker. C4 implementado
  bajo R43 y review independiente final APPROVED con docs58176ce. C4 respondido
  en4488439090/hilo resuelto; Copilot5795057762 sobre5b confirmó su resolución.
- Fuente actual C5 c59ae331ebf124400ef3e476e5f0fcc029e83999 desde Green
  e8bd181f12fcfc25b8d66365689823552fa4e50b: src5c66169c0a93a8f0606a011bd9570bbdd9137903,
  testsfb958d021eaf7d149c00a3a1be174e36fbef2a1c, iguales al worker. R44 implementado;
  review final del diff/docs pendiente. PR21 sigue publicado en5b515e8; C5 aún sin
  respuesta/publicación ni Copilot/CI del nuevo head.
- Los nueve grupos se conservan: M2 A (`M2_CATALOG`, 1–3), M2 B
  (`M2_LIFECYCLE`, 4–5), M3 A (`M3_SEARCH`, 1–2), M3 B (`M3_LIBRARY`, 3–4),
  M4 A (`M4_GROUNDED`, 1–3), M4 B (`M4_PRODUCT`, 4–5), M5 A
  (`M5_SETTINGS`, 1–2), M5 B (`M5_PORTABILITY`, 3), M5 C (`M5_RELEASE`, 4–5).
  Versiones al cerrar hitos: 0.3.0, 0.4.0, 0.5.0, 1.0.0.
- Root es el único orquestador de contratos, leases por archivo, integración y
  merges. Un escritor por archivo; migraciones, pyproject/uv.lock y versión en
  serie. Suites de deadlines, benchmarks y builds pesados se ejecutan en ventanas
  exclusivas, sin otros tests/builds/mypy pesados.
- Cada tarea: subagente RED → GREEN → review independiente → docs/evidencia por
  otro rol. Cada grupo: PR con evidencia/review → Copilot recibido y respondido
  sobre head actual, nueva review tras fixes sustantivos → CI Windows verde en ese
  head → merge commit con `--admin --match-head-commit` → ledger. Ningún gate se omite.

## Tarea en curso y resultados reales

`docs/development/M2_CATALOG_PLAN.md`, tarea 3. SPEC `M2_CATALOG_SPEC.md`; hito
`docs/MILESTONE_2_PLAN.md`. Tarea 3 implementada, verificada automáticamente en
local y aceptada por review técnica independiente en la fuente previa a Copilot.
C1–C4 resueltos por Copilot; C5 implementado y verificado automáticamente en focos y
package locales. Review final del diff/docs y suite Windows completa del head corregido
pendientes. Sin certificación nueva; grupo A abierto.

- Tarea 1: review independiente aceptada después del fix Busy. Foco 34 pasan/
  1 skip/4,70 s; full `M2A-T1-FINAL-20261105` 778 pasan/7 skips/173,68 s,
  ambos smokes de wheel pasan. Historial de fuente `9a80060`.
- Tarea 2: review SPEC/calidad PASS. Fuente `037fc03`, integración `1cdaadb`;
  foco 310/1/24,60 s, capture recovery 68/25,64 s, full 783/8/160,71 s.
  Sus dos skips de wheel no sustituyen la evidencia de wheel de otras corridas.
- Candidato inicial Task3 `6c422f0`: full estable 883 pasan/9 skips/598,56 s.
  Ronda 1 marcó F1–F7 ADDRESSED, pero abrió NB1. El full histórico `1f1bc88`
  dio **5 fallos/920 pasan/7 skips/553,39 s**, logs/XML
  `.superpowers/m2a-integrated-final-1.*`; ambos smokes wheel pasaron. Sus tres
  snapshots con locks nuevos, stub sf.info y retry audio-only tienen corrección
  y cobertura posterior; el resultado fallido se conserva, sin relajar assertions.
- Ronda 2: NB1 y las regresiones de integración ADDRESSED por review independiente;
  61 pasan/68,06 s y foco controller/procesamiento/UI 69/23,02 s. Un timeout hijo
  intermedio de 30 s permanece registrado: caso idéntico pasó aislado 5,02 s y en
  foco final (call 2,72 s), sin causa probada ni cambio de timeout.
- Ronda 3: NB2 ADDRESSED, sin breakage nuevo en su diff; review
  `task-3-fix-3-review-report.md`. Guard UUID antes de construir/probar journal
  cuando falta metadata; IDs opacos del manifiesto existente siguen compatibles.
  Focos 3/2,40 s y 27/20,86 s. Ruff/format y mypy comparativo 79/79 sin nuevos.
- R36/R37 implementados en el histórico `8087e25`: marker v2 con token/source_kind, receipt v3 con prueba
  exacta y restauración de operaciones, replay tras crash real con DB presente o
  reconstruida; handoff privado consumido una vez por controller. Reserva audio-only
  canónica con origen/revisión NULL, cero revisiones históricas y fechas captura
  NULL durables. DB sigue schema 2, receipt v2 compatible sin inventar prueba.
- **Full integrado 2: 940 pasan/7 skips/cero fallos en 325,09 s**, fuente
  `8087e25`, iniciado solo a 2026-11-06T00:01:58Z. Log/XML
  `.superpowers/m2a-integrated-final-2.*`: 947 casos, cero failures/errors,
  tiempo interno 325,029 s; ambos smokes wheel pasan, skips siete symlinks.
  Full verde no equivale a aceptación de tarea/grupo mientras falte review global.
- Review de todo A por `catalog_group_review` terminada, base `763a891`..`8087e25`,
  suplemento docs `e00dd08`. Tres Important: metadata completed puede persistir antes del primer
  memory_pending marker (kill deja sin replay e interrupted); receipt JSON array
  puede abortar rebuild de vecinos por `.get`/AttributeError; cancelación aceptada
  durante persistencia del fallo de captura deja UI/manifest cancelled y catálogo
  failed con conflicto/marker atascado. En ese checkpoint GR1/GR2/GR3 estaban OPEN en
  `group-review-report.md`, veredicto fuente/docs changes requested. GD1/GD2 Minor
  corregidos en `7be93af`; tanda única R38 integrada. La re-review final acotada
  APPROVED sobre fuente7bfd/fixtureef70/docs8845877 marca GR1–GR3/GD1/GD2 ADDRESSED.
  Sin aceptación. No repetir esta review global ni reabrir NB1/NB2 ya addressed;
  después corresponde una sola re-review del diff de la tanda.
- Wheel/sdist nuevos CPython 3.13.15 y probe off-repo `-I` no editable de `8087e25`:
  65 módulos byte-idénticos, schema 2/FTS5/FK/integrity correctos. Hashes en
  `.superpowers/m2a-wheel-catalog-probe-8087e25.json`; probes anteriores históricos.
  Sin hardware/proveedor ni certificación nueva. No procesos, protocolos, deadlines,
  dependencias o versión nuevos. Meta final M5/1.0.0 intacta.

### Tanda integrada GR1–GR3 — checkpoint histórico anterior a Copilot

R38 implementado en `7bfd0cb`: todos los productores finales sellan marker 3 antes
del replace de metadata, con token/padre/source kind/intent y hashes exactos del
manifiesto prospectivo y artefactos con roles validados. Manifest pending o evidencia
distinta permanece con aviso y bytes/marker intactos, sin importar resultado viejo
ni llamar al proveedor. El aviso usa `RecoveryCandidate.error_summary`. El controller
real difiere privadamente fallo de captura hasta sellar cancelación aceptada antes
de la primera publicación, manteniendo una identidad/revisión parcial y causa/audio.
Recibos malformados se aíslan con aviso sin abortar vecinos válidos. Lectura de
marker 1/2, receipt 2/3 y schema 2 se mantienen.

RED independiente original: 11 fallos/1 control pasa en 12 casos. Primer GREEN:
12 pasan/21,67 s. Foco afectado: 404 pasan/1 symlink skip/159,03 s, anterior solo al
último fix de aviso pending-v3. El control 13 fue autorado sin ejecución baseline;
contra candidato dio 1 fallo/5 deselected/3,93 s por aviso ausente, manteniendo
R1/interrupted/bytes/marker. Foco final tras corregirlo: **26 pasan en 34,46 s**,
incluidos los 13 casos y controles recovery/ownership/NB2. Ruff/format limpios en
nueve archivos; mypy canónico base 8087/current 79/79, 20 archivos con errores/65
fuentes, mensajes nuevos `{}`; no base global limpia.

Build fresco wheel/sdist y probe no editable de `7bfd0cb` pasaron:
`.superpowers/m2a-wheel-catalog-probe-7bfd0cb.json`, 65 módulos byte-idénticos,
schema 2/FTS5/FK/integrity correctos. Full 3 empezó **solo** a
2026-11-06T00:44:04.5472024Z; logs/XML `.superpowers/m2a-integrated-final-3.*`:
**8 fallos/945 pasan/7 skips en 364,23 s**, XML 960/8F/0E/7S/interno 362,276 s.
Ambos smokes wheel pasan, verificados por lectura de XML. Todos los fallos son
TypeError por `before_metadata_commit` aditivo en dos wrappers existentes de
test_capture_recovery. Corrección test-only integrada en `ef70f78`: foco 13 pasa
en 24,82 s (6 kills reales/7 simulados), 114 assertions/cortes intactos. Full 4
empezó solo a 2026-11-06T00:54:42.9822634Z en `ef70f78`, mismo wheel exacto de
`7bfd0cb` porque fuente/config/scripts/workflows siguen iguales. Log/XML
`.superpowers/m2a-integrated-final-4.*`: **953 pasan/7 skips/cero fallos en
358,94 s**, XML 960/0F/0E/7S/interno 358,621 s. Lectura independiente confirma
summary/XML y ambos smokes wheel CLI/GUI pasan (3,018/2,634 s). Los siete skips
requieren privilegios symlink. Es evidencia automática local sobre fuente
7bfd/HEAD ef70f78, no CI hospedado ni certificación.
El full 940/7 pertenece a `8087e25`, no al head nuevo.
Review técnica independiente final APPROVED: GR1–GR3/GD1/GD2 ADDRESSED, cero
hallazgos técnicos abiertos. Reporte `group-fix-review-report.md`, docs revisados
`8845877d75342d1c0f65f1a6674155c7439ab4ce`. Gates hospedados/merge pendientes.

### Copilot PR #21 — checkpoint histórico c3cdf08

Review 5794475912 COMMENTED recibida sobre 3394149 a 2026-11-06T01:21:05Z:
C1 rollback de callback ordinario (`4488007346`), C2 catálogo compartido del
procesador por defecto (`4488007374`) y C3 aviso de replay completed rechazado
(`4488007407`). R40 implementado en tres fuentes/tres tests nuevos. Rollback
conserva BaseException/proceso; default processor comparte el MemoryService real;
replay completed rechazado deja candidato existente completed/can_retryFalse con
aviso, sin sondear audio ni ofrecer proveedor. Controles successful/no-marker intactos.

RED independiente: 5 fallos/3 controles pasan en 8 casos, 26,91 s secuenciales.
GREEN 8 pasan 9,77 s; foco afectado 325 pasan/1 skip symlink/141,36 s en 15 archivos.
Ruff/format 6 limpios; mypy canónico339/current 79/79 errores en 20 archivos/65 fuentes,
mensajes nuevos {}; no base global limpia. Review independiente acotada de esta
corrección en curso, sin aceptación todavía de C1–C3.

Wheel/sdist nuevos y probe no editable de c3cdf08 pasan, 65 módulos byte-idénticos,
schema 2/FTS5/FK/integrity. Full 5 empezó solo a 2026-11-06T01:47:43.289345Z,
logs/XML `.superpowers/m2a-integrated-final-5.*`: **961 pasan/7 skips/cero fallos
en 371,80 s**. Lectura independiente XML 968/0F/0E/7S/interno 371,710 s; ambos smokes
wheel CLI 3,747/GUI 2,978 s pasan, root exit 0. Full4 y su
review aprobada son históricos. CI Windows 339: 958 pasan/2 skips wheel/250,79 s,
run 42142394863; package 42142395026 pasa solo sobre 339, no sobre el head corregido.

## Próxima acción obligatoria

1. C5 integrado c59ae33; RED independiente de seis casos distintos: tres fallos/tres
   controles pasan en dos corridas secuenciales (13,93 s sumados). GREEN6/4,24 s y
   foco afectado169/114,05 s sin skips. Fuente/tests congelados; runtime de GREEN
   liberado. Review independiente del diff/docs pendiente (`copilot-3-review-report.md`).
   No local full7: R43/R44 exigen suite Windows completa en el head exacto corregido
   antes del merge, además de packageCI.
2. PR #21 sigue abierto/publicado en 5b515e8844cc0d135461517dd55e907b65845c4d.
   Copilot 5794829768 COMMENTED2026-11-06T02:09:04Z resolvió C1–C3 y abrió C4
   inline 4488285573. C4 fue respondido en reply `4488439090` con commit56af525;
   hilo `PRRT_kwDORsQx4M60We91` resuelto. Se pidió otra review Copilot sobre 5b515e8;
   review recibida `5795057762`, COMMENTED sobre 5b515e8 a 2026-11-06T02:42:41Z,
   confirma C4 resuelto y abre C5 (`4488473222`): catálogo corrupto oculta sesiones
   legadas cuando falla resolver ownership. R44 implementado; RED independiente
   `catalog_red` en m2-copilot-red-3 y GREEN `catalog_group_fixes` en
   m2-copilot-fix-3 terminaron y liberaron archivos/runtime. Ambos parten de5b515e8;
   no reiniciar ni duplicar su trabajo. No contar reviews propias como Copilot.
3. CI_D23 es histórico: tests42147566672,966 pasan/2 skips wheel/331,02 s;
   package42147566683 pasa. No prueba56af525. CI actual5b: suite `42151367541` y
   bundle `42151367491`, ambos success sobre 5b515e8; suite974 pasan/2 skips wheel/
   349,23 s. Son históricos para C5. Finalizar review independiente y responder C5
   con el commit publicado; pedir/recibir Copilot otra vez y
   verificar la suite completa/package en el nuevo head exacto antes del merge.
   Los pases 5b quedan históricos para la corrección nueva. Ningún gate se omite.
4. Después del merge A y PRs docs de contratos necesarios, root crea chats separados
   M2 B/M3 A con rama/worktree y brief sobre merge real. Ningún chat creado;
   borradores futuros/rulings R39–R44 se conservan, sin implementación adelantada.

### C5 — evidencia automática local actual

Solo recovery.py y un test independiente nuevo cambian. El fallo de retry_parent
por LibraryError o ambigüedad ValueError produce aviso fijo acotado no reintentable
con metadata validada e identidad/estado/duración reales. Antes de ese aviso no se
inspeccionan marker, error, audio ni material nativo; no se cambia fuente/DB ni se
inventa padre/UUID. OwnershipBusy/UnsafeSessionPath siguen propagándose y materialize
falla cerrado. Controles sin catálogo y dueño mapeado real conservan su conducta.
Ruff/format2 limpios; mypy canónico5b/current79/79 en65 fuentes/20 archivos, sin nuevos.

Probe fresco no editable c59ae33 de2026-11-06T03:08:14.816511Z:65 fuentes iguales,
schema2/FTS5/FK/integrity correctos. Distribución4 pasan7,51 s, XML4/0F0E0S/interno7,456;
wheel CLI3,996/GUI2,757, mutex0,007/path0,559 pasan. Logs/XML
`.superpowers/m2a-copilot-3-wheel-smokes.*`; hashes en el probe y ACCEPTANCE.
Evidencia implementada y verificada automáticamente en local; review final y suite
CI del head nuevo pendientes, sin certificación de hardware/proveedor/instalación.
Full5CDF y CI5b conservan sus fuentes históricas; no hay full local7 ni merge de A.

### C4 — evidencia local histórica de56af525

Solo recovery.py y un nuevo test cambian; src a17ec107c212d62e1984bb1cede094b579db5137,
tests c6a29f96af9ffc1e0c488ca73a430ef9a99b86b6. ValueError del marker se maneja
localmente, UnsafeSessionPath se propaga; aviso fijo acotado no repite payload ni
rutas. Candidato can_retryFalse antes de WAV/provider; bytes/identidad/estado real/
siblings y controles de retry válido intactos. RED 6/2controles/6,69 s; GREEN 8/8,18 s;
afectado 124/90,52 s sin skips, Ruff/format 2 limpios y mypy 79/79 sin nuevos.
Probe56af525 fresco 65 archivos iguales/no editable/schema 2/FTS5/FK/integrity;
distribución4 PASS7,44 s (XML 4/0F0E0S/interno 7,314), ambos wheel CLI 3,928/GUI 2,748
pasan. Review final APPROVED con docs58176ce y CI5b completo pasa974/2/349,23 s;
Copilot5795057762 confirma C4 resuelto. Full5CDF previo conserva
961/7/371,80 s en su fuente; no se relabela como resultado de56af525.
## Dos frentes futuros — ruling 35, todavía pendientes

Proyecto descubierto: `local-e307e59e09194b354122137b16ef6d77`, host `local`, Git
habilitado, raíz original `C:/Users/savar/Documents/proyects/here`.

| Frente | Chat ID | Rama | Worktree | Base merge A | Responsable ejecutor |
|---|---|---|---|---|---|
| M2 B · ciclo de vida/borrado | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente de asignar por root |
| M3 A · búsqueda/acciones | Pendiente | Pendiente | Pendiente | Pendiente | Pendiente de asignar por root |

Ningún chat fue creado. Antes de iniciar, cada brief fija objetivo, SPEC/PLAN
aceptados, SHA base real, archivos exclusivos/compartidos, dependencias, límites y
DoD verificable. Cada chat ejecuta su grupo completo con subagentes RED/GREEN/
review/docs; root conserva la orquestación, contratos, integración y merges.
No reiniciar ni duplicar trabajo ya realizado. Registrar IDs, ramas, worktrees y
responsables reales aquí cuando existan, sin inventar identificadores.

Los cuatro SPEC/PLAN de B y M3 A están preparados en `m2-segments`, todavía sin PR;
ver `lifecycle-plan-preparation-report.md` y `m3-plan-preparation-report.md` en el
ledger local. Reconciliarlos con el A finalmente mergeado y entregar los PRs docs
necesarios antes de despachar. M3 A puede avanzar búsqueda desde A aceptado, pero
sus acciones de título/reindex esperan los contratos aceptados de B que requieren;
no duplicar mutaciones ni contar fakes como esa integración. El texto de agenda del
plan de hito M3 debe alinearse en ese PR, sin cambiar sus criterios de aceptación.
Tras M2 B puede adelantarse M5 B con el gancho de claims pendiente hasta M4 B.
El resto conserva sus dependencias; no se inicia ningún frente en este checkpoint.

## Recuperación, decisiones y límites

- Tanda única GR1–GR3: RED `catalog_red` entregó tres archivos nuevos y resultados
  contra baseline detached `8087e25` en `ruling19-review`. GREEN
  `catalog_group_fixes` congeló `ec55b95` en `m2-ingestion`/`codex/m2-final-fixes`,
  integrado por root en `7bfd0cb`. RED/GREEN liberaron archivos/runtime; control
  extra autorado sin ejecución baseline tiene RED candidato y GREEN final reales.
  Scope: processing/controller/recovery, memory service/artifacts y output
  session_writer; ningún otro escritor. Reviewer `catalog_group_review` terminó
  su misma review global con el suplemento docs `e00dd08`; después revisa el diff
  de esta tanda una vez: veredicto final APPROVED. Docs/evidencia liberó sus once
  archivos; full 4 y verificación local completados, sin ventana runtime activa.
  Brief persistente: `.superpowers/sdd/M2_CATALOG_PLAN/group-fix-brief.md`.
  No se repiten las tareas ni las rondas anteriores; root integra y controla gates.
- Rulings 31–34 registran asociación revision/audio/schema 2, legacy confiable y
  lease nativa `.capture_owners/` retenida por scope del trabajo hasta cierre real
  y persistencia. La review de ronda 1 está en `task-3-fix-1-review-report.md`.
- Rulings 36/37 fijan el contrato de marker v2/recibo v3 con token y
  prueba durable del replay, handoff privado de una sola publicación y reserva
  audio-only sin padre ficticio ni fechas de captura inventadas. No se agrega
  migración. Implementados e integrados en `8087e25`, aún sin aceptación de tarea/grupo.
- R38 implementa el diseño interno de sello antes del commit final de metadata
  (marker 3, lectura1/2 compatible) y el cutover de fallo/cancelación antes de la
  primera publicación. Integrado en `7bfd0cb`, foco final 26 PASS y static sin
  nuevos errores; full 3 fallido por wrappers conservado y corrección test-only
  integrada ef70f78/foco 13/24,82 s/full 4 verde 953/7/358,94 s. Re-review final
  APPROVED y tarea 3 aceptada técnicamente; grupo A pendiente de gates hospedados/merge.
- Todas las corridas anteriores, incluidos fulls fallidos y diagnóstico de captura,
  conservan alcance y límites en ACCEPTANCE/DELIVERY/evidence JSON. El arnés revisado
  no cambió deadlines del producto; H1 bootstrap se observó en un caso y H2 sharing
  se reprodujo sin probar el solapamiento del fallo original.
- Fixtures sintéticos de memoria/búsqueda/playback/ZIPs permanecen en sus worktrees
  bajo ownership; no son benchmark, importador ni certificación de seek.
- PR #20/ruling19 está mergeado en `763a891`: head revisado `2f25a0d`, Copilot
  `5791519593` COMMENTED, inline `4485511464` respondido `4485574698`/resuelto,
  CI Windows `42102225168` y package `42102224714` pasaron. M2 no usa proveedor real.
- Máquina limpia: pedir con protocolo cuando una fila la necesite. Dos horas:
  **omitida por ruling 5**, nunca aprobada. No cambiar ruleset/permisos ni usar datos privados.
- Versiones y calidad/privacidad/aceptación/merges conservan sus requisitos.
  No hay cierre M2 A/M2; CI hospedado del head corregido pendiente, sin certificación nueva real.
- Para retomar: leer este STATUS, el ledger local
  `.superpowers/sdd/M2_CATALOG_PLAN/progress.md`, los reportes y git log; continuar
  los hallazgos de grupo y las verificaciones agregadas antes de crear los dos chats; no repetir
  el trabajo ya resuelto de NB1 y de los cinco fallos históricos.
