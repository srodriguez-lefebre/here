# Estado de here

Actualizado: 2026-11-05. Leé CLAUDE.md primero. Historial y rulings:
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
- Código integrado actual: `1f1bc88513840b84c388a61d9c077907cf2fba2e`, con fixture
  `1277e19`, candidato `3370483`, fix 1 `3a83482` y guard de legacy `1f1bc88`.
  Equivale a Green `c28472b` (fixes `16263c4` + `c28472b`): src tree
  `780529d42936c979740443c51f1022fd2bb99a5e`, tests tree
  `d226358ec038837dba6e40790c77eac74e74b538`. Source/config/scripts/workflows sin
  diferencias con Green. Integración no equivale a aceptación ni merge del grupo.
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
`docs/MILESTONE_2_PLAN.md`. Tarea 3 y grupo A siguen sin aprobar.

- Tarea 1: review independiente aceptada después del fix Busy. Foco 34 pasan/
  1 skip/4,70 s; full `M2A-T1-FINAL-20261105` 778 pasan/7 skips/173,68 s,
  ambos smokes de wheel pasan. Historial de fuente `9a80060`.
- Tarea 2: review SPEC/calidad PASS. Fuente `037fc03`, integración `1cdaadb`;
  foco 310/1/24,60 s, capture recovery 68/25,64 s, full 783/8/160,71 s.
  Sus dos skips de wheel no sustituyen la evidencia de wheel de otras corridas.
- Candidato inicial Task3 `6c422f0`: full estable 883 pasan/9 skips/598,56 s.
  La review abrió F1–F7. Ronda 1 los marca **ADDRESSED** tras 40 nuevos casos RED
  pasando juntos en 69,70 s, Ruff/format de 20 archivos limpios y mypy 79/79 sin
  nuevos mensajes. Esa re-review sigue solicitando cambios por **NB1 Important/P2**:
  doble publicación del controller y replay de marcador postcommit se confunden
  con un retry nuevo y dejan parent obsoleto. No hay aceptación de tarea/grupo.
- **Full integrado root cerrado: 5 fallos, 920 pasan, 7 skips en 553,39 s**.
  Logs/XML `.superpowers/m2a-integrated-final-1.*`; 932 casos, cinco failures y
  cero errors. Ambos smokes del wheel pasan; los siete skips son de symlink.
  Pendientes de diagnóstico/fix: tres snapshots de `test_capture_recovery.py`
  incluyen locks nuevos, `test_trans_command_completes_failed_session_audio`
  tiene stub sf.info sin subtype y `test_legacy_audio_only_session_still_retries`
  queda bloqueado por session.json ausente. No se omiten ni relajan assertions.
- Wheel y sdist frescos CPython 3.13.15 construidos. Probe no editable, off-repo
  con `-I`: 65 módulos Python byte-idénticos, schema 2, FTS5, FK/integrity correctos;
  `.superpowers/m2a-wheel-catalog-probe.json` guarda hashes. Es evidencia automática
  local, sin hardware/proveedor ni certificación de instalador o hito.

## Próxima acción obligatoria

1. Completar fixes de ronda 2: NB1 tiene RED independiente de 6 fallos/2 controles
   que pasan en 35,00 s, copiado a Green en `test_memory_publication_replay.py`.
   La fixture CLI pasó con WAV real y todas sus aserciones intactas; los otros
   cuatro fallos del full requieren correcciones de producto. La separación de
   guardas también tiene dos RED de materialización de journal legado y control
   de dos PASS contra `c28472b`. Los casos de lineage audio-only siguen en curso.
   Corregir sin debilitar snapshots, validación ni evidencia.
2. Focos, re-review de NB1/correcciones, nueva suite estable y review de todo el
   grupo. No convertir los siete ADDRESSED en aceptación mientras quede NB1.
3. PR de A, Copilot sobre head exacto recibido/respondido y CI Windows verde;
   merge guardado y ledger. Todavía no hay PR ni CI de A aprobados.
4. Después del merge A y de los PRs de docs de contratos necesarios, root crea
   dos chats de proyecto separados para M2 B y M3 A, con briefs completos y leases.
   No usar `1f1bc88` como base de merge futura: la SHA real aún es desconocida.

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

- Implementador Green: `ingestion_green`, `m2-ingestion`; reviewer independiente
  `ingestion_review`; autor RED mantiene archivos propios de ronda 2; docs
  `segments_red` en root. Nadie toca source de otro writer.
- Rulings 31–34 registran asociación revision/audio/schema2, legacy confiable y
  lease nativa `.capture_owners/` retenida por scope del trabajo hasta cierre real
  y persistencia. La review de ronda 1 está en `task-3-fix-1-review-report.md`.
- Rulings 36/37 fijan el contrato pendiente de marker v2/recibo v3 con token y
  prueba durable del replay, handoff privado de una sola publicación y reserva
  audio-only sin padre ficticio ni fechas de captura inventadas. No se agrega
  migración. Implementación de ronda 2 en `m2-ingestion`, aún sin aceptar.
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
  No hay cierre M2 A/M2, hosted CI de grupo ni certificación nueva real.
- Para retomar: leer este STATUS, el ledger local
  `.superpowers/sdd/M2_CATALOG_PLAN/progress.md`, los reportes y git log; continuar
  NB1 + los cinco fallos antes de crear los dos chats.
