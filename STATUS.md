# Estado de here

Actualizado: 2026-11-04. Este archivo se reescribe; el historial vive en
`docs/development/DELIVERY.md`. Leé `CLAUDE.md` primero.

## Objetivo vigente

**Pendiente de confirmación del usuario.** El último objetivo registrado
(2026-10-03) fue "cerrar el hito 1 en Windows y parar". M1 quedó cerrado bajo alcance
reducido con el PR #14 (merge `beeedac`, v0.2.0). El siguiente hito en el plan maestro
es M2, memoria estructurada. Hasta que el usuario lo confirme, no se arranca M2.

## Dónde está el código

- `origin/main` = `beeedac`, v0.2.0. Suite local: 741 pasan, 8 skips esperados,
  1 falla por falta de `pwsh` en la máquina (no es un bug).
- Rama sin disposición: `codex/ui-configuration-errors`, dos commits del 2026-10-03
  sobre `origin/main`. Agrega editor de configuración en la app, controles del
  indicador y feedback persistente de errores. Contradice "sin UI de credenciales en
  este hito" del plan de M1. Ver decisión abierta 2.
- Lo omitido de M1 por instrucción del usuario, y que sigue omitido: la prueba de dos
  horas con medición de recursos y la transcripción larga con diarización contra el
  proveedor real.

## Tarea en curso

Ninguna. Preparación del plan para ejecución autónoma, en este orden:

1. ~~`CLAUDE.md` y este archivo~~ — hecho 2026-11-04.
2. Resolver las decisiones abiertas de abajo y registrarlas como rulings en `DELIVERY.md`.
3. Actualizar `PRODUCT_DESIGN.md` §1: marcar qué brechas ya cerró M1
   (segmentos en `TranscriptionResult`, `meeting_id` UUID, API key opcional) y cuáles
   siguen (estados duales en `session.json`, rutas de la CLI que escriben directo).
4. Escribir la lista de criterios verificables de M2, con el formato de la de M1.
5. Escribir `docs/development/M2_CATALOG_SPEC.md` y `M2_CATALOG_PLAN.md` para el
   primer grupo de PRs (tareas 1–3 del diseño de M2).
6. Separar `DELIVERY.md` en ledger append-only; mover sus rulings vigentes a una
   sección propia al principio.
7. Unificar idioma y archivar los docs locales obsoletos.
8. Sanear lint: `ruff check --fix` y `ruff format` sobre toda la base (53 avisos,
   15 archivos), mover `select` a `[tool.ruff.lint]` en `pyproject.toml`, y decidir si
   mypy se integra al layout plano o se saca de las convenciones. Hoy ninguno es gate.

## Decisiones abiertas

Cada una tiene un valor por defecto. Un agente que no obtiene respuesta aplica el
default y lo anota; no se bloquea. Al responder, el usuario convierte la fila en un
ruling en `DELIVERY.md` y la borra de acá.

| # | Decisión | Default mientras no se responda |
|---|---|---|
| 1 | ¿Arranca M2 o el objetivo sigue siendo "parar"? | No arrancar M2. Hacer solo preparación de plan (tareas 2–7 de arriba). |
| 2 | ¿Qué se hace con `codex/ui-configuration-errors`? | No mergear. Se revisa cuando se planifiquen los ajustes de M5. |
| 3 | ¿`PRODUCT_DESIGN.md` es decisión tomada o propuesta? | Decisión tomada, salvo lo que marca como pendiente. Ya está escrito así en `CLAUDE.md`. |
| 4 | ¿El agente puede mergear sus PRs a `main` con permiso de administrador, como en M1? | No. Abre el PR con evidencia y sigue en otra rama. |
| 5 | ¿Hay máquina limpia, credencial real y tiempo para la reunión de dos horas? | No. Esas filas de aceptación quedan "pendiente", con protocolo reproducible entregado. |
| 6 | ¿Idioma de los documentos de plan? | Español para planes, STATUS y rulings nuevos; inglés se mantiene donde ya está (código, `development/`, commits). |
| 7 | ¿Qué modelo para `INTELLIGENCE_MODEL` en M4? | Se decide al implementar M4, con un ruling que nombre modelo y fecha. No hay "modelo de texto ya usado" en el proyecto. |
| 8 | ¿Cómo se ve una cancelación en el overlay y cuál es la marca final? | Cancelación: el overlay vuelve a reposo y desaparece sin check ni cruz; el detalle queda en la ventana. Marca: la actual, sin rediseño. |
| 9 | ¿Versionado por hito? | M2 → 0.3.0, M3 → 0.4.0, M4 → 0.5.0, M5 → 1.0.0. Reemplaza "v0.3.0 = producto completo". |
| 10 | ¿Los docs locales no versionados son vigentes? | No. Se archivan en `docs/archive/` o se borran; no se usan como contexto. |
| 11 | ¿El workflow de Ubuntu sigue siendo requisito? | No. Queda manual, no es gate de nada. |
| 12 | ¿Se siguen pidiendo reviews a Copilot? | No. Review independiente (otro agente o humano) más CI de Windows. |

## Bloqueos que dependen de un humano

- Máquina Windows limpia para certificar el instalador (decisión 5).
- Credencial real del proveedor para el smoke opt-in (decisión 5).
- Autorización de merge a `main` (decisión 4).

## Próxima acción sugerida

Que el usuario responda las decisiones 1 a 5 en una sesión. Con eso, un agente puede
ejecutar las tareas 3 a 6 de preparación sin volver a preguntar.
