# Estado de here

Actualizado: 2026-11-04. Este archivo se reescribe; el historial y los rulings viven en
`docs/development/DELIVERY.md`. Leé `CLAUDE.md` primero.

## Objetivo vigente

**Preparar el plan para ejecución autónoma. No implementar.** Decidido por el usuario
el 2026-11-04 (ruling 1). M2 no arranca hasta una instrucción explícita. M1 quedó
cerrado bajo alcance reducido con el PR #14 (merge `beeedac`, v0.2.0).

## Dónde está el código

- `origin/main` = `beeedac`, v0.2.0. Suite local: 741 pasan, 8 skips esperados,
  1 falla por falta de `pwsh` en la máquina (no es un bug).
- PR #16 abierto (`docs/agent-start`): este archivo, `CLAUDE.md` y los rulings.
- `codex/ui-configuration-errors`: descartada por ruling 2. Pendiente de borrar en
  local y `origin` cuando el usuario confirme.
- Omitido de M1 y que sigue omitido: la reunión de dos horas con medición de
  recursos (ruling 5). La transcripción larga con diarización contra el proveedor
  real pasa a ser ejecutable porque hay credencial disponible, pero no está
  programada.

## Tarea en curso

Preparación del plan, en este orden. Ninguna implementa producto.

1. ~~`CLAUDE.md` y este archivo~~ — hecho.
2. ~~Resolver las doce decisiones abiertas y registrarlas como rulings~~ — hecho,
   rulings 1–12 en `DELIVERY.md`.
3. ~~Actualizar `PRODUCT_DESIGN.md` §1 y §6~~ — hecho. Tres brechas cerradas por M1,
   dos abiertas, una abierta en parte; la frase del modelo de M4 ahora apunta a
   `CLEANUP_MODEL` y al ruling 7.
4. ~~Criterios verificables de M2~~ — hecho, 16 criterios en `docs/MILESTONE_2_PLAN.md`.
5. ~~`M2_CATALOG_SPEC.md` y `M2_CATALOG_PLAN.md`~~ — hecho, grupo A (tareas 1–3).
   El grupo B (`M2_LIFECYCLE_*`) se escribe cuando A esté mergeado.
6. ~~Cerrar pendientes de `LIVE_LOGO_PLAN.md`~~ — hecho con el ruling 8.
7. ~~`DELIVERY.md` como ledger append-only~~ — hecho.
8. Sanear lint en un PR aparte: `ruff check --fix` y `ruff format` sobre toda la base
   (53 avisos, 15 archivos), mover `select` a `[tool.ruff.lint]`, y decidir si mypy se
   integra al layout plano o se saca de las convenciones. Hoy ninguno es gate.

## Acciones decididas que esperan confirmación del usuario

Son irreversibles o tocan el remoto; el agente no las ejecuta solo.

- ~~Borrar `codex/ui-configuration-errors` en `origin`~~ — hecho el 2026-11-04. La
  rama local sigue existiendo porque está checkeada en el worktree de Codex
  `C:/Users/savar/.codex/worktrees/m1-windows-distribution/here`, que tiene trabajo
  sin commitear encima (README, WHATS_NEW, `src/ui/configuration.py`,
  `src/ui/main_window.py`, `tests/test_ui_lifecycle.py`, y nuevos
  `src/ui/recordings.py`, `tests/test_desktop_navigation.py`). Último commit de la
  rama: `5b93630`. Falta decidir si ese trabajo se descarta con el worktree o se
  guarda en una rama de respaldo antes de borrarlo.
- ~~Eliminar el workflow de Ubuntu~~ — hecho en la rama `ci/remove-ubuntu-workflow`,
  PR aparte, pendiente de merge.

## Bloqueos que dependen de un humano

- Reunión de dos horas con medición de recursos: sin disponibilidad (ruling 5).
- Máquina Windows limpia y credencial real: disponibles a pedido; el agente las pide
  cuando una fila de aceptación las necesite, no antes.
- Merge de cualquier PR a `main`: siempre el usuario (ruling 4).

## Próxima acción sugerida

El usuario revisa y mergea el PR #16. Después, en este orden y sin preguntar nada:
la tarea 8 (lint) en un PR aparte, y las dos acciones destructivas de arriba cuando
las confirme. La implementación de M2 espera una instrucción explícita (ruling 1);
cuando llegue, el agente arranca por `docs/development/M2_CATALOG_PLAN.md`, tarea 1.
