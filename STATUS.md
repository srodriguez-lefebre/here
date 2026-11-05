# Estado de here

Actualizado: 2026-11-05. Este archivo se reescribe; el historial y los rulings viven en
`docs/development/DELIVERY.md`. Leé `CLAUDE.md` primero.

## Objetivo vigente

**Implementar el hito 2, memoria estructurada**, según `docs/MILESTONE_2_PLAN.md`
(ruling 13). Empieza por el grupo A, `docs/development/M2_CATALOG_PLAN.md`, tarea 1.
Versión objetivo al cerrar M2: 0.3.0.

## Dónde está el código

- `origin/main` = `16d7beb`, v0.2.0, con PR #16 (hoja de arranque, rulings, plan de
  M2) y PR #17 (workflow de Ubuntu eliminado) mergeados el 2026-11-05.
- Suite en `main`: 741 pasan, 8 skips esperados, 1 falla local por falta de `pwsh`.
- Omitido de M1 y que sigue omitido: la reunión de dos horas con medición de
  recursos (ruling 5). Máquina limpia y credencial real están disponibles a pedido.
- Worktrees viejos en disco con ramas ya mergeadas, sin tocar:
  `.codex/worktrees/five-milestones`, `here-worktrees/live-logo-overlay`,
  `here-worktrees/windows-app-core`. Candidatos a limpieza; no es urgente.

## Tarea en curso

Ninguna empezada. La siguiente es M2 grupo A, tarea 1: modelos, migraciones y
repositorio (`M2_CATALOG_PLAN.md`). Rama sugerida: `claude/m2-catalog`.

Secuencia prevista, cada grupo en su PR, el usuario mergea cada uno (ruling 4):

1. M2 grupo A: catálogo inmutable (tareas 1–3). Plan escrito.
2. M2 grupo B: ciclo de vida y borrado (tareas 4–5). El agente escribe
   `M2_LIFECYCLE_SPEC.md` y `_PLAN.md` al cerrar A (ruling 14).
3. Cierre de M2: criterios de `MILESTONE_2_PLAN.md` verificados, `ACCEPTANCE.md` con
   las corridas, versión 0.3.0, `WHATS_NEW.md`. Después el agente escribe
   `MILESTONE_3_PLAN.md` y el primer SPEC/PLAN de M3 (ruling 14).
4. M3, M4, M5 con la misma mecánica. Diseño en `PRODUCT_DESIGN.md` (ruling 3).

Pendiente aparte, sin bloquear nada: sanear lint en un PR propio (`ruff check --fix`
y `ruff format` sobre toda la base, mover `select` a `[tool.ruff.lint]`, decidir si
mypy se integra al layout plano). Hoy ninguno es gate.

## Decisiones abiertas

Ninguna. Las catorce decididas están en `DELIVERY.md`, "Rulings vigentes". Los tres
defaults de `MILESTONE_2_PLAN.md` ("Lo que queda pendiente de decidir") se adoptan
sin consulta y se registran como ruling si se usan.

## Bloqueos que dependen de un humano

- Merge de cada PR a `main` (ruling 4).
- Revisión y merge de cada plan nuevo antes de implementarlo (ruling 14).
- Reunión de dos horas con medición de recursos: sin disponibilidad (ruling 5).

## Próxima acción

Un agente con este archivo delante arranca `M2_CATALOG_PLAN.md`, tarea 1, en una
rama nueva desde `main`, y abre el PR del grupo A cuando las tres tareas tengan
evidencia. No necesita preguntar nada antes de eso.
