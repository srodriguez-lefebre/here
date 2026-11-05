# Estado de here

Actualizado: 2026-11-05. Leé CLAUDE.md primero. Historial y rulings:
`docs/development/DELIVERY.md`.

## Objetivo vigente

Implementar y entregar M2–M5 completos por instrucción explícita del usuario
(ruling 20, reemplaza ruling 15). Conservar definiciones de terminado y los tres
niveles de evidencia. M5 exige distribución Windows 1.0.0 y aceptación real.

## Código y secuencia

- Base: `origin/main` = `878e7d4`, v0.2.0, PR #19 mergeado.
- Grupo actual: `codex/m2-catalog`, worktree
  `C:/Users/savar/.codex/worktrees/m2-catalog/here`.
- Orden: M2 A (`M2_CATALOG`, 1–3), M2 B (`M2_LIFECYCLE`, 4–5),
  M3 A (`M3_SEARCH`, 1–2), M3 B (`M3_LIBRARY`, 3–4),
  M4 A (`M4_GROUNDED`, 1–3), M4 B (`M4_PRODUCT`, 4–5),
  M5 A (`M5_SETTINGS`, 1–2), M5 B (`M5_PORTABILITY`, 3),
  M5 C (`M5_RELEASE`, 4–5). Versiones: 0.3.0, 0.4.0, 0.5.0, 1.0.0.
- Solo solapamientos de CLAUDE §11; migraciones, pyproject y versión en serie.
- Cada tarea: RED por subagente → GREEN → review independiente → docs/evidencia
  por otro subagente. Cada grupo: PR con evidencia/review → Copilot recibido y
  respondido sobre head actual → CI Windows verde → merge commit con
  `--admin --match-head-commit` → ledger. Ningún gate se omite.

## Tarea en curso

`docs/development/M2_CATALOG_PLAN.md`, tarea 1: modelos, migraciones y repositorio.
SPEC: `docs/development/M2_CATALOG_SPEC.md`. Hito: `docs/MILESTONE_2_PLAN.md`.
Todavía sin implementación ni evidencia M2.

En paralelo: tests de segmentos (tarea 2, archivos disjuntos), preparación de
M2 B y fixtures sintéticos. Tarea 3 espera sus contratos.

## Decisiones abiertas y bloqueos

- Sin decisiones de producto abiertas; defaults de los PLAN se registran al usarse.
- PR #20 (`docs/ruling-19-smoke-key`) abierto. La autorización de credencial fue
  reiterada en la instrucción actual; M2 no la necesita.
- Máquina limpia disponible a pedido: solicitar con protocolo cuando corresponda.
- Prueba de dos horas: **omitida por ruling 5**, nunca aprobada.
- No tocar ruleset, permisos ni grabaciones privadas.

## Próxima acción

Completar RED/GREEN/review/docs de tarea 1 y el grupo M2 A, abrir PR con suite y
evidencia. En una reanudación consultar el ledger local
`.superpowers/sdd/M2_CATALOG_PLAN/progress.md` del worktree y git log antes de
redistribuir tareas.
