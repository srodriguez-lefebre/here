# Estado de here

Actualizado: 2026-11-05. Este archivo se reescribe; el historial y los rulings viven en
`docs/development/DELIVERY.md`. Leé `CLAUDE.md` primero.

## Objetivo vigente

**El plan de M2 a M5 está completo. No hay implementación en curso ni autorizada**
(ruling 15). Cualquier implementación arranca solo con una instrucción explícita del
usuario; cuando llegue, el punto de partida es `docs/MILESTONE_2_PLAN.md` y
`docs/development/M2_CATALOG_PLAN.md`, tarea 1, en una rama nueva desde `main`.

## Dónde está el código

- `origin/main` = `cd3dfc9`, v0.2.0, con los PRs #16, #17 y #18 mergeados.
- Suite en `main`: 741 pasan, 8 skips esperados, 1 falla local por falta de `pwsh`.
- Omitido de M1 y que sigue omitido: la reunión de dos horas con medición de
  recursos (ruling 5). Máquina limpia y credencial real están disponibles a pedido.
- Worktrees viejos en disco con ramas ya mergeadas, sin tocar:
  `.codex/worktrees/five-milestones`, `here-worktrees/live-logo-overlay`,
  `here-worktrees/windows-app-core`. Candidatos a limpieza; no es urgente.

## El plan completo

| Hito | Plan (español, criterios) | Grupos de PR (SPEC + PLAN en inglés) | Versión al cerrar |
|---|---|---|---|
| M2 Memoria estructurada | `docs/MILESTONE_2_PLAN.md`, 16 criterios | A `M2_CATALOG_*` (tareas 1–3) · B `M2_LIFECYCLE_*` (4–5) | 0.3.0 |
| M3 Memoria navegable | `docs/MILESTONE_3_PLAN.md`, 18 criterios | A `M3_SEARCH_*` (1–2) · B `M3_LIBRARY_*` (3–4) | 0.4.0 |
| M4 Inteligencia | `docs/MILESTONE_4_PLAN.md`, 18 criterios | A `M4_GROUNDED_*` (1–3) · B `M4_PRODUCT_*` (4–5) | 0.5.0 |
| M5 Producto cotidiano | `docs/MILESTONE_5_PLAN.md`, 20 criterios | A `M5_SETTINGS_*` (1–2) · B `M5_PORTABILITY_*` (3) · C `M5_RELEASE_*` (4–5) | 1.0.0 |

Nueve grupos de PR en total, cada uno con su SPEC y su PLAN en `docs/development/`.
Todos marcados "not in execution (ruling 15)". Orden: M2 A, M2 B, M3 A, M3 B, M4 A,
M4 B, M5 A, M5 B, M5 C, con los solapamientos que permite `CLAUDE.md` §11. Cada grupo
es un PR que el agente mergea solo cuando CI está verde, Copilot revisó y fue
respondido, y un subagente independiente revisó (rulings 16 y 17). Si al mergear un
grupo cambian contratos que el siguiente asume, el agente ajusta el PLAN siguiente en
un PR de documentación antes de empezarlo (ruling 14). El ejecutor previsto es Codex,
que entra por `AGENTS.md` (ruling 18).

Cada plan de hito tiene una sección "Lo que queda pendiente de decidir" con un default
por ítem. Los defaults se adoptan sin consulta y se registran como ruling al usarse.

## Hallazgos de la planificación que un implementador debe conocer

Salieron al escribir los planes, están anotados en el plan del hito correspondiente:

- `packaging/windows.spec` excluye `PySide6.QtMultimedia` y filtra sus plugins. M3
  los necesita para reproducir audio; M3 grupo B los agrega y M5 grupo C lo verifica.
- La versión `0.2.0` está hardcodeada en `packaging/installer.iss`,
  `scripts/build_windows.ps1` y `scripts/smoke_windows.ps1`. M5 grupo C la lee de
  `pyproject.toml`; hasta entonces cada cierre de hito la cambia a mano en los tres.
- No hay ningún sink de `loguru` configurado hoy; M5 grupo A agrega
  `src/logging_setup.py` llamado desde `src/entrypoints.py`.
- El instalador de M1 rechaza reemplazar una app en ejecución a propósito; el "cierre
  seguro" del diseño se cumple con `Salir`, no con el instalador.
- El lock de captura entre CLI y GUI va como lock de archivo del sistema operativo en
  la capa de aplicación, no `QLockFile`, porque los servicios no importan Qt.
- La reproducción nunca arranca sola desde ninguna fuente; M3 no expone un punto de
  entrada de autoplay y M4 no lo necesita.

## Tarea en curso

Ninguna. Pendiente aparte, sin bloquear nada: sanear lint en un PR propio
(`ruff check --fix` y `ruff format` sobre toda la base, mover `select` a
`[tool.ruff.lint]`, decidir si mypy se integra al layout plano).

## Decisiones abiertas

Ninguna. Las quince decididas están en `DELIVERY.md`, "Rulings vigentes".

## Bloqueos que dependen de un humano

- Instrucción explícita para empezar a implementar (ruling 15).
- Máquina Windows limpia y credencial real, a pedido, cuando una fila de aceptación
  las necesite (ruling 5).
- Reunión de dos horas con medición de recursos: sin disponibilidad (ruling 5).
- Cambios al ruleset o a los permisos del repositorio.

## Próxima acción

Esperar la instrucción del usuario para empezar. Cuando llegue, el agente la registra
como ruling, arranca `M2_CATALOG_PLAN.md` tarea 1 en una rama nueva desde `main`, y
trabaja como orquestador según `CLAUDE.md` §11 hasta entregar 1.0.0.
