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
3. Actualizar `docs/development/PRODUCT_DESIGN.md` §1: marcar qué brechas ya cerró M1
   (segmentos en `TranscriptionResult`, `meeting_id` UUID, API key opcional) y cuáles
   siguen (cuatro estados literales en `session.json`, tres llamadas directas a
   `write_session_artifacts` en `cli.py`). Tachar la frase "modelo de texto ya usado"
   por el ruling 7.
4. Escribir la lista de criterios verificables de M2, con el formato de la de
   `docs/MILESTONE_1_PLAN.md`, en español.
5. Escribir `docs/development/M2_CATALOG_SPEC.md` y `M2_CATALOG_PLAN.md` (en inglés)
   para el primer grupo de PRs de M2: tareas 1–3 del diseño (modelos, migraciones,
   repositorio; segmentos y snapshots atómicos; ingestión e identidad).
6. Cerrar las dos decisiones pendientes de `docs/LIVE_LOGO_PLAN.md` con el ruling 8.
7. Reordenar `DELIVERY.md`: rulings vigentes arriba (hecho), historial como ledger
   append-only debajo con un encabezado que lo diga.
8. Sanear lint en un PR aparte: `ruff check --fix` y `ruff format` sobre toda la base
   (53 avisos, 15 archivos), mover `select` a `[tool.ruff.lint]`, y decidir si mypy se
   integra al layout plano o se saca de las convenciones. Hoy ninguno es gate.

## Acciones decididas que esperan confirmación del usuario

Son irreversibles o tocan el remoto; el agente no las ejecuta solo.

- Borrar `codex/ui-configuration-errors` en local y en `origin` (ruling 2).
- Eliminar `.github/workflows/ci-ubuntu-manual.yml` y sus referencias en un PR aparte
  (ruling 11).

## Bloqueos que dependen de un humano

- Reunión de dos horas con medición de recursos: sin disponibilidad (ruling 5).
- Máquina Windows limpia y credencial real: disponibles a pedido; el agente las pide
  cuando una fila de aceptación las necesite, no antes.
- Merge de cualquier PR a `main`: siempre el usuario (ruling 4).

## Próxima acción sugerida

Tareas 3 a 7 de arriba, en ese orden, en el mismo PR #16 o en uno siguiente. Ninguna
requiere preguntar nada.
