# AGENTS.md

Leé `CLAUDE.md` entero antes de hacer nada. Es la hoja de arranque de este repositorio
y manda sobre este archivo. Después `STATUS.md`, después `docs/MASTER_PLAN.md`, después
el SPEC y el PLAN de la tarea en curso que `STATUS.md` nombra.

Resumen para no perder tiempo, con la sección de `CLAUDE.md` que lo detalla:

- Precedencia entre documentos y cuándo preguntar: §2 y §3. Los principios de
  `docs/MASTER_PLAN.md` no se pisan.
- Entorno y comandos: §5 y §6. La suite necesita `QT_QPA_PLATFORM=offscreen`, una
  `OPENAI_API_KEY` sintética y `pwsh`.
- Git y merge: §7. Mergeás tus propios PRs con CI verde y review de un subagente
  independiente. Para los PR pendientes se omite Copilot por ruling 64.
- Orquestación con subagentes y qué se puede solapar: §11.
- Rulings vigentes: `docs/development/DELIVERY.md`, sección "Rulings vigentes".

No uses como contexto nada de `docs/archive/` ni de los archivos de `docs/` que git
ignora.
