# Estado de here

Actualizado el 2026-11-27. Leé `CLAUDE.md` completo, este archivo,
`docs/MASTER_PLAN.md`, y el SPEC y PLAN del grupo activo. Los rulings y
el historial de entrega están en [DELIVERY.md](docs/development/DELIVERY.md).

## M3 B en revisión de entrega

`main` sigue en 0.3.0. El [PR #30](https://github.com/srodriguez-lefebre/here/pull/30)
contiene el candidato 0.4.0. La revisión independiente y los dos workflows
Windows deben corresponder al head exacto antes del merge protegido.
Copilot se omite por ruling 64. La prueba de dos horas sigue **omitida por ruling 5**.

- **Nivel 2.** El candidato integrado `063d974` pasó 1.801 casos y omitió
  24 en `m3-b-r76-full-063d974`. La búsqueda pública dio 395,381 ms p95
  en 5+40. El heartbeat dio 13,2683 ms máximo en ocho consultas completas.
  [El paquete integrado](https://github.com/srodriguez-lefebre/here/actions/runs/46036939106)
  verificó 776 entradas, QtMultimedia, 110 controles nativos y seis smokes congelados.
- **Nivel 3.** El ensayo real `m3-b-seek-912f63f-20261127` midió las cuatro
  marcas a 1× con cota máxima de 36,245 ms. El alcance de la medición y la
  equivalencia de nivel 2 al paquete integrado figuran en
  [ACCEPTANCE.md](docs/development/ACCEPTANCE.md) y ruling 78.
- **Siguiente.** Confirmar revisión independiente y Windows CI/paquete sobre
  el head 0.4.0, verificar archivo de hashes y procedencia, ejecutar el
  merge de §7 y registrar la entrega. Antes de abrir ejecución M4 A, hacer
  el PR del contrato M4 enlazado al merge real de M3.

## Orden de grupos

| Grupo | SPEC y PLAN en `docs/development/` | Estado |
| --- | --- | --- |
| M2 A | `M2_CATALOG` | [Merge #21](https://github.com/srodriguez-lefebre/here/pull/21) |
| M2 B | `M2_LIFECYCLE` | [Merge #25](https://github.com/srodriguez-lefebre/here/pull/25), 0.3.0 |
| M3 A | `M3_SEARCH` | [Merge #28](https://github.com/srodriguez-lefebre/here/pull/28); contrato M3 B en #29 |
| M3 B | `M3_LIBRARY` | [PR #30](https://github.com/srodriguez-lefebre/here/pull/30), candidato 0.4.0 |
| M4 A | `M4_GROUNDED` | Contrato preparado y revisado; espera cierre M3 |
| M4 B | `M4_PRODUCT` | Espera contrato y ventana de solapamiento de M4 A |
| M5 A | `M5_SETTINGS` | Espera cierre M4 |
| M5 B | `M5_PORTABILITY` | Adelanto permitido verificado a nivel 2, claims esperan M4 B |
| M5 C | `M5_RELEASE` | Espera entregas previas y, en nivel 3, máquina Windows limpia |

## Ramas, chats y reanudación

Todos los worktrees están bajo `C:/Users/savar/.codex/worktrees/`. El
orquestador conserva las integraciones, los contratos compartidos, los merges
y la ventana exclusiva de suites sensibles. Migraciones, `pyproject.toml` y
versiones están en serie.

| Frente | Rama y worktree | Responsable y checkpoint |
| --- | --- | --- |
| M3 A | `codex/m3-a-r75-integration`, `m3-a-r75-integration/here` | Chat `01a1af04-e779-7363-be61-c840651b0b18` y orquestador. Merge `a1a8362` |
| M3 B | `codex/m3-library` → remoto `codex/m3-library-product`, `m3-library-group/here` | Orquestador. 34 commits preservados sobre `8d0b03d`; revisión independiente por `m5_portability_package_io_review_sol` |
| M3 B, docs/evidencia | `codex/m3-b-closeout-docs`, `m3-b-closeout-docs/here` | `m3_nov27_evidence_sol`, `fd3a4f5` |
| M4 A, contrato | `codex/m4-grounded-candidate-handoff`, `m4-grounded-candidate-handoff/here` | `m3_b_native_handoff_docs_sol`, `f10c3ec`, revisión estática independiente |
| M4 A, fixtures | `codex/m4-grounded-fixture`, `m4-grounded-fixture/here`; `codex/m4-grounded-verifier-fixtures`, `m4-grounded-verifier-fixtures/here` | `dc5e0e3` y `0c35b6b`. Reutilizar también `tests/grounded_fixtures.py` del worktree `m2-preparation/here` |
| M5 B | `codex/m5-portability-adapter-integration`, `m5-portability-adapter-integration/here` | Chat `01a1af04-b158-75a2-8933-98783880f3e8`, checkpoint `5f6324f`, GPT-6.2 Sol |

Evidencia local de M3 en
`C:/Users/savar/.codex/evidence/here/m3-native-windows-distribution-20261127/`
y `C:/Users/savar/.codex/evidence/here/m3-b-seek-063d974-20261127/`.
El baseline 0.2.0 real para el upgrade M5 sigue guardado y verificado.
