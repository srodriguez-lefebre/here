# Estado de here

Actualizado el 2026-11-27. Leé `CLAUDE.md`, este archivo,
`docs/MASTER_PLAN.md` y el SPEC y PLAN del grupo activo. Los rulings y
la evidencia histórica están en [DELIVERY.md](docs/development/DELIVERY.md).

## M3 cerrado en 0.4.0. Siguiente, contrato documental de M4 A

El [PR #30](https://github.com/srodriguez-lefebre/here/pull/30) entró a
`main` como `7239ba64e03e7e3230b36b5d8d952cefbbe54e47` el 27 de noviembre
a las 14:21 Montevideo. El head revisado y mergeado fue `d29f83a`. Copilot
se omite por ruling 64. La prueba de dos horas sigue **omitida por ruling 5**.

- **Nivel 2.** Windows CI de 0.4.0 pasó 1.720 casos y omitió 105, con
  credencial sintética y Qt offscreen. El paquete exacto verificó 776
  entradas, QtMultimedia, 110 controles nativos y seis smokes congelados.
  [CI](https://github.com/srodriguez-lefebre/here/actions/runs/46039586001)
  [Paquete Windows](https://github.com/srodriguez-lefebre/here/actions/runs/46039585947/artifacts/15219935874)
- **Nivel 2.** La búsqueda pública integrada dio 395,381 ms p95 en 5+40.
  El heartbeat Qt fue 13,2683 ms máximo en ocho consultas completas.
- **Nivel 3.** El ensayo `m3-b-seek-912f63f-20261127` midió cuatro marcas
  a 1× con cota máxima 36,245 ms. La atribución del render y la equivalencia
  de reproducción de nivel 2 en los paquetes sucesores están registradas
  en [ACCEPTANCE.md](docs/development/ACCEPTANCE.md) y ruling 78.

El contrato M4 A vincula el producto M3 aceptado a las operaciones de
recuperación y revalidación de evidencia de ruling 79. El orquestador inicia
T1 una vez mergeado este PR documental con §7. El primer paso es RED
independiente contra M4_GROUNDED, con los fixtures preparados. M5 B sigue
adelantado dentro de §11, con integración de claims después de M4 B.

## Orden de grupos

| Grupo | SPEC y PLAN en `docs/development/` | Estado |
| --- | --- | --- |
| M2 A | `M2_CATALOG` | [Merge #21](https://github.com/srodriguez-lefebre/here/pull/21) |
| M2 B | `M2_LIFECYCLE` | [Merge #25](https://github.com/srodriguez-lefebre/here/pull/25), 0.3.0 |
| M3 A | `M3_SEARCH` | [Merge #28](https://github.com/srodriguez-lefebre/here/pull/28), contrato #29 |
| M3 B | `M3_LIBRARY` | [Merge #30](https://github.com/srodriguez-lefebre/here/pull/30), 0.4.0 |
| M4 A | `M4_GROUNDED` | PR documental de contrato, producto por iniciar |
| M4 B | `M4_PRODUCT` | Espera contrato y ventana de solapamiento de M4 A |
| M5 A | `M5_SETTINGS` | Espera cierre M4 |
| M5 B | `M5_PORTABILITY` | Adelanto permitido de nivel 2, claims esperan M4 B |
| M5 C | `M5_RELEASE` | Espera entregas previas y, a nivel 3, máquina Windows limpia |

## Ramas, chats y reanudación

Worktrees bajo `C:/Users/savar/.codex/worktrees/`. El orquestador integra,
coordina contratos compartidos, hace los merges y controla la ventana de
suites sensibles. Migraciones, `pyproject.toml` y versiones van en serie.

| Frente | Rama y worktree | Responsable y checkpoint |
| --- | --- | --- |
| M3 A, cerrado | `codex/m3-a-r75-integration`, `m3-a-r75-integration/here` | Chat `01a1af04-e779-7363-be61-c840651b0b18` y orquestador. Merge `a1a8362` |
| M3 B, cerrado | `codex/m3-library` → remoto `codex/m3-library-product`, `m3-library-group/here` | Orquestador, review `m5_portability_package_io_review_sol`, merge `7239ba6` |
| M4 A, contrato | `codex/m4-grounded-accepted-handoff`, `m4-grounded-accepted-handoff/here` | Orquestador, docs `m3_b_native_handoff_docs_sol`; base `7239ba6` y preparación `f10c3ec` |
| M4 A, fixtures | `codex/m4-grounded-fixture`, `m4-grounded-fixture/here`; `codex/m4-grounded-verifier-fixtures`, `m4-grounded-verifier-fixtures/here` | `dc5e0e3` y `0c35b6b`. Transferir también `tests/grounded_fixtures.py` del worktree `m2-preparation/here` |
| M5 B, core y adaptadores | `codex/m5-portability-adapter-integration`, `m5-portability-adapter-integration/here` | Chat `01a1af04-b158-75a2-8933-98783880f3e8`, GPT-6.2 Sol, checkpoint `5f6324f` |
| M5 B, mapa de traslado | `codex/m5-b-post-m3-prep`, `m5-b-post-m3-prep/here` | Mismo chat, `c643b1e`, cruces R50 y claims |
| M5 B, traslado sobre 0.4.0 | `codex/m5-b-post-m3-integration`, `m5-b-post-m3-integration/here` | Chat `01a1af04-b158-75a2-8933-98783880f3e8`, despacho sobre `7239ba6`, sin claims ni cambios de versión |
| M5 B, fixture legado | `codex/m5-b-cross-group-fixtures`, `m5-b-cross-group-fixtures/here` | Mismo chat, `8d70d6f`, dos revisiones sintéticas con WAV legado |

Evidencia local de M3 en
`C:/Users/savar/.codex/evidence/here/m3-native-windows-distribution-20261127/`
y `C:/Users/savar/.codex/evidence/here/m3-b-seek-063d974-20261127/`.
El baseline real 0.2.0 para el upgrade M5 sigue guardado y verificado.
