# Estado de here

Actualizado el 2026-11-27. Leé `CLAUDE.md`, este archivo,
`docs/MASTER_PLAN.md` y el SPEC y PLAN del grupo activo. Los rulings y
la evidencia histórica están en [DELIVERY.md](docs/development/DELIVERY.md).

## M3 cerrado en 0.4.0. M4 A en ejecución

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

El [PR documental #31](https://github.com/srodriguez-lefebre/here/pull/31)
se mergeó a las 14:33 Montevideo en `5792a080544688e7d7da59056128748c28f276b7`
desde el head revisado `f900d11`. [CI 46041541216](https://github.com/srodriguez-lefebre/here/actions/runs/46041541216)
y [paquete 46041541205](https://github.com/srodriguez-lefebre/here/actions/runs/46041541205)
pasaron sobre ese head, nivel 2. La anotación de merge en `DELIVERY.md`
está en `4641742` para llevar a la entrega M4 A.

El chat M4 A tiene el brief de T1 sobre el merge exacto del contrato y la
ventana Windows para los RED de modelos y recuperación, en serie. M5 B
devolvió la ventana sin procesos residuales. Nivel 2 RED en `15b5048`,
cuatro pases y un fallo causal en cita histórica del WAV legado. Nivel 2
RED en `35c1fa0`, un pase y cuatro fallos causales de preflight R50 en
captura y publicación de ZIP y Markdown. M5 B escribe GREEN de R50 sin
ejecución local hasta reasignar la ventana. Claims esperan M4 B.

## Orden de grupos

| Grupo | SPEC y PLAN en `docs/development/` | Estado |
| --- | --- | --- |
| M2 A | `M2_CATALOG` | [Merge #21](https://github.com/srodriguez-lefebre/here/pull/21) |
| M2 B | `M2_LIFECYCLE` | [Merge #25](https://github.com/srodriguez-lefebre/here/pull/25), 0.3.0 |
| M3 A | `M3_SEARCH` | [Merge #28](https://github.com/srodriguez-lefebre/here/pull/28), contrato #29 |
| M3 B | `M3_LIBRARY` | [Merge #30](https://github.com/srodriguez-lefebre/here/pull/30), 0.4.0 |
| M4 A | `M4_GROUNDED` | Contrato [merge #31](https://github.com/srodriguez-lefebre/here/pull/31), T1 RED despachado |
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
| M4 A, grupo | `codex/m4-grounded`, `m4-grounded-group/here` | Chat `01a1af04-e779-7363-be61-c840651b0b18`, GPT-6.2 Sol, base `5792a08`, responsable de T1–T3 |
| M4 A, RED T1 | `codex/m4-grounded-models-red`, `m4-grounded-models-red/here`; `codex/m4-grounded-retrieval-red`, `m4-grounded-retrieval-red/here` | Chat M4 A con escritores separados. Fixtures una sola vez y tests de modelos en `dc94be6` |
| M4, ledger del contrato | `codex/m4-delivery-ledger`, `m4-delivery-ledger/here` | `4641742`, orquestador |
| M4 A, fixtures | `codex/m4-grounded-fixture`, `m4-grounded-fixture/here`; `codex/m4-grounded-verifier-fixtures`, `m4-grounded-verifier-fixtures/here` | `dc5e0e3` y `38bdd2e`. El chat transfirió `tests/grounded_fixtures.py` para T1. Transferir el blob final de verificador una sola vez para T3 |
| M4 B, preparación documental | `codex/m4-product-contract-prep`, `m4-product-contract-prep/here` | `m3_b_native_handoff_docs_sol`, `041c892`, revisión estática independiente. Sin producto |
| M5 B, core y adaptadores | `codex/m5-portability-adapter-integration`, `m5-portability-adapter-integration/here` | Chat `01a1af04-b158-75a2-8933-98783880f3e8`, GPT-6.2 Sol, checkpoint `5f6324f` |
| M5 B, mapa de traslado | `codex/m5-b-post-m3-prep`, `m5-b-post-m3-prep/here` | Mismo chat, `c643b1e`, cruces R50 y claims |
| M5 B, traslado sobre 0.4.0 | `codex/m5-b-post-m3-integration`, `m5-b-post-m3-integration/here` | Chat `01a1af04-b158-75a2-8933-98783880f3e8`, `ff6805d` revisado estáticamente sobre `7239ba6` |
| M5 B, RED de cruces | `codex/m5-b-m3-cross-group-red`, `m5-b-m3-cross-group-red/here`; `codex/m5-b-read-preflight-red`, `m5-b-read-preflight-red/here` | Chat M5 B, RED medidos `15b5048` y `35c1fa0` sobre `ff6805d` |
| M5 B, GREEN de R50 | `codex/m5-b-read-preflight-green`, `m5-b-read-preflight-green/here` | Chat M5 B, lease exclusivo de `src/memory/service.py`. Audio legado espera este cambio y otro lease |
| M5 B, fixture legado | `codex/m5-b-cross-group-fixtures`, `m5-b-cross-group-fixtures/here` | Mismo chat, `8d70d6f`, dos revisiones sintéticas con WAV legado |

Evidencia local de M3 en
`C:/Users/savar/.codex/evidence/here/m3-native-windows-distribution-20261127/`
y `C:/Users/savar/.codex/evidence/here/m3-b-seek-063d974-20261127/`.
El baseline real 0.2.0 para el upgrade M5 sigue guardado y verificado.
