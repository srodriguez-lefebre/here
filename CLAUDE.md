# here — hoja de arranque para agentes

`here` graba micrófono y audio del sistema en Windows, transcribe con OpenAI y deja
sesiones recuperables. App Qt como interfaz principal, CLI secundaria, mismo núcleo.
Este archivo existe para que un agente pueda tomar el proyecto y avanzar con las
mínimas preguntas posibles. Léelo entero antes de tocar nada.

## 1. Leé en este orden

1. Este archivo.
2. `STATUS.md` — objetivo vigente, tarea en curso, decisiones abiertas, bloqueos.
3. `docs/MASTER_PLAN.md` — visión, principios y los cinco hitos. Los principios son
   la regla de desempate para todo lo demás.
4. El SPEC y el PLAN de la tarea en curso, que `STATUS.md` nombra
   (patrón `docs/development/M<n>_<tema>_SPEC.md` + `_PLAN.md`).
5. `ARCHITECTURE.md` — cuando vayas a tocar código.

Consultá solo cuando haga falta: `docs/development/PRODUCT_DESIGN.md` (diseño de
M2–M5), `docs/MILESTONE_1_PLAN.md`, `docs/LIVE_LOGO_PLAN.md`,
`docs/development/DELIVERY.md` (rulings y ledger de evidencia),
`docs/development/ACCEPTANCE.md` (matriz de aceptación), los `*_EVIDENCE.json`.

No uses como contexto nada de `docs/archive/` ni ningún archivo de `docs/` que git
ignore. Describen una etapa anterior del proyecto (MVP en WSL2) y contradicen el
estado actual.

## 2. Precedencia cuando dos documentos se contradicen

De mayor a menor autoridad:

1. Una instrucción del usuario en el chat.
2. `STATUS.md`.
3. Un ruling registrado en `docs/development/DELIVERY.md`; entre dos, gana el más reciente.
4. El SPEC/PLAN de la tarea en curso.
5. `PRODUCT_DESIGN.md`, `MILESTONE_1_PLAN.md`, `LIVE_LOGO_PLAN.md`.
6. `MASTER_PLAN.md`.

Regla general: lo más específico y más reciente gana sobre lo más general y más viejo.
Excepción: los principios de `MASTER_PLAN.md` no se pisan. Si una decisión de
cualquier nivel los contradice, no es una decisión, es una pregunta para el usuario.

`PRODUCT_DESIGN.md` se trata como decisión tomada, no como propuesta, salvo en lo que
él mismo marca como pendiente. Sus "brechas" de la sección 1 están fechadas antes del
cierre de M1; verificá contra el código antes de asumir que siguen abiertas.

## 3. Cuándo decidir solo y cuándo preguntar

Decidí solo, sin consultar, y registrá si no es trivial:

- nombres, estructura interna de módulos, orden de las tareas dentro de un PLAN;
- parámetros menores dentro de los rangos que el plan ya fija (márgenes, tiempos de
  animación, tamaños de lote, timeouts);
- cómo testear algo, siempre con datos sintéticos;
- usar una librería que ya está en `pyproject.toml`.

Preguntá antes de hacerlo, y mientras tanto hacé todo lo que no depende de la respuesta:

- cambia lo que el usuario ve o puede hacer (una pantalla, un comando, un flujo);
- agrega una dependencia, un servicio, un proceso residente, tráfico de red o una cuenta;
- toca retención, borrado, privacidad o dónde se guardan los datos;
- contradice un principio del plan maestro o la definición de terminado de un hito;
- necesita hardware, credencial real, máquina limpia o tiempo de reunión que no tenés;
- mergear a `main` (ver sección 7).

Cada decisión no trivial que tomes solo se registra en `DELIVERY.md` con el formato
que ya se usa: `Ruling: <decisión>. Cost if wrong: <qué habría que rehacer>.`

Nunca, sin importar quién lo pida:

- inventar timestamps, hablantes, nombres de dispositivos o progreso que el sistema no midió;
- llamar al proveedor sin una acción explícita del usuario (abrir la app no es una);
- leer el `.env` real, grabaciones privadas o la biblioteca del usuario en tests o evidencia;
- borrar audio salvo en la cancelación destructiva confirmada que el plan define;
- presentar como aprobado algo omitido, no corrido o fallado.

## 4. Tres niveles de evidencia

Cada vez que digas que algo está "hecho", decí cuál de los tres es:

1. **Implementado**: el código existe y la suite pasa.
2. **Verificado automáticamente**: tests, smoke y CI con datos sintéticos lo cubren.
3. **Certificado**: probado en hardware real, instalación limpia o proveedor real, con
   fecha, equipo y límites anotados en `ACCEPTANCE.md`.

Un test con un fake prueba un contrato de software, no un micrófono ni un instalador.
No cambies una definición de terminado para convertir un bloqueo de hardware en éxito.

## 5. Entorno

- Windows 11 x64. Linux solo para tests sin hardware; no es objetivo de producto.
- `uv` administra todo. CI y build usan CPython 3.13 (`uv python install 3.13.15`);
  desarrollo acepta 3.12.6 o superior.
- La suite completa necesita:
  - `QT_QPA_PLATFORM=offscreen` para los tests Qt;
  - `OPENAI_API_KEY` con cualquier valor sintético (CI usa `synthetic-test-key`);
    ningún test hace llamadas reales;
  - `pwsh` (PowerShell 7) en el PATH para `tests/test_distribution_smoke.py`; sin él,
    un test falla con `FileNotFoundError`, no es un bug del código;
  - `HERE_WHEEL_PYTHON` apuntando a un Python con el wheel instalado no editable para
    los dos tests de aceptación de wheel; sin la variable se saltean.
- Skips esperados en una máquina sin privilegio de symlinks: seis, todos en
  `test_output_session_writer.py` y `test_session_artifact_safety.py`.
- `scripts/build_windows.ps1` descarga y verifica Inno Setup 6.7.0 por su cuenta;
  no hace falta instalarlo. Necesita el Python 3.13.15 administrado por `uv`.

## 6. Comandos

Instalar dependencias bloqueadas:

```powershell
uv sync --frozen --dev
```

Suite completa, igual que CI:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"; $env:OPENAI_API_KEY = "synthetic-test-key"; uv run --no-sync pytest -vv
```

Lint y formato, solo sobre los archivos que tocaste:

```powershell
uv run ruff check <archivos>; uv run ruff format <archivos>
```

CI no corre ruff ni mypy, y la línea base no está limpia (al 2026-11-04: 53 avisos de
ruff, casi todos E501 e I001; 15 archivos que `ruff format` reescribiría; mypy sobre
`src` da 231 errores, 147 de ellos porque no resuelve el layout plano `here.*`).
La regla es no empeorar lo que tocás: ruff limpio y formateado en tus archivos, y
ningún error nuevo de mypy en ellos. Limpiar la base entera e integrar mypy al layout
es una tarea aparte, no se hace de pasada.

Wheel y source:

```powershell
uv build
```

Bundle e instalador Windows, y la aceptación del instalador:

```powershell
./scripts/build_windows.ps1 -Python (uv python find --managed-python 3.13.15)
./scripts/smoke_windows.ps1 -BundlePath ./dist/windows/here -InstallerPath ./dist/windows/here-0.2.0-windows-x64-setup.exe
```

Entradas: `here`, `record`, `here-gui` (producción) y `here-gui-preview` (UI con
eventos sintéticos, sin audio ni proveedor).

## 7. Git, PRs y merge

- Una rama por grupo de PR, nombrada `<agente>/<hito>-<tema>` (ej. `codex/m2-catalog`).
- Commits cohesivos por tema, no un squash de todo el PR.
- Nunca push directo a `main`. Siempre PR con descripción y evidencia.
- CI de GitHub corre solo en Windows. No hay gate de Linux.
- Mergear a `main` lo hace el usuario, siempre. El agente abre el PR con self-review
  y review de un agente independiente, deja la evidencia y sigue en otra rama. No se
  piden reviews a Copilot. El permiso de administrador no se usa para merges.
- Versiones: una minor por hito. M2 → 0.3.0, M3 → 0.4.0, M4 → 0.5.0, M5 → 1.0.0.
- Idioma: planes, `STATUS.md`, criterios y rulings nuevos en español; código,
  commits, PRs, specs técnicos y el resto de `docs/development/` en inglés.

## 8. Convenciones de código

- `typer` para CLI, `loguru` para logs (nunca `print`), `pydantic-settings` para config.
- Imports absolutos siempre (`from here.config.settings import ...`), nunca relativos.
- Layout `src/` plano mapeado a `here.*`; un subpaquete nuevo se registra en
  `pyproject.toml` bajo `[tool.setuptools]` y `[tool.setuptools.package-dir]`.
- `application/`, `recording/`, `transcription/`, `output/` y los futuros `memory/` e
  `intelligence/` no importan Qt ni Typer. La UI consume el contrato de aplicación y
  nunca lanza la CLI como subproceso.
- Ruff: línea 100, reglas E/F/I, comillas dobles. Mypy con la config del `pyproject`
  (ver sección 6 sobre el estado real de ambos).
- Tests con pytest y pytest-qt; datos sintéticos; DBs y raíces temporales; transporte
  de red simulado salvo smoke opt-in explícito.
- Archivos pequeños se escriben con temporal + flush + replace en el mismo volumen.
- Rutas que vienen de datos (journal, sesión, ZIP) se validan contra traversal,
  symlinks y junctions antes de leer, escribir o borrar.

## 9. Gates que solo destraba un humano

| Gate | Qué hacés mientras tanto |
|---|---|
| Máquina Windows limpia para aceptar el instalador | Está disponible a pedido. Pedila cuando una fila de aceptación la necesite, con el protocolo escrito; no antes |
| Credencial real del proveedor | Está disponible a pedido. Test con transporte simulado por defecto; el smoke real es opt-in, con audio sintético, y se pide cuando toca |
| Reunión de dos horas con medición de recursos | No hay disponibilidad. Tests de duración virtual; la fila sigue "omitida", nunca "aprobada" |
| Merge a `main` | PR abierto con evidencia; seguí con la siguiente tarea en otra rama |
| Cualquier cambio de alcance de un hito | Proponelo en `STATUS.md` como decisión abierta con tu recomendación |

## 10. Documentación con cada cambio

Se actualiza junto con el cambio, no al final:

- `README.md` si cambia instalación, configuración o uso;
- `ARCHITECTURE.md` si cambian fronteras o flujo;
- `WHATS_NEW.md` con una nota corta cuando una capacidad queda lista para usar;
- el plan del hito y `ACCEPTANCE.md` cuando cambia alcance, estado o evidencia;
- `DELIVERY.md` con cada ruling y cada checkpoint de evidencia;
- `STATUS.md` al terminar o bloquear una tarea.

Las notas de What's New explican qué cambió, por qué importa y cómo afecta al
proyecto. No son un changelog ni se escriben durante la planificación.
