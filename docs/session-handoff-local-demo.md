# Helix Local Demo Handoff

## Repo

- Workspace principal: `C:\Users\Big Duck\proyectos\helix-backend-repo`
- Fecha de referencia: `2026-03-26`

## Directorios en uso

- Codigo principal: `C:\Users\Big Duck\proyectos\helix-backend-repo\src`
- API y runtime: `C:\Users\Big Duck\proyectos\helix-backend-repo\src\helix_proto`
- Frontend demo: `C:\Users\Big Duck\proyectos\helix-backend-repo\frontend`
- Scripts de arranque: `C:\Users\Big Duck\proyectos\helix-backend-repo\scripts`
- Workspace de modelos: `C:\Users\Big Duck\proyectos\helix-backend-repo\workspace`
- GGUF local: `C:\Users\Big Duck\proyectos\helix-backend-repo\workspace-gguf`
- Herramientas llama.cpp: `C:\Users\Big Duck\proyectos\helix-backend-repo\tools`
- Benchmarks y salidas: `C:\Users\Big Duck\proyectos\helix-backend-repo\benchmark-output`
- Tests: `C:\Users\Big Duck\proyectos\helix-backend-repo\tests`
- Documentacion: `C:\Users\Big Duck\proyectos\helix-backend-repo\docs`

## Archivo/modelo clave

- GGUF activo: `C:\Users\Big Duck\proyectos\helix-backend-repo\workspace-gguf\qwen35-4b-q4_k_m.gguf`
- Alias del modelo: `qwen35-4b-q4`
- Config de asistentes: `C:\Users\Big Duck\proyectos\helix-backend-repo\workspace\assistants.json`
- Registro del alias: `C:\Users\Big Duck\proyectos\helix-backend-repo\workspace\models\qwen35-4b-q4\model_info.json`

## Estado funcional actual

- Los tres asistentes `general`, `code` y `legal` apuntan al alias `qwen35-4b-q4`.
- Existe endpoint SSE de streaming en `POST /assistants/chat/stream`.
- El frontend demo es un unico archivo en `frontend/index.html`.
- La UI fue pulida para demo local: layout mas cuidado, prompts rapidos, estados de streaming, chat mas limpio.
- El flujo bueno del assistant GGUF usa prompt plano para evitar respuestas tipo "Thinking Process".
- Se hicieron imports lazy en CDNA/substrate para que el backend no obligue a tener `torch` solo por arrancar la API.

## Archivos principales tocados

- `C:\Users\Big Duck\proyectos\helix-backend-repo\src\helix_proto\api.py`
- `C:\Users\Big Duck\proyectos\helix-backend-repo\src\helix_proto\assistants.py`
- `C:\Users\Big Duck\proyectos\helix-backend-repo\src\helix_proto\cdna.py`
- `C:\Users\Big Duck\proyectos\helix-backend-repo\src\helix_proto\workspace.py`
- `C:\Users\Big Duck\proyectos\helix-backend-repo\frontend\index.html`
- `C:\Users\Big Duck\proyectos\helix-backend-repo\scripts\run-local-backend.ps1`
- `C:\Users\Big Duck\proyectos\helix-backend-repo\scripts\run-local-frontend.ps1`
- `C:\Users\Big Duck\proyectos\helix-backend-repo\scripts\start-local-demo.ps1`

## Comandos para levantar

### PowerShell

Backend:

```powershell
Set-Location "C:\Users\Big Duck\proyectos\helix-backend-repo"
$env:PYTHONPATH = "src"
python -m helix_proto.cli serve-api --workspace-root workspace --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
Set-Location "C:\Users\Big Duck\proyectos\helix-backend-repo"
python -m http.server 3000 --bind 127.0.0.1 --directory frontend
```

### CMD / Anaconda Prompt

Backend:

```bat
cd /d C:\Users\Big Duck\proyectos\helix-backend-repo
set PYTHONPATH=src
python -m helix_proto.cli serve-api --workspace-root workspace --host 127.0.0.1 --port 8000
```

Frontend:

```bat
cd /d C:\Users\Big Duck\proyectos\helix-backend-repo
python -m http.server 3000 --bind 127.0.0.1 --directory frontend
```

## URLs de uso

- Frontend: `http://127.0.0.1:3000`
- API asistentes: `http://127.0.0.1:8000/assistants`
- Stream SSE: `http://127.0.0.1:8000/assistants/chat/stream`

## Smoke test rapido

Archivo ya preparado:

- `C:\Users\Big Duck\proyectos\helix-backend-repo\stream-body.json`

Comando:

```bat
curl.exe -N http://127.0.0.1:8000/assistants/chat/stream -H "Content-Type: application/json" --data-binary "@stream-body.json"
```

## Bloqueos reales actuales

1. `llama-cpp-python` no se pudo instalar en el env conda nuevo sin compilador C/C++ en Windows.
2. En `Python 3.13` y tambien en `3.12`, `pip` quiso compilar desde source y fallo por falta de `nmake`/compiler.
3. El repo ya no pide `torch` solo para arrancar la API, pero sigue necesitando un entorno donde exista `llama_cpp` si se quiere usar el backend GGUF por Python.

## Lo que ya fue validado

- `frontend/index.html` responde bien servido con `python -m http.server`.
- `GET /assistants` respondio `200`.
- El stream SSE de assistants emitio tokens validos contra el backend correcto en sesiones anteriores.
- El modelo base esta configurado en el workspace y el alias existe.

## Siguiente paso recomendado

La forma mas corta de seguir es una de estas dos:

1. Usar una instalacion de Python/venv que ya tenga `llama_cpp` funcionando y levantar el backend desde ahi.
2. Implementar un fallback de demo usando `tools\llama-b8502\llama-cli.exe` como backend local sin depender de `llama-cpp-python`.

## Notas utiles para otra sesion

- En PowerShell no usar `cd /d ...`; usar `Set-Location "ruta con espacios"`.
- En PowerShell no usar `set PYTHONPATH=src`; usar `$env:PYTHONPATH = "src"`.
- El warning `n_ctx_seq (4096) < n_ctx_train (262144)` es informativo, no un error.
- El objetivo actual no es frontend complejo ni features nuevas; es demo local usable.
