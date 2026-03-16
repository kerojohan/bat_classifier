# bat_classifier

Servicio para clasificar llamadas de murciélagos a partir de un audio completo.
Acepta un audio entero, detecta automáticamente candidatos acústicos o usa un
intervalo manual, recorta el clip y devuelve un JSON con metadatos e inferencia.

## Características

- API HTTP con `FastAPI`
- CLI para procesamiento local
- Auto-detección de candidatos por energía
- Backend de clasificación configurable por comando externo
- Soporte opcional para `BatDetect2`
- Salida JSON pensada para pipelines

## Licencia

Este proyecto se publica bajo `CC BY-NC 4.0`.

- Uso no comercial únicamente
- Atribución obligatoria al repositorio `bat_classifier`
- Ver [LICENSE](LICENSE) y [NOTICE](NOTICE)

Nota: esto es software source-available, no una licencia open source aprobada por OSI.

## Requisitos

- Python `>=3.10`
- `ffmpeg` y `ffprobe` disponibles en el sistema

## Instalación

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
```

## Ejecución por CLI

Con auto-detección:

```bash
export BAT_CLASSIFIER_COMMAND="/ruta/al/backend"
python -m app.cli /ruta/audio.wav
```

Con intervalo manual:

```bash
export BAT_CLASSIFIER_COMMAND="/ruta/al/backend"
python -m app.cli /ruta/audio.wav --start 12.4 --end 13.1
```

Con conservación del clip extraído:

```bash
export BAT_CLASSIFIER_COMMAND="/ruta/al/backend"
export BAT_KEEP_ARTIFACTS=true
python -m app.cli /ruta/audio.wav --keep-artifacts
```

## Ejecución por API

```bash
export BAT_CLASSIFIER_COMMAND="/ruta/al/backend"
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Petición con auto-detección:

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -F "audio=@/ruta/audio.wav"
```

Petición con intervalo manual:

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -F "audio=@/ruta/audio.wav" \
  -F "start_seconds=12.4" \
  -F "end_seconds=13.1"
```

## Backend real opcional: BatDetect2

El adaptador incluido ejecuta `batdetect2.api.process_file()` y normaliza su
salida al JSON de este proyecto. El entrypoint es:

```bash
python -m app.run_batdetect2_backend
```

Limitaciones:

- `BatDetect2` está entrenado para especies del Reino Unido
- Requiere validación local antes de uso ecológico serio
- Su propia licencia también es no comercial

Fuentes:

- https://github.com/macaodha/batdetect2
- https://github.com/macaodha/batdetect2/blob/main/LICENSE.md
- https://raw.githubusercontent.com/macaodha/batdetect2/main/batdetect2/api.py

Entorno local dedicado:

```bash
python3.10 -m venv .venv-batdetect2
. .venv-batdetect2/bin/activate
pip install -e .
pip install batdetect2
export BAT_CLASSIFIER_COMMAND="/ruta/al/proyecto/.venv-batdetect2/bin/python -m app.run_batdetect2_backend"
python -m app.cli /ruta/audio.wav
```

Docker:

```bash
docker build -f Dockerfile.batdetect2 -t bat-classifier-batdetect2 .
docker run --rm -p 8000:8000 bat-classifier-batdetect2
```

## Contrato del backend

El comando configurado en `BAT_CLASSIFIER_COMMAND` debe:

- leer JSON desde `stdin`
- escribir JSON a `stdout`

Entrada enviada por este servicio:

```json
{
  "clip_path": "/tmp/bat-classifier/bat-123/clip.wav",
  "original_filename": "grabacion.wav",
  "start_seconds": 12.4,
  "end_seconds": 13.1,
  "sample_rate_hz": 384000,
  "duration_seconds": 0.7
}
```

Variable opcional:

```bash
BAT_KEEP_ARTIFACTS=true
```

## Ejemplo de salida

```json
{
  "request": {
    "filename": "grabacion.wav",
    "start_seconds": 4.492,
    "end_seconds": 5.792
  },
  "audio": {
    "duration_seconds": 600.702,
    "sample_rate_hz": 250000,
    "channels": 1,
    "codec": "pcm_s16le"
  },
  "detection": {
    "mode": "auto",
    "candidates": [
      {
        "start_seconds": 4.492,
        "end_seconds": 5.792,
        "score": 0.9849
      }
    ]
  },
  "clip": {
    "duration_seconds": 1.3,
    "sample_rate_hz": 250000,
    "channels": 1,
    "frame_count": 325000,
    "peak_amplitude": 0.9999,
    "rms_amplitude": 0.3096
  },
  "classifier": {
    "species": "Pipistrellus pipistrellus",
    "confidence": 0.047508,
    "top_k": [
      {
        "species": "Pipistrellus pipistrellus",
        "confidence": 0.047508
      }
    ]
  }
}
```

## Tests

```bash
python -m unittest discover -s tests -v
```
