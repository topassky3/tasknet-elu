# TaskNet-ELU Φ

Marco experimental para descubrir qué información merece viajar por una red:
la mínima información útil que conserva una decisión, medida de forma honesta.

## Estado
- Fase 0: estructura y entorno (este commit).
- Fase 1: curva utilidad vs bytes en Fashion-MNIST (pendiente).

## Estructura
- `scripts/`        scripts ejecutables por fase
- `src/tasknet_elu/` módulos del proyecto
- `data/`           datos (raw/processed); no se versionan
- `results/`        tablas y figuras generadas
- `reports/`        informes y cierres por fase
- `paper/`          documentos LaTeX

## Uso (Fase 0)
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/00_phase0_check.py
```
