# Reproducibilidad de TaskNet-ELU v0.1.0

## Alcance

Esta release permite inspeccionar los scripts, tablas, figuras e informes que sustentan el preprint. Debe distinguirse entre:

- **reproducción de artefactos**: verificar archivos, rutas, metadatos, cifras principales y trazabilidad;
- **repetición completa**: volver a entrenar los modelos y ejecutar todas las fases.

La segunda puede producir pequeñas diferencias por plataforma, versiones de bibliotecas, paralelismo y operaciones no completamente deterministas.

## Datos

Fashion-MNIST y CIFAR-10 no se redistribuyen. Los scripts los descargan mediante `torchvision` o usan la caché local. Los datos conservan sus licencias originales.

## Fases y artefactos

- `scripts/01_utility_per_bit_fashion_mnist.py`: Fase 1.
- `scripts/02_learned_vs_classic.py`: Fase 2.
- `scripts/03_cifar10_validation_v3.py`: Fase 3.
- `scripts/04_elu_threshold_policy.py`: Fase 4.
- `results/tables/`: resultados tabulares versionados.
- `results/figures/`: figuras versionadas.
- `reports/`: informes y cierres por fase.

Las corridas principales declaradas utilizaron tres semillas explícitas, CPU y PyTorch. Las cifras exactas deben contrastarse con los CSV versionados.

## Entorno

`requirements.txt` expresa dependencias, pero no conserva necesariamente las versiones históricas exactas de las corridas originales. No deben inferirse ni inventarse retrospectivamente.

Para registrar un nuevo entorno de validación:

```bash
python --version
python -m pip freeze > environment/validation-environment.txt
```

El archivo generado debe incluir esta advertencia:

> Este archivo describe el entorno utilizado para validar la release y no necesariamente el entorno histórico exacto de las corridas originales.

## Verificación

```bash
python scripts/00_phase0_check.py
python scripts/05_verify_release.py --check
python scripts/05_verify_release.py --write-manifest
```

El manifiesto SHA-256 se guarda en `release/MANIFEST.sha256` y debe generarse nuevamente cuando cambie cualquier artefacto incluido.

## Limitaciones

- No se mide energía física ni latencia real de red.
- El embedding está entrenado para la tarea.
- Los parámetros del cuantizador se asumen compartidos.
- El bit de señalización transmitir/callar no se contabiliza.
- La política de Fase 4 usa un score softmax sin calibración explícita.
- Los resultados no se generalizan automáticamente a redes reales.
