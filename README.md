# TaskNet-ELU

**Un protocolo experimental de utilidad por byte para comunicación orientada a tarea.**

> Si un dato no mejora la decisión más de lo que cuesta enviarlo, ese dato no debería viajar.

TaskNet-ELU estudia cuánta información debe transmitirse para conservar una decisión. El repositorio compara datos originales, compresión clásica, reducciones de resolución, representaciones aprendidas, decisiones ya tomadas, silencio y políticas de transmisión selectiva sobre Fashion-MNIST y CIFAR-10.

La contribución es un **protocolo experimental reproducible** para medir utilidad frente a bytes transmitidos. El trabajo no presenta una teoría general nueva de comunicaciones.

## Preprint

- **Versión:** `v0.1.0`
- **Estado editorial:** preprint técnico, no revisado por pares
- **Autor:** Juan Felipe Orozco
- **Afiliación declarada:** Investigación independiente — Ingeniería de Telecomunicaciones
- **DOI:** pendiente de publicación en Zenodo
- **Documento:** [`paper/TaskNet-ELU-preprint-v0.1.0.pdf`](paper/TaskNet-ELU-preprint-v0.1.0.pdf)
- **Fuente LaTeX:** [`paper/TaskNet-ELU-preprint-v0.1.0.tex`](paper/TaskNet-ELU-preprint-v0.1.0.tex)

## Resultados principales

Bajo el catálogo, los modelos y el protocolo evaluados:

- el *embedding* conservó la exactitud del dato completo con aproximadamente **90,2 % menos bytes** en Fashion-MNIST;
- en CIFAR-10 conservó utilidad con aproximadamente **97,4 % menos bytes** que el PNG original;
- obtuvo aproximadamente **7,9×** y **19,1×** más utilidad por byte que el mejor método clásico que conservó utilidad en Fashion-MNIST y CIFAR-10, respectivamente;
- la política ELU redujo **92,0 %** los bytes frente a transmitir siempre el original, con una pérdida inferior a tres puntos de exactitud;
- estos resultados son evidencia experimental acotada y no una garantía universal.

## Estructura

```text
paper/             preprint en PDF y fuente LaTeX
scripts/           experimentos y verificadores de release
results/tables/    resultados tabulares en CSV
results/figures/   figuras generadas
reports/           informes y cierres de cada fase
docs/              reproducibilidad y publicación en Zenodo
release/           notas y manifiesto SHA-256
```

## Instalación

```bash
git clone https://github.com/topassky3/tasknet-elu.git
cd tasknet-elu
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducir los experimentos

```bash
python scripts/00_phase0_check.py
python scripts/01_utility_per_bit_fashion_mnist.py
python scripts/02_learned_vs_classic.py
python scripts/03_cifar10_validation_v3.py
python scripts/04_elu_threshold_policy.py
```

Los scripts escriben sus artefactos en `results/tables/`, `results/figures/` y `reports/`. Fashion-MNIST y CIFAR-10 no se redistribuyen: se descargan mediante `torchvision` o se toman de la caché local.

## Compilar el preprint

Desde la raíz del repositorio:

```bash
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=paper \
  paper/TaskNet-ELU-preprint-v0.1.0.tex
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=paper \
  paper/TaskNet-ELU-preprint-v0.1.0.tex
```

## Reproducir y verificar la release

```bash
python scripts/00_phase0_check.py
python scripts/05_verify_release.py --check
```

El manifiesto `release/MANIFEST.sha256` registra hashes SHA-256 deterministas de los artefactos de la release. GitHub Actions valida la estructura, los metadatos, los principales resultados frente a los CSV, la compilación del preprint y la integridad del manifiesto.

Documentación detallada:

- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- [`docs/ZENODO_PUBLISHING.md`](docs/ZENODO_PUBLISHING.md)
- [`release/RELEASE_NOTES_v0.1.0.md`](release/RELEASE_NOTES_v0.1.0.md)

## Alcance y límites

TaskNet-ELU no afirma:

- haber encontrado un mínimo teórico absoluto de información;
- que el *embedding* sea óptimo;
- que gzip mida entropía real;
- que los tiempos de inferencia sean latencia física de red;
- que el oráculo sea desplegable;
- que los resultados se generalicen automáticamente a canales o redes reales.

## Cita provisional

Hasta que exista un DOI:

> Orozco, Juan Felipe (2026). *TaskNet-ELU: un protocolo experimental de utilidad por byte para comunicación orientada a tarea*. Versión 0.1.0. Preprint técnico no revisado por pares.

```bibtex
@misc{orozco2026tasknetelu,
  author       = {Orozco, Juan Felipe},
  title        = {TaskNet-ELU: un protocolo experimental de utilidad por byte para comunicaci{\'o}n orientada a tarea},
  year         = {2026},
  version      = {0.1.0},
  howpublished = {GitHub repository},
  url          = {https://github.com/topassky3/tasknet-elu},
  note         = {Preprint técnico, no revisado por pares; DOI pendiente de Zenodo}
}
```

Tras la publicación en Zenodo, esta cita deberá sustituirse por la cita que incluya el DOI definitivo.

## Licencias

- Código fuente original: **MIT**.
- Paper, documentación, tablas y figuras originales: **CC BY 4.0**.
- Datasets y materiales de terceros: conservan sus licencias originales.

Consulte [`LICENSE`](LICENSE) y [`LICENSES.md`](LICENSES.md).

## Autor

**Juan Felipe Orozco**  
Investigación independiente — Ingeniería de Telecomunicaciones  
GitHub: https://github.com/topassky3
