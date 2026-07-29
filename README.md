# TaskNet-ELU

**Un protocolo experimental de utilidad por byte para comunicación orientada a tarea.**

> Si un dato no mejora la decisión más de lo que cuesta enviarlo, ese dato no debería viajar.

TaskNet-ELU estudia cuánta información debe transmitirse para conservar una decisión. Compara datos originales, compresión clásica, reducciones de resolución, representaciones aprendidas, decisiones ya tomadas, silencio y políticas de transmisión selectiva en Fashion-MNIST y CIFAR-10.

La contribución declarada es un **protocolo experimental reproducible**. No se propone una teoría general nueva de comunicaciones.

## Estado

- Versión: `v0.1.0-preprint`.
- Estado editorial: preprint técnico, no revisado por pares.
- Autor: Juan Felipe Orozco.
- Afiliación declarada: Investigación independiente — Ingeniería de Telecomunicaciones.
- DOI: pendiente de publicación en Zenodo.
- PDF: [`paper/TaskNet-ELU-preprint-v0.1.0.pdf`](paper/TaskNet-ELU-preprint-v0.1.0.pdf).

## Resultados principales

Bajo el catálogo y protocolo evaluados:

- el embedding conservó la exactitud del dato completo con aproximadamente 90,2 % menos bytes en Fashion-MNIST;
- en CIFAR-10 conservó utilidad con aproximadamente 97,4 % menos bytes que el PNG original;
- el embedding mostró aproximadamente 7,9× y 19,1× más utilidad por byte que el mejor clásico que conservó utilidad en Fashion-MNIST y CIFAR-10, respectivamente;
- la política ELU redujo 92,0 % los bytes frente a transmitir siempre el original, con pérdida menor a tres puntos de exactitud;
- estos resultados no constituyen una ley universal y están limitados a los datasets, modelos y representaciones evaluados.

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

Los resultados se escriben en:

- `results/tables/`;
- `results/figures/`;
- `reports/`.

Fashion-MNIST y CIFAR-10 no se redistribuyen; los scripts los descargan o usan la caché local.

## Reproducir y verificar la release

```bash
python scripts/00_phase0_check.py
python scripts/05_verify_release.py --check
```

Para regenerar el manifiesto determinista después de cualquier cambio en los artefactos:

```bash
python scripts/05_verify_release.py --write-manifest
python scripts/05_verify_release.py --check
```

La documentación detallada está en:

- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md);
- [`docs/release-audit-v0.1.0.md`](docs/release-audit-v0.1.0.md);
- [`docs/ZENODO_PUBLISHING.md`](docs/ZENODO_PUBLISHING.md).

## Fuente del preprint

El PDF final de esta versión está en:

```text
paper/TaskNet-ELU-preprint-v0.1.0.pdf
```

La auditoría de release no localizó todavía en la rama base un archivo `.tex` con el nombre final. La publicación en Zenodo debe permanecer bloqueada hasta recuperar o confirmar la fuente LaTeX real, compilarla dos veces y verificar que produce el PDF esperado.

## Qué se mide

- exactitud de la decisión;
- bytes promedio por representación;
- utilidad por byte;
- ahorro frente al dato original;
- dominancia de Pareto;
- políticas selectivas de transmisión;
- comparación contra baselines y un oráculo no desplegable.

## Qué no se afirma

TaskNet-ELU no afirma:

- haber creado una teoría nueva de comunicaciones;
- haber encontrado un mínimo teórico absoluto de información;
- que el embedding sea óptimo;
- que gzip mida entropía real;
- que las latencias sean mediciones físicas de red;
- que el oráculo sea desplegable;
- que los resultados se generalicen automáticamente a redes reales.

## Cita provisional

Hasta que exista un DOI, cite esta versión como:

> Orozco, Juan Felipe (2026). *TaskNet-ELU: un protocolo experimental de utilidad por byte para comunicación orientada a tarea*. Versión 0.1.0, preprint técnico no revisado por pares. GitHub: https://github.com/topassky3/tasknet-elu

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

Después de publicar en Zenodo, sustituya esta cita provisional por la cita exportada con el DOI definitivo.

## Licencias

- Código fuente original: MIT.
- Paper, documentación, tablas y figuras originales: CC BY 4.0.
- Datasets y materiales de terceros: conservan sus licencias originales.

Consulte [`LICENSE`](LICENSE) y [`LICENSES.md`](LICENSES.md).

## Autor

**Juan Felipe Orozco**  
Investigación independiente — Ingeniería de Telecomunicaciones  
GitHub: https://github.com/topassky3
