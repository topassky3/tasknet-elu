# TaskNet-ELU

**Un protocolo experimental de utilidad por byte para comunicación orientada a tarea.**

> Si un dato no mejora la decisión más de lo que cuesta enviarlo, ese dato no debería viajar.

TaskNet-ELU es un proyecto experimental que mide cuánta información necesita transmitirse para conservar una decisión. En lugar de evaluar una representación solo por su tamaño o por su fidelidad visual, este repositorio evalúa su **utilidad para una tarea**: en este caso, clasificación sobre Fashion-MNIST y CIFAR-10.

El objetivo no es proponer una teoría nueva de comunicaciones, sino construir un protocolo reproducible para comparar:

- datos originales;
- compresión clásica;
- reducciones de resolución;
- representaciones aprendidas;
- decisiones ya tomadas;
- silencio;
- y políticas de transmisión selectiva.

---

## English summary

TaskNet-ELU is an experimental protocol for measuring **utility per byte** in task-oriented communication. It evaluates whether compact learned representations can preserve task accuracy while reducing transmitted bytes, and whether a simple threshold policy can decide when transmission is worth its cost.

This repository contains scripts, tables, figures, phase reports, and a technical preprint. The current contribution is experimental and reproducible, not a new general theory.

---

## Estado del proyecto

**Versión actual:** `v0.1.0-preprint`  
**Estado:** preprint técnico, no revisado por pares.  
**Repositorio:** <https://github.com/topassky3/tasknet-elu>  
**DOI:** pendiente de publicación en Zenodo.

Este proyecto debe leerse como una prueba experimental acotada. Los resultados están limitados a los datasets, modelos y representaciones evaluadas.

---

## Idea central

En redes tradicionales, todos los datos se transmiten como paquetes, sin que la red observe directamente la decisión que esos datos deben habilitar.

TaskNet-ELU propone medir las representaciones por su relación entre:

```text
utilidad de decisión / bytes transmitidos
```

La pregunta práctica es:

> ¿Cuál es la representación más pequeña que conserva la decisión dentro de una tolerancia aceptable?

Y también:

> ¿Cuándo conviene no transmitir?

---

## Marco experimental

Una fuente produce pares:

```text
(X, Y)
```

donde:

- `X` es el dato observado, por ejemplo una imagen;
- `Y` es la clase o variable de tarea;
- el transmisor genera una representación `Z`;
- el receptor toma una decisión `A`;
- la utilidad se mide como acierto de clasificación.

La utilidad usada en los experimentos es:

```text
U = P(A = Y)
```

El costo principal medido es el número de bytes de la representación serializada.

El objetivo conceptual de ELU es:

```text
J_ELU = utilidad - costo_de_bytes - costo_de_energía - costo_de_latencia
```

En este repositorio, el foco principal está en la relación **utilidad--bytes**. Energía y latencia se reportan solo como proxies computacionales, no como mediciones físicas de red.

---

## Resultados principales

### Fase 1 — Fashion-MNIST: utilidad por byte

Se compararon:

- imagen original;
- reducción 16x16;
- reducción 8x8;
- embedding aprendido;
- etiqueta;
- silencio.

Resultado principal:

| Representación | Bytes aprox. | Exactitud | Ahorro vs original |
|---|---:|---:|---:|
| Original PNG | 512.4 | 0.8895 | — |
| Resize 16x16 | 212.3 | 0.8577 | 58.6% |
| Resize 8x8 | 78.8 | 0.7894 | 84.6% |
| Embedding | 50.4 | 0.8925 | 90.2% |
| Etiqueta | 21.0 | 0.8876 | 95.9% |
| Silencio | 0.0 | 0.1000 | 100% |

El embedding conserva la utilidad del dato completo con aproximadamente **90.2% menos bytes**.

---

### Fase 2 — Representación aprendida vs compresión clásica

Se comparó el embedding contra PNG, JPEG en varias calidades y reducciones de resolución.

Resultado principal:

| Representación | Bytes aprox. | Exactitud | Utilidad/byte |
|---|---:|---:|---:|
| Original PNG | 512.5 | 0.8859 | 0.00173 |
| JPEG q90 | 742.9 | 0.8851 | 0.00119 |
| JPEG q5 | 383.5 | 0.8622 | 0.00225 |
| Resize 8x8 | 78.8 | 0.7984 | 0.01014 |
| Embedding | 50.4 | 0.8916 | 0.01769 |

Bajo este protocolo, el embedding domina en Pareto a las representaciones clásicas evaluadas: obtiene mayor utilidad con menos bytes.

---

### Fase 3 — Validación en CIFAR-10

CIFAR-10 es más complejo que Fashion-MNIST: color, texturas y fondos.

Resultado principal:

| Representación | Bytes aprox. | Exactitud | Utilidad/byte |
|---|---:|---:|---:|
| Original PNG | 2267.7 | 0.7136 | 0.00031 |
| JPEG q90 | 1106.2 | 0.6963 | 0.00063 |
| JPEG q5 | 670.8 | 0.5732 | 0.00085 |
| Resize 8x8 | 211.7 | 0.5340 | 0.00252 |
| Embedding | 59.8 | 0.7200 | 0.01203 |
| Etiqueta | 21.0 | 0.7057 | 0.03360 |

En CIFAR-10, el embedding conserva utilidad con aproximadamente **97.4% menos bytes** que la imagen original.

---

### Fase 4 — Política de transmisión por umbral

La Fase 4 evalúa cuándo transmitir.

La política ELU usa un score simple de valor marginal frente al silencio:

```text
score = p(clase top) - p(clase mayoritaria)
```

La idea no es transmitir cuando el modelo duda, sino cuando transmitir probablemente aporta valor frente a callar.

Resultado principal:

| Política | Utilidad | Bytes medios | Fracción transmitida | Reducción vs original |
|---|---:|---:|---:|---:|
| Transmitir original siempre | 0.8895 | 511.8 | 100% | 0% |
| Siempre embedding | 0.8913 | 49.4 | 100% | 90.3% |
| Etiqueta | 0.8876 | 21.0 | 100% | 95.9% |
| Silencio | 0.1000 | 0.0 | 0% | 100% |
| Confianza q=0.500 | 0.4134 | 24.9 | 50% | 95.1% |
| ELU q=0.500 | 0.5963 | 24.5 | 50% | 95.2% |
| ELU mejor punto | 0.8696 | 41.1 | 83.3% | 92.0% |
| Oráculo | 0.9103 | 40.0 | 81.0% | 92.2% |

La política ELU reduce **92.0%** los bytes frente a transmitir siempre la imagen original, con pérdida menor a 3 puntos de utilidad.

Frente a transmitir siempre el embedding, captura aproximadamente **88% del ahorro marginal alcanzable por un oráculo**.

---

## Estructura del repositorio

```text
tasknet-elu/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── paper/
│   └── TaskNet_ELU_preprint_v10.tex
│
├── reports/
│   ├── cierre_fase_1.md
│   ├── cierre_fase_2.md
│   ├── cierre_fase_3.md
│   ├── cierre_fase_4.md
│   ├── informe_fase_1.md
│   ├── informe_fase_2.md
│   ├── informe_fase_3.md
│   └── informe_fase_4.md
│
├── results/
│   ├── figures/
│   │   ├── accuracy_vs_bytes.png
│   │   ├── phase2_accuracy_vs_bytes.png
│   │   ├── phase2_learned_vs_classic_frontier.png
│   │   ├── phase2_utility_per_byte.png
│   │   ├── phase3_accuracy_vs_bytes.png
│   │   ├── phase3_learned_vs_classic_frontier.png
│   │   ├── phase3_utility_per_byte.png
│   │   ├── phase4_traffic_utility.png
│   │   └── utility_per_byte.png
│   │
│   └── tables/
│       ├── cifar10_learned_vs_classic.csv
│       ├── elu_policy_comparison.csv
│       ├── elu_policy_comparison_raw.csv
│       ├── learned_vs_classic_fashion_mnist.csv
│       ├── utility_per_bit_fashion_mnist.csv
│       └── utility_per_byte_fashion_mnist.csv
│
└── scripts/
    ├── 00_phase0_check.py
    ├── 01_utility_per_bit_fashion_mnist.py
    ├── 02_learned_vs_classic.py
    ├── 03_cifar10_validation_v3.py
    └── 04_elu_threshold_policy.py
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/topassky3/tasknet-elu.git
cd tasknet-elu
```

### 2. Crear entorno virtual

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

En Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## Reproducción de experimentos

Primero se recomienda ejecutar la verificación inicial:

```bash
python scripts/00_phase0_check.py
```

Luego ejecutar las fases:

```bash
python scripts/01_utility_per_bit_fashion_mnist.py
python scripts/02_learned_vs_classic.py
python scripts/03_cifar10_validation_v3.py
python scripts/04_elu_threshold_policy.py
```

Los resultados se guardan en:

```text
results/tables/
results/figures/
reports/
```

---

## Compilar el preprint

El archivo principal está en:

```text
paper/TaskNet_ELU_preprint_v10.tex
```

Para compilar desde la raíz del repositorio:

```bash
pdflatex -interaction=nonstopmode -output-directory=paper paper/TaskNet_ELU_preprint_v10.tex
pdflatex -interaction=nonstopmode -output-directory=paper paper/TaskNet_ELU_preprint_v10.tex
```

El PDF queda en:

```text
paper/TaskNet_ELU_preprint_v10.pdf
```

---

## Datos

Los datasets usados son:

- Fashion-MNIST;
- CIFAR-10.

Los datos crudos no se versionan en GitHub. Los scripts descargan los datasets cuando es necesario o usan la caché local si ya existe.

La carpeta local de datos debe quedar fuera de Git:

```text
data/raw/
```

---

## Qué se mide

Este repositorio mide principalmente:

- exactitud de la decisión;
- bytes promedio por representación;
- utilidad por byte;
- ahorro relativo frente al dato original;
- dominancia de Pareto;
- políticas de transmisión selectiva;
- comparación contra baselines;
- comparación contra un oráculo no desplegable.

---

## Qué no se afirma

Este proyecto no afirma:

- haber creado una teoría nueva de comunicaciones;
- haber encontrado un mínimo teórico absoluto de información;
- que los resultados se generalicen automáticamente a redes reales;
- que el embedding sea óptimo;
- que gzip mida entropía real;
- que las latencias sean mediciones físicas de red;
- que el oráculo sea desplegable;
- que los resultados sean universales para todos los datasets, codecs o modelos.

---

## Amenazas a la validez

Las principales limitaciones son:

1. El embedding proviene de un transmisor entrenado en la tarea, por lo que representa una cota optimista.
2. Fashion-MNIST y CIFAR-10 son datasets de laboratorio.
3. Los modelos usados son compactos y se ejecutan en CPU.
4. gzip es una regla reproducible de medición, no una estimación perfecta de entropía.
5. La energía y la latencia se tratan como proxies computacionales.
6. El score de la Fase 4 usa softmax sin calibración explícita.
7. Los parámetros del cuantizador se asumen compartidos y no se cuentan por muestra.
8. El bit de señalización transmitir/callar no se contabiliza.
9. El catálogo de representaciones evaluado no prueba un mínimo teórico absoluto.
10. El oráculo usa verdad de terreno y no es una política desplegable.

---

## Trabajo futuro

Las extensiones inmediatas son:

1. Evaluar extractores no entrenados en la tarea.
2. Medir información comprimida en pesos del modelo y su relación con generalización.
3. Aplicar la regla de valor marginal a asignación de cómputo.
4. Construir un simulador multi-nodo con presupuesto compartido de canal.
5. Probar tareas con utilidad no binaria y costos de error asimétricos.
6. Evaluar canales con ruido, latencia real y tráfico multiusuario.

---

## Cita

El DOI está pendiente de publicación. Mientras tanto, este repositorio puede citarse como:

```bibtex
@misc{orozco2026tasknetelu,
  author       = {Orozco, Juan Felipe},
  title        = {TaskNet-ELU: un protocolo experimental de utilidad por byte para comunicaci{\'o}n orientada a tarea},
  year         = {2026},
  howpublished = {\url{https://github.com/topassky3/tasknet-elu}},
  note         = {Preprint t{\'e}cnico, no revisado por pares}
}
```

---

## Licencia

Este proyecto se publica con licencia MIT.

---

## Autor

**Juan Felipe Orozco**  
Investigación independiente — Ingeniería de Telecomunicaciones  
GitHub: <https://github.com/topassky3>

---

## Nota final

TaskNet-ELU no busca inflar una idea conocida, sino hacerla medible.

La apuesta del proyecto es simple:

> medir utilidad por byte puede convertir la comunicación orientada a tarea en un protocolo evaluable, falsable y reproducible.
