# Cierre de Fase 1 — TaskNet-ELU

## Estado
Fase 1 ejecutada con Fashion-MNIST real y backend torch.

## Comando ejecutado
`python scripts/01_utility_per_bit_fashion_mnist.py --epochs 5 --seeds 3`

## Archivos generados
- `results/tables/utility_per_bit_fashion_mnist.csv`
- `results/tables/utility_per_byte_fashion_mnist.csv`
- `results/figures/accuracy_vs_bytes.png`
- `results/figures/utility_per_byte.png`
- `reports/informe_fase_1.md`

## Resultado SSP
- **SSP general: `label`** (grupo=decision, accuracy=0.8876, bytes=21.00, ahorro=95.9%).
- **SSP semantico: `embedding`** (grupo=semantic, accuracy=0.8925, bytes=50.39, ahorro=90.2%).

## Decision siguiente
Si los archivos existen, las columnas pasan validacion y los resultados se interpretan sin forzar, esta corrida puede considerarse Fase 1 v0.2. La siguiente decision es validar visualmente graficas y pasar a Fase 2 o repetir con mas epocas/semillas.