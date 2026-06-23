# Cierre de Fase 2 — TaskNet-ELU

## Estado
Fase 2 ejecutada con Fashion-MNIST real y backend torch.

## Comando ejecutado
`python scripts/02_learned_vs_classic.py --epochs 5 --seeds 3`

## Archivos generados
- `results/tables/learned_vs_classic_fashion_mnist.csv`
- `results/figures/phase2_accuracy_vs_bytes.png`
- `results/figures/phase2_utility_per_byte.png`
- `results/figures/phase2_learned_vs_classic_frontier.png`
- `reports/informe_fase_2.md`
- `reports/cierre_fase_2.md`

## Veredicto
- El EMBEDDING domina a la compresion clasica en utilidad por byte entre representaciones que conservan utilidad.
- Pareto: el embedding domina en Pareto a todos los clasicos

## Decision siguiente
Pasar a Fase 3 con CIFAR-10 para evaluar si el comportamiento persiste en imagenes mas complejas.