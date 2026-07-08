# Cierre de Fase 3 — TaskNet-ELU

## Estado
Fase 3 ejecutada con CIFAR-10 real y backend torch.

## Comando ejecutado
`python scripts/03_cifar10_validation.py --epochs 10 --seeds 3`

## Archivos generados
- `results/tables/cifar10_learned_vs_classic.csv`
- `results/figures/phase3_accuracy_vs_bytes.png`
- `results/figures/phase3_utility_per_byte.png`
- `results/figures/phase3_learned_vs_classic_frontier.png`
- `reports/informe_fase_3.md`
- `reports/cierre_fase_3.md`

## Veredicto
- El EMBEDDING domina a la compresion clasica en utilidad por byte entre representaciones que conservan utilidad.
- Pareto: el embedding domina en Pareto a todos los clasicos

## Decision siguiente
Comparar el resultado contra Fase 2. Si el patron persiste, pasar a Fase 4: politica de transmision por umbral. Si cambia, documentar como evidencia de dependencia del SSP respecto a la complejidad de la fuente.