# Cierre de Fase 4 — TaskNet-ELU

## Estado
Fase 4 ejecutada con Fashion-MNIST real y backend torch.

## Comando ejecutado
`python scripts/04_elu_threshold_policy.py --epochs 5 --seeds 3`

## Veredicto
- CRITERIO 3 CUMPLIDO: la politica ELU reduce >=50% el trafico frente a TRANSMITIR TODO, conservando utilidad dentro de epsilon.
- Reduccion de trafico en el mejor punto dentro de epsilon: 16.8%
- Contra heuristica de confianza: elu_domina_a_confianza

## Decision siguiente
Con las Fases 1-4 cerradas, integrar el informe final del proyecto acotado (plan maestro) y evaluar apertura de Fases 5-6 (informacion en pesos y generalizacion) o el piloto YaruChess-ELU.