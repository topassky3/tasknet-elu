# Informe Fase 4 — TaskNet-ELU (politica de transmision por umbral)

## Configuracion
- Fuente de datos: **fashion_mnist** · Backend: **torch**
- Epocas: **5** · Semillas: **3**
- limit_train: **0** · limit_test: **0**
- Bytes medios del embedding: **49.66**
- Epsilon: **0.03** · Puntos de barrido: **25**

## Veredicto

**CRITERIO 3 CUMPLIDO: la politica ELU reduce >=50% el trafico frente a TRANSMITIR TODO, conservando utilidad dentro de epsilon.**

- Mejor punto ELU dentro de epsilon: q=0.833, utilidad=0.8696, bytes medios=41.12, trafico=83.3%
- Reduccion de trafico vs TRANSMITIR TODO (criterio 3 del plan): **92.0%**
- Reduccion de trafico vs always_embedding (contribucion marginal de la politica): **16.8%**
- Techo del oraculo (maximo ahorro posible vs always_embedding en este escenario): **19.1%**. Con respaldo de clase mayoritaria en dataset balanceado, este techo es bajo por diseno: callar casi nunca acierta, asi que casi toda muestra amerita transmision.
- Comparacion contra heuristica de confianza (umbral barrido): **elu_domina_a_confianza**

## Resultados (agregados entre semillas)

| policy           | param     |   utility |   utility_std |   mean_bytes |   traffic |   seeds |
|:-----------------|:----------|----------:|--------------:|-------------:|----------:|--------:|
| always_embedding |           |  0.891333 |      0.002984 |     49.397   |  1        |       3 |
| always_label     |           |  0.8876   |      0.003703 |     21       |  1        |       3 |
| always_original  |           |  0.8895   |      0.001931 |    511.752   |  1        |       3 |
| confidence       | q=0.000   |  0.1      |      0        |      0       |  0        |       3 |
| confidence       | q=0.042   |  0.113133 |      0.003102 |      2.07837 |  0.0417   |       3 |
| confidence       | q=0.083   |  0.1298   |      0.006409 |      4.14827 |  0.0833   |       3 |
| confidence       | q=0.125   |  0.148167 |      0.008775 |      6.22627 |  0.125    |       3 |
| confidence       | q=0.167   |  0.169433 |      0.00973  |      8.30287 |  0.1667   |       3 |
| confidence       | q=0.208   |  0.1938   |      0.009875 |     10.3747  |  0.2083   |       3 |
| confidence       | q=0.250   |  0.2216   |      0.010693 |     12.4501  |  0.25     |       3 |
| confidence       | q=0.292   |  0.250833 |      0.011663 |     14.5259  |  0.2917   |       3 |
| confidence       | q=0.333   |  0.281533 |      0.011303 |     16.5992  |  0.3333   |       3 |
| confidence       | q=0.375   |  0.3126   |      0.010718 |     18.6763  |  0.375    |       3 |
| confidence       | q=0.417   |  0.345767 |      0.010661 |     20.7549  |  0.4167   |       3 |
| confidence       | q=0.458   |  0.3792   |      0.011321 |     22.8255  |  0.4583   |       3 |
| confidence       | q=0.500   |  0.413367 |      0.01092  |     24.8982  |  0.5      |       3 |
| confidence       | q=0.542   |  0.447933 |      0.009589 |     26.9662  |  0.5417   |       3 |
| confidence       | q=0.583   |  0.4845   |      0.008007 |     29.0258  |  0.5833   |       3 |
| confidence       | q=0.625   |  0.5222   |      0.005789 |     31.093   |  0.625    |       3 |
| confidence       | q=0.667   |  0.561267 |      0.004717 |     33.1536  |  0.6667   |       3 |
| confidence       | q=0.708   |  0.601067 |      0.003564 |     35.2104  |  0.7083   |       3 |
| confidence       | q=0.750   |  0.6419   |      0.003041 |     37.268   |  0.75     |       3 |
| confidence       | q=0.792   |  0.683233 |      0.002977 |     39.3214  |  0.7917   |       3 |
| confidence       | q=0.833   |  0.724667 |      0.003009 |     41.3599  |  0.8333   |       3 |
| confidence       | q=0.875   |  0.766333 |      0.002984 |     43.3949  |  0.875    |       3 |
| confidence       | q=0.917   |  0.808033 |      0.002984 |     45.4149  |  0.9167   |       3 |
| confidence       | q=0.958   |  0.849633 |      0.002984 |     47.4146  |  0.9583   |       3 |
| confidence       | q=1.000   |  0.891333 |      0.002984 |     49.397   |  1        |       3 |
| elu_threshold    | q=0.000   |  0.1      |      0        |      0       |  0        |       3 |
| elu_threshold    | q=0.042   |  0.1417   |      0        |      1.98243 |  0.0417   |       3 |
| elu_threshold    | q=0.083   |  0.1833   |      0        |      3.98193 |  0.0833   |       3 |
| elu_threshold    | q=0.125   |  0.225    |      0        |      6.00373 |  0.125    |       3 |
| elu_threshold    | q=0.167   |  0.2667   |      0        |      8.0367  |  0.1667   |       3 |
| elu_threshold    | q=0.208   |  0.308233 |      5.8e-05  |     10.0763  |  0.2083   |       3 |
| elu_threshold    | q=0.250   |  0.3499   |      0        |     12.1291  |  0.25     |       3 |
| elu_threshold    | q=0.292   |  0.3915   |      0.0001   |     14.1886  |  0.2917   |       3 |
| elu_threshold    | q=0.333   |  0.4328   |      0.000265 |     16.2451  |  0.3333   |       3 |
| elu_threshold    | q=0.375   |  0.474033 |      5.8e-05  |     18.3063  |  0.375    |       3 |
| elu_threshold    | q=0.417   |  0.514867 |      0.000208 |     20.3711  |  0.4167   |       3 |
| elu_threshold    | q=0.458   |  0.555633 |      0.000321 |     22.4298  |  0.4583   |       3 |
| elu_threshold    | q=0.500   |  0.596267 |      0.000321 |     24.4997  |  0.5      |       3 |
| elu_threshold    | q=0.542   |  0.636633 |      0.000551 |     26.5745  |  0.5417   |       3 |
| elu_threshold    | q=0.583   |  0.676267 |      0.001012 |     28.6464  |  0.5833   |       3 |
| elu_threshold    | q=0.625   |  0.714    |      0.002291 |     30.7272  |  0.625    |       3 |
| elu_threshold    | q=0.667   |  0.750133 |      0.002013 |     32.8077  |  0.6667   |       3 |
| elu_threshold    | q=0.708   |  0.784567 |      0.001692 |     34.8818  |  0.7083   |       3 |
| elu_threshold    | q=0.750   |  0.816333 |      0.002572 |     36.961   |  0.75     |       3 |
| elu_threshold    | q=0.792   |  0.843967 |      0.002857 |     39.0431  |  0.7917   |       3 |
| elu_threshold    | q=0.833   |  0.869567 |      0.00294  |     41.1212  |  0.8333   |       3 |
| elu_threshold    | q=0.875   |  0.888    |      0.001997 |     43.2044  |  0.875    |       3 |
| elu_threshold    | q=0.917   |  0.8911   |      0.002498 |     45.2666  |  0.9167   |       3 |
| elu_threshold    | q=0.958   |  0.890967 |      0.002673 |     47.3286  |  0.9583   |       3 |
| elu_threshold    | q=1.000   |  0.891333 |      0.002984 |     49.397   |  1        |       3 |
| never_silence    |           |  0.1      |      0        |      0       |  0        |       3 |
| oracle           | DU_true>0 |  0.910333 |      0.002686 |     39.9679  |  0.810333 |       3 |

## Notas de honestidad
- DU_hat se estima con el softmax del transmisor (proxy de calibracion declarado): DU_hat(x) = p_top(x) - p_mayoritaria(x). No usa la verdad de terreno.
- El silencio cae a la clase mayoritaria del TRAIN (el receptor mudo no ve test).
- El bit de senalizacion transmitir/callar no se cuenta en los bytes; se declara.
- El oraculo usa la verdad de terreno: es un techo teorico, no una politica real.
- La heuristica de confianza se barre en todo su rango de umbral para competir en igualdad de condiciones (curva completa, no un punto).
- tx y receptores se entrenan solo con train; las politicas se evaluan solo en test.