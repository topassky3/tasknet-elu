# Informe Fase 3 — TaskNet-ELU (CIFAR-10)

## Configuracion
- Fuente de datos: **cifar10**
- Backend: **torch:cpu**
- Epocas: **10**
- Semillas: **3**
- limit_train: **0**
- limit_test: **0**
- U_max = **0.7200**
- Epsilon SSP = **0.03**
- Ahorro minimo = **80%**

## Lectura SSP
- Utilidad maxima observada: **U_max = 0.7200**
- **SSP general: `label`** (grupo=decision, accuracy=0.7057, bytes=21.00, ahorro=99.1%).
- **SSP de decision: `label`** (grupo=decision, accuracy=0.7057, bytes=21.00, ahorro=99.1%).
- **SSP semantico: `embedding`** (grupo=semantic, accuracy=0.7200, bytes=59.83, ahorro=97.4%).
- **SSP visual:** no encontrado bajo epsilon=0.03 y ahorro minimo=80%.

## Veredicto de dominancia
**El EMBEDDING domina a la compresion clasica en utilidad por byte entre representaciones que conservan utilidad.**

- Lectura de eficiencia: comparado contra clasicos dentro de epsilon.
- Embedding dentro de epsilon: **True**
- Utilidad por byte del embedding: **0.01203410**
- Accuracy embedding: **0.7200**
- Bytes embedding: **59.83**
- Mejor clasico considerado: **`jpeg_q90`** (utility/byte=0.00062950, accuracy=0.6963, bytes=1106.17, within_epsilon=True)
- Lectura Pareto: el embedding domina en Pareto a todos los clasicos.

## Comparacion con Fase 2 (Fashion-MNIST)
- Fase 2 embedding accuracy: **0.8916**
- Fase 2 embedding bytes: **50.39**
- Fase 2 embedding utilidad/byte: **0.01769192**
- Fase 2 mejor clasico por utilidad/byte: **`resize_8x8`** (acc=0.7984, bytes=78.77, u/byte=0.01013542)
- Fase 3 embedding accuracy: **0.7200**
- Fase 3 embedding bytes: **59.83**
- Fase 3 embedding utilidad/byte: **0.01203410**

## Resultados
| dataset   | representation    | representation_group   | compression_family   | quality   |   bytes_raw |    bytes |     bits | serialization_method   |   accuracy |   accuracy_std |   latency_ms |   latency_ms_std |   utility_per_byte |   byte_savings | within_epsilon   | enough_savings   | is_ssp_any   | is_ssp_decision   | is_ssp_semantic   | is_ssp_visual   | receiver                        | is_silence_baseline   |   seed_count |
|:----------|:------------------|:-----------------------|:---------------------|:----------|------------:|---------:|---------:|:-----------------------|-----------:|---------------:|-------------:|-----------------:|-------------------:|---------------:|:-----------------|:-----------------|:-------------|:------------------|:------------------|:----------------|:--------------------------------|:----------------------|-------------:|
| cifar10   | original_png      | visual                 | png                  | lossless  |        3072 | 2267.73  | 18141.8  | png                    |   0.7136   |       0.008702 |     0.432069 |         0.046655 |         0.00031468 |       0        | yes              | no               | no           | no                | no                | no              | independent_full_image          | no                    |            3 |
| cifar10   | jpeg_q90          | visual                 | jpeg                 | 90        |        3072 | 1106.17  |  8849.36 | jpeg                   |   0.696333 |       0.014271 |     0.198187 |         0.041129 |         0.0006295  |       0.512212 | yes              | no               | no           | no                | no                | no              | independent_jpeg                | no                    |            3 |
| cifar10   | jpeg_q50          | visual                 | jpeg                 | 50        |        3072 |  825.806 |  6606.45 | jpeg                   |   0.678167 |       0.007069 |     0.193302 |         0.01071  |         0.00082122 |       0.635844 | no               | no               | no           | no                | no                | no              | independent_jpeg                | no                    |            3 |
| cifar10   | jpeg_q20          | visual                 | jpeg                 | 20        |        3072 |  741.082 |  5928.66 | jpeg                   |   0.6617   |       0.00479  |     0.203725 |         0.012785 |         0.00089288 |       0.673205 | no               | no               | no           | no                | no                | no              | independent_jpeg                | no                    |            3 |
| cifar10   | jpeg_q10          | visual                 | jpeg                 | 10        |        3072 |  699     |  5592    | jpeg                   |   0.632767 |       0.006914 |     0.174796 |         0.014671 |         0.00090525 |       0.691762 | no               | no               | no           | no                | no                | no              | independent_jpeg                | no                    |            3 |
| cifar10   | jpeg_q5           | visual                 | jpeg                 | 5         |        3072 |  670.814 |  5366.51 | jpeg                   |   0.5732   |       0.002677 |     0.1886   |         0.028927 |         0.00085448 |       0.704191 | no               | no               | no           | no                | no                | no              | independent_jpeg                | no                    |            3 |
| cifar10   | resize16_jpeg_q50 | visual                 | resize               | 16+q50    |         768 |  680.864 |  5446.91 | jpeg                   |   0.5713   |       0.003021 |     0.054952 |         0.005813 |         0.00083908 |       0.699759 | no               | no               | no           | no                | no                | no              | independent_resize_jpeg         | no                    |            3 |
| cifar10   | resize_8x8        | visual                 | resize               | 8         |         192 |  211.686 |  1693.49 | uint8+gzip             |   0.534    |       0.001564 |     0.02154  |         0.00135  |         0.0025226  |       0.906653 | no               | yes              | no           | no                | no                | no              | independent_resize              | no                    |            3 |
| cifar10   | embedding         | semantic               | learned              | uint8     |          64 |   59.83  |   478.64 | uint8+gzip             |   0.72     |       0.004418 |     0.001975 |         0.000331 |         0.0120341  |       0.973617 | yes              | yes              | no           | no                | yes               | no              | independent_embedding_quantized | no                    |            3 |
| cifar10   | label             | decision               | decision             | argmax    |           1 |   21     |   168    | uint8+gzip             |   0.7057   |       0.004631 |     0        |         0        |         0.0336048  |       0.99074  | yes              | yes              | yes          | yes               | no                | no              | ground_truth                    | no                    |            3 |
| cifar10   | silence           | silence                | silence              | none      |           0 |    0     |     0    | none                   |   0.1      |       0        |     0        |         0        |       nan          |       1        | no               | yes              | no           | no                | no                | no              | majority_class                  | yes                   |            3 |

## Notas de honestidad
- CIFAR-10 es mas complejo que Fashion-MNIST: color, texturas, fondos y clases mas ambiguas. Exactitudes absolutas mas bajas son esperables.
- JPEG opera aqui en un escenario mas natural que en Fashion-MNIST; si JPEG mejora, eso no refuta ELU, muestra dependencia de la fuente y la tarea.
- El embedding sale de un transmisor entrenado en la tarea; sigue siendo una cota optimista aunque el receptor sea independiente.
- Los parametros min/max del cuantizador se asumen compartidos como metadatos de calibracion y no se cuentan por muestra.
- La latencia es inferencia del receptor por muestra. No incluye entrenamiento, compresion, serializacion ni red.
- La utilidad por byte no basta por si sola; el veredicto exige conservar utilidad dentro de epsilon.