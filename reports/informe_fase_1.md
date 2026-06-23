# Informe Fase 1 — TaskNet-ELU

## Configuracion
- Fuente de datos: **fashion_mnist**
- Backend de modelos: **torch**
- Epocas: **5**
- Semillas: **3**
- limit_train: **0**
- limit_test: **0**
- Tolerancia SSP epsilon: **0.03**
- Ahorro minimo SSP: **80%**

## Resultados

| representation   | representation_group   |   bytes_raw |   bytes |     bits | serialization_method   |   accuracy |   accuracy_std |   latency_ms |   latency_ms_std |   utility_per_byte |   byte_savings | within_epsilon   | enough_savings   | is_ssp_any   | is_ssp_decision   | is_ssp_semantic   | is_ssp_visual   | receiver                        | is_silence_baseline   |   seed_count |
|:-----------------|:-----------------------|------------:|--------:|---------:|:-----------------------|-----------:|---------------:|-------------:|-----------------:|-------------------:|---------------:|:-----------------|:-----------------|:-------------|:------------------|:------------------|:----------------|:--------------------------------|:----------------------|-------------:|
| original         | visual                 |         784 | 512.438 | 4099.5   | png                    |   0.8895   |       0.001577 |     0.118609 |         0.003895 |         0.00173582 |       0        | yes              | no               | no           | no                | no                | no              | independent_full_image          | no                    |            3 |
| resize_16x16     | visual                 |         256 | 212.262 | 1698.1   | uint8+gzip             |   0.857667 |       0.003838 |     0.045551 |         0.002153 |         0.0040406  |       0.58578  | no               | no               | no           | no                | no                | no              | independent_resize              | no                    |            3 |
| resize_8x8       | visual                 |          64 |  78.77  |  630.16  | uint8+gzip             |   0.7894   |       0.005175 |     0.010269 |         0.001351 |         0.0100216  |       0.846284 | no               | yes              | no           | no                | no                | no              | independent_resize              | no                    |            3 |
| embedding        | semantic               |          32 |  50.394 |  403.152 | uint8+gzip             |   0.892467 |       0.002798 |     0.000518 |         7.6e-05  |         0.0177098  |       0.901658 | yes              | yes              | no           | no                | yes               | no              | independent_embedding_quantized | no                    |            3 |
| label            | decision               |           1 |  21     |  168     | uint8+gzip             |   0.8876   |       0.003023 |     0        |         0        |         0.0422667  |       0.959019 | yes              | yes              | yes          | yes               | no                | no              | ground_truth                    | no                    |            3 |
| silence          | silence                |           0 |   0     |    0     | none                   |   0.1      |       0        |     0        |         0        |       nan          |       1        | no               | yes              | no           | no                | no                | no              | majority_class                  | yes                   |            3 |

## Lectura SSP

- Utilidad maxima observada: **U_max = 0.8925**
- **SSP general de menor byte: `label`** (grupo=decision, accuracy=0.8876, bytes=21.00, ahorro=95.9%).
- **SSP de decision: `label`** (grupo=decision, accuracy=0.8876, bytes=21.00, ahorro=95.9%).
- **SSP semantico reutilizable: `embedding`** (grupo=semantic, accuracy=0.8925, bytes=50.39, ahorro=90.2%).
- **SSP visual:** no encontrado bajo epsilon=0.03 y ahorro minimo=80%.

## Interpretacion tecnica

La etiqueta (`label`) puede aparecer como SSP general porque representa una decision ya tomada por el transmisor. Esto es valido si la red solo necesita transportar la decision final, pero no sirve como evidencia reutilizable ni como representacion rica del dato.

El `embedding` es la representacion semantica mas relevante para TaskNet-ELU porque conserva informacion reutilizable para un receptor independiente. En esta version se evalua despues de cuantizarlo y reconstruirlo, por lo que la utilidad se mide sobre lo que realmente viajaria.

Los puntos crudos de utilidad vs bytes no tienen que ser monotonos. La frontera empirica se interpreta como la envolvente superior best-so-far al ordenar por bytes.

La latencia reportada es un proxy de computo por representacion, no una medicion de red.