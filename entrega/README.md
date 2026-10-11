# DMEyF 2026 - Competencia 01 - Entrega final

**Patricio Jarmoluk**

Esta carpeta tiene todo lo necesario para replicar en forma exacta el archivo que subí como entrega final.

## Contenido

| Archivo | Qué es |
|---|---|
| `competencia01_final.R` | El script. Parte del dataset crudo de la competencia y genera el archivo de la entrega. |
| `README.md` | Este instructivo. |

No hace falta ningún otro archivo. El script descarga `competencia_01_crudo.csv` si no lo encuentra.

## Cómo correrlo

En la VM de Google Cloud, con cualquiera de estas dos formas:

- Abrir `competencia01_final.R` como notebook y ejecutar todas las celdas en orden.
- Desde una terminal: `Rscript competencia01_final.R`

El archivo de la entrega queda en:

```
~/buckets/b1/exp/s1027_vm/entrega_final_competencia01.csv
```

Es una sola columna con los `numero_de_cliente` a los que se les envía el estímulo, sin encabezado: **11.250** clientes.

## Cuánto tarda

Unas 9 horas en la VM de 8 vCPU: cerca de 1 hora la optimización de hiperparámetros, media hora la curva de junio y 7 horas los 30 modelos del ensamble (14 minutos cada uno).

Si la corrida se corta, se vuelve a lanzar desde el principio. La optimización terminada, la curva de junio y cada modelo ya entrenado quedan guardados en la carpeta de trabajo y no se repiten.

## Qué hace el script

1. Lee el dataset crudo y calcula `clase_ternaria` con el código de la cátedra.
2. Calidad de datos: `ccajas_depositos` viene rota en 202105 y se pasa a NA.
3. Ranking con cero fijo, mes por mes, de las 152 variables originales: los positivos quedan entre 0 y 1, los negativos entre -1 y 0, el cero queda en cero y los NA se dejan como NA.
4. Feature engineering histórico, calculado sobre los rankings:
   - lags y deltas de orden 1 y 2;
   - tendencia y promedio de los últimos 3 meses (función `TendenciaYmuchomas` publicada por Gustavo en Zulip).

   El modelo no ve valores en pesos: solo rankings y columnas derivadas de rankings. Son 1.064 variables predictoras (152 × 7). `numero_de_cliente` y `foto_mes` no son predictoras.
5. Optimización de hiperparámetros con `mlr3mbo`: entrena en 202103 y 202104, valida en 202106 midiendo AUC, sin undersampling y sin las variables de préstamos personales. Se optimizan `num_iterations`, `num_leaves` y `min_sum_hessian_in_leaf`. Son 12 pruebas (el diseño inicial); se elige la de mayor AUC.
6. Curva de ganancia en 202106, para mirar cuántos estímulos conviene enviar.
7. Modelo final: entrena en 202103, 202104, 202105 y 202106, sin undersampling, con las variables de préstamos personales reincorporadas (rankeadas, igual que el resto). Ensamble de 30 LightGBM con los mismos hiperparámetros y distinta semilla; se promedian las probabilidades y se predice 202108.
8. Se ordenan los clientes de mayor a menor probabilidad y se toman los primeros **11.250**. El corte se eligió comparando la curva de ganancia de junio con en el bot. Los archivos de 10.500 a 12.000 envíos: 11.250 queda en el centro de la zona más alta y pareja (11.000 a 11.500).

## Reproducibilidad

- Semilla primigenia: **123457**. Las semillas del ensamble salen de ella con el método de la cátedra: se sortean 50 números primos entre 100.000 y 1.000.000 y se usan los primeros 30.
- La optimización arranca siempre con la semilla primigenia, así que elige siempre los mismos hiperparámetros: `num_iterations` 2139, `num_leaves` 82, `min_sum_hessian_in_leaf` 0,00337931960271636 (AUC de 0,91854 en 202106).
- LightGBM corre con `deterministic = TRUE` y `force_row_wise = TRUE`.
- El resultado no depende de la cantidad de núcleos de la máquina.

Versiones con las que se generó la entrega, en la VM de la cátedra: R 4.6.1, lightgbm 4.7.0.99, data.table 1.18.6.1, mlr3mbo 1.2.1. El script las imprime al final de la corrida. El archivo exacto depende de la versión de LightGBM: con la versión pública de CRAN las probabilidades difieren a partir del quinto decimal, así que para replicarlo hay que usar la VM de la cátedra.

## Salidas del script

En `~/buckets/b1/exp/s1027_vm/`:

| Archivo | Contenido |
|---|---|
| `entrega_final_competencia01.csv` | **El archivo de la entrega.** |
| `BO_log.txt` | Las 12 pruebas de la optimización de hiperparámetros. |
| `curva_junio.txt` | Ganancia en 202106 según la cantidad de envíos. |
| `modelos/prob_<semilla>.txt` | Probabilidades de cada uno de los 30 modelos. |
| `s1027_vm_probabilidades.txt` | Probabilidad promedio del ensamble para cada cliente de 202108. |
| `s1027_vm_<N>envios.csv` | Un archivo por cada cantidad de envíos entre 10.000 y 13.500, usados para comparar puntos de corte en el bot. |
