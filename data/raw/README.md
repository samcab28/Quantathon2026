# data/raw/

CSV original del dataset **Water Potability**, sin ninguna modificación.

## Cómo obtenerlo (descarga manual)

1. Ir a https://www.kaggle.com/datasets/adityakadiwal/water-potability
2. Descargar el ZIP (botón "Download") — requiere cuenta de Kaggle.
3. Descomprimir y colocar el archivo aquí como:
   ```
   data/raw/water_potability.csv
   ```

Alternativa vía API (si más adelante se configuran credenciales en `~/.kaggle/kaggle.json`):
```bash
pip install kaggle
kaggle datasets download -d adityakadiwal/water-potability -p data/raw --unzip
```

## Qué debe contener

Un único CSV con 3276 filas y 10 columnas:

| Columna | Descripción |
|---|---|
| `ph` | pH del agua (491 NaN esperados) |
| `Hardness` | dureza |
| `Solids` | sólidos disueltos totales |
| `Chloramines` | cloraminas |
| `Sulfate` | sulfato (781 NaN esperados) |
| `Conductivity` | conductividad |
| `Organic_carbon` | carbono orgánico |
| `Trihalomethanes` | trihalometanos (162 NaN esperados) |
| `Turbidity` | turbidez |
| `Potability` | objetivo binario (0 = no potable, 1 = potable) |

## Reglas

- **No editar ni limpiar nada aquí.** Toda transformación (imputación,
  estandarización, balanceo) va en `src/data_prep/` y su salida a
  `data/processed/`.
- No versionar el CSV en git si el equipo prefiere mantener el repo liviano
  (agregar `data/raw/*.csv` a `.gitignore`); documentar en el README raíz cómo
  regenerarlo en ese caso.
