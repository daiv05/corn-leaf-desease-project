# corn-leaf-roboflow

## Identificador

`corn_leaf_roboflow`

## Metodología

El dataset original usa formato YOLO (bounding boxes + polígonos de segmentación) con splits `train/valid/test`. Se creó un script para extraer y reordenar las imágenes según las clases objetivo del proyecto, integrándolas en la estructura de `data/clean/` para su uso en el pipeline de entrenamiento.

## Mapeo de clases YOLO → carpetas clean

| Clase YOLO | Carpeta destino | Clases objetivo |
|---|---|---|
| `fall_armyworm_damage` (0) | `fall_armyworm/real/` | Sí  |
| `healthy` (1) | `healthy/real/` | Sí  |
| `leaf_spot` (2) | `gray_leaf_spot/real/` | Parcial  |
| `magnesium_deficiency` (3) | `magnesium_deficiency/real/` | No (referencia) |
| `nitrogen_deficiency` (4) | `nitrogen_deficiency/real/` | Sí  |
| `northern_corn_leaf_blight` (5) | `northern_corn_leaf_blight/real/` | Sí  |
| `phosphorus_deficiency` (6) | `phosphorus_deficiency/real/` | Sí  |
| `potassium_deficiency` (7) | `potassium_deficiency/real/` | Sí  |

## Resultados de extracción

Imágenes integradas en `data/clean/` tras ejecutar el script (campo `real`, sin contrapartes de `lab`):

| Clase | Imágenes copiadas |
|---|---|
| `fall_armyworm` | 4 858 |
| `healthy` | - (mezclado con otras fuentes) |
| `gray_leaf_spot` | - (mezclado con otras fuentes) |
| `magnesium_deficiency` | - (pendiente de conteo) |
| `nitrogen_deficiency` | 523 |
| `northern_corn_leaf_blight` | - (mezclado con otras fuentes) |
| `phosphorus_deficiency` | 612 |
| `potassium_deficiency` | 266 |

## Decisiones y hallazgos

- Las imágenes ya vienen redimensionadas a **640 × 640 px** por Roboflow, lo que puede introducir distorsión (stretch) respecto a las proporciones originales.
- La clase `nitrogen_deficiency` (~55 instancias en etiquetas) y `phosphorus_deficiency` (~46) tienen muy baja representación en el raw.
