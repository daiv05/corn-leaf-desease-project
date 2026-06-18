# Limpieza y ordenado de datasets

Aquí se resume el proceso y hallazgos encontrados al limpiar, clasificar y estandarizar las imágenes antes de incorporarlas al dataset final. La prioridad es preservar la trazabilidad (origen del dataset) y evitar modificar el material crudo (raw).

## Objetivo

- Consolidar imágenes útiles por clase (enfermedad) y contexto (lab/real).
- Eliminar duplicados, outliers y material procesado (recortes, filtros, etc.).
- Estandarizar nombres para facilitar auditorías y entrenamiento.

### Descartado

**`corn-leaf-diseases-plant-village-augmented-data`**

Debido a que contiene imágenes procesadas (recortes, filtros, augmentaciones) y no se dispone del material original, se decidió omitir este dataset para evitar confusiones. Se mantiene documentado en esta etapa de limpieza para referencia futura.

### A procesar

Se exploran los siguientes datasets, y se definen identificadores para cada uno, que se incluirán en los nombres de las imágenes para mantener la trazabilidad:

**DATASET                                       -----> IDENTIFICADOR**

`corn-leaf-roboflow                                --> corn_leaf_roboflow`

`cropdg-unified-multidomain                        --> cropdg`

`maize-beans-and-tomatoes-image-dataset-for-africa --> maize_africa`

`maize-diseases                                    --> maize_desease`

`maize-in-field-dataset                            --> maize_field`

`maize-nutrient-deficiency                         --> maize_nutrient`

`multicrop-disease-maiz-disease-pests-and-disease  --> multi_desease`

## Clases consideradas (junio 2026)

### Enfermedades foliares

1. `Roya común` | `Common Rust` | `Puccinia sorghi`

2. `Tizón foliar del norte (NCLB)` | `Northern Corn Leaf Blight` | `Exserohilum turcicum`

3. `Mancha gris de la hoja (GLS)` | `Gray Leaf Spot` | `Cercospora zeae-maydis`

4. `Hoja sana` | `Healthy` | `-`

5. `Gusano cogollero` | `Fall Armyworm` | `Spodoptera frugiperda`

### Deficiencias nutricionales

6. `Deficiencia de nitrógeno` | `Nitrogen Deficiency`

7. `Deficiencia de fósforo` | `Phosphorus Deficiency`

8. `Deficiencia de potasio` | `Potassium Deficiency`

## Flujo de trabajo implementado

1. Clasificar por entorno en `/data/clean`:
    1. `lab`: fotos en entornos controlados (fondo negro/blanco/gris, estudio).
    2. `real`: fotos tomadas en campo.

    - Agregar el identificador del dataset en un paso intermedio para conservar el origen.
    - Excluir imágenes procesadas (recortes, filtros, augmentaciones). Si un dataset tiene este tipo de imágenes, documentarlo y omitir.

2. Eliminar duplicados con `imagededup` y el script disponible en el repositorio. Asegurarse de no afectar las imágenes originales ni de otras carpetas.
3. Estandarizar nombres:
    - Anteponer el nombre de la clase, en inglés:
        - common_rust
        - northern_corn_leaf_blight
        - gray_leaf_spot
        - healthy
    - Agregar el identificador del dataset (ver sección de [Por procesar](./#por-procesar))
    - Incluir el ambiente: `real` o `lab`.
    - Terminar con correlativo ascendente: `_1`, `_2`, `_3`, etc.
    - Ejemplo final: `common_rust_maize_africa_lab_1`.

    Este orden garantiza trazabilidad y evita colisiones entre datasets.

4. Revisar y eliminar outliers: imágenes que no correspondan a la enfermedad, que posean carteles o marcas, tomas aéreas o detalles inconsistentes.

## Documentación

- Para cada dataset se registran decisiones y hallazgos (dataset descartado, criterios, problemas de calidad), además de anotar cualquier ajuste al proceso para mantener la trazabilidad del dataset final.

## Resultados obtenidos

Tras aplicar las rutinas automatizadas de deduplicación y filtros de exclusión por calidad, el volumen neto de imágenes útiles integradas en `data/clean/` por clase es el siguiente:

| Clase                        | Lab    | Real   | Total  | Tamaño    |
|------------------------------|-------:|-------:|-------:|----------:|
| `aphids_pest`                |      0 |     77 |     77 | 302.6 MB  |
| `common_rust`                |  2 150 |    106 |  2 256 | 570.9 MB  |
| `fall_armyworm`              |      0 |  4 858 |  4 858 |   2.1 GB  |
| `gray_leaf_spot`             |    513 |    606 |  1 119 |   2.2 GB  |
| `healthy`                    |      0 |  8 744 |  8 744 |   4.6 GB  |
| `nitrogen_deficiency`        |      0 |    523 |    523 |  33.1 MB  |
| `northern_corn_leaf_blight`  |    888 |  5 942 |  6 830 |   4.4 GB  |
| `phosphorus_deficiency`      |      0 |    612 |    612 |  41.3 MB  |
| `potassium_deficiency`       |      0 |    266 |    266 |  17.8 MB  |
| **TOTAL**                    |  **3 551** | **21 734** | **25 285** | **14.3 GB** |

## Observaciones

> Junio 2026 - Se están analizando opciones para aumentar la cantidad de imágenes en clases con menor representación, como `aphids_pest` y las deficiencias nutricionales. Se evalúa la posibilidad de incluir datasets adicionales. Asi mismo se está por definir la cantidad o techo de imágenes por clase para evitar sesgos en el entrenamiento (500, 1000 o 2000 por clase).