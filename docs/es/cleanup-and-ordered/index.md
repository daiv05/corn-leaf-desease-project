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

6. `Necrosis letal del maíz` | `Lethal Necrosis` | `MLN (MCMV + SCMV)`

### Deficiencias nutricionales

7. `Deficiencia de nitrógeno` | `Nitrogen Deficiency`

8. `Deficiencia de fósforo` | `Phosphorus Deficiency`

9. `Deficiencia de potasio` | `Potassium Deficiency`

> **Nota:** `aphids_pest` (áfidos del maíz) fue evaluada pero descartada definitivamente. Solo se hallaron ~77 imágenes en los datasets disponibles y no se encontraron fuentes adicionales. Con ese volumen, el data augmentation replicaría variaciones sintéticas de un corpus mínimo sin aportar variedad visual real, lo que sesgaría el modelo hacia artefactos de augmentation. En su lugar se incorporó `lethal_necrosis`, que cuenta con ~6 415 imágenes de campo real de dos datasets distintos.

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

| Clase                        | Lab    | Real   | Total  |
|------------------------------|-------:|-------:|-------:|
| `common_rust`                |  2 150 |    106 |  2 256 |
| `fall_armyworm`              |      0 |  4 858 |  4 858 |
| `gray_leaf_spot`             |    513 |    606 |  1 119 |
| `healthy`                    |      0 |  8 744 |  8 744 |
| `lethal_necrosis`            |      0 |  6 415 |  6 415 |
| `nitrogen_deficiency`        |      0 |    523 |    523 |
| `northern_corn_leaf_blight`  |    888 |  5 942 |  6 830 |
| `phosphorus_deficiency`      |      0 |    612 |    612 |
| `potassium_deficiency`       |      0 |    266 |    266 |
| **TOTAL**                    |  **3 551** | **28 072** | **31 623** |

## Observaciones

> Junio 2026 — Las clases con menor representación son las deficiencias nutricionales (nitrógeno 523, fósforo 612, potasio 266). Se evalúa la posibilidad de incluir datasets adicionales para estas clases. Está pendiente definir un techo de imágenes por clase para evitar sesgos en el entrenamiento (500, 1000 o 2000 por clase).

---

## corn-leaf-diseases

Se descartó su uso, debido a que el dataset incluye imágenes ya augmentadas, no incluye las imágenes originales (además de que la mayoría de ellas están presentes en otros datasets).

---

## corn-leaf-roboflow

### Identificador

`corn_leaf_roboflow`

### Metodología

El dataset original usa formato YOLO (bounding boxes + polígonos de segmentación) con splits `train/valid/test`. Se creó un script para extraer y reordenar las imágenes según las clases objetivo del proyecto, integrándolas en la estructura de `data/clean/` para su uso en el pipeline de entrenamiento.

### Mapeo de clases YOLO -> carpetas clean

| Clase YOLO | Carpeta destino | Clases objetivo |
|---|---|---|
| `fall_armyworm_damage` (0) | `fall_armyworm/real/` | Sí  |

> **Contenido visual:** la clase `fall_armyworm_damage` de Roboflow contiene imágenes de **hoja con daño** (perforaciones, mordeduras características) sin presencia visible del insecto. No hay imágenes del gusano en sí.
| `healthy` (1) | `healthy/real/` | Sí  |
| `leaf_spot` (2) | `gray_leaf_spot/real/` | Parcial  |
| `magnesium_deficiency` (3) | `magnesium_deficiency/real/` | No (referencia) |
| `nitrogen_deficiency` (4) | `nitrogen_deficiency/real/` | Sí  |
| `northern_corn_leaf_blight` (5) | `northern_corn_leaf_blight/real/` | Sí  |
| `phosphorus_deficiency` (6) | `phosphorus_deficiency/real/` | Sí  |
| `potassium_deficiency` (7) | `potassium_deficiency/real/` | Sí  |

### Resultados de extracción

Imágenes integradas:

| Clase | Imágenes copiadas |
|---|---|
| `fall_armyworm` | 503 |
| `healthy` | 619 |
| `gray_leaf_spot` | 3 946 |
| `nitrogen_deficiency` | 55 |
| `northern_corn_leaf_blight` | 292 |
| `phosphorus_deficiency` | 46 |
| `potassium_deficiency` | 52 |

### Decisiones y hallazgos

- Las imágenes ya vienen redimensionadas a **640 × 640 px**, lo que puede introducir distorsión (stretch) respecto a las proporciones originales.
- La clase `nitrogen_deficiency` (~55 instancias en etiquetas) y `phosphorus_deficiency` (~46) tienen muy baja representación.

---

## cropdg-unified-multidomain

### Criterio de selección específico

El proceso de extracción para este dataset se limitó estrictamente a la carpeta **`PV`** (PlantVillage). La carpeta `CCMT` fue descartada por completo tras identificar que las imágenes en su interior habían sido tratadas y alteradas previamente.

### Common Rust (CR) | Puccinia sorghi

### Gray Leaf Spot (GLS) | Cercospora zeae-maydis

Se recopilaron un total de **513 imágenes** en un entorno de laboratorio (`lab`), **no se encontraron imágenes duplicadas** en este lote.

### Northern Corn Leaf Blight (NCLB) | Exserohilum turcicum

Se recopiló un total de **888 imágenes** en un entorno de laboratorio (`lab`)

### Healthy | Sana

Muestras complementarias evaluadas bajo el mismo estándar de entorno controlado.

---

## maize-beans-tomatoes-africa

### Identificador

`maize_africa`

### Common Rust (CR) | Puccinia sorghi

Se detectó que la enfermedad presente en este dataset **NO corresponde a la roya común** del maíz (*Puccinia sorghi*), sino que muestra características de la roya del sur (*Puccinia polysora*).

No se incluye en la versión final del dataset, pero se mantiene en esta etapa de limpieza y ordenado para evitar confusiones.

### Gray Leaf Spot (GLS) | Cercospora zeae-maydis

No se encontraron imagenes correspondientes a esta enfermedad en este dataset, por lo que no se incluye en la versión final del dataset.

### Northern Corn Leaf Blight (NCLB) | Exserohilum turcicum

Se reunieron un total de **4,223 imágenes** tomadas en un entorno de campo abierto (`real`)

### Lethal Necrosis (MLN) | Maize Lethal Necrosis

Se recopilaron imágenes de las carpetas `Maize Lethal Necrosis Disease` presentes en las versiones v1 y v2 del dataset. Ambas versiones contienen exactamente las mismas imágenes, por lo que se procesó únicamente la v1. Las imágenes fueron renombradas con el prefijo `lethal_necrosis_maize_africa_real_` e integradas en `clean/lethal_necrosis/real/`. No se reportaron duplicados.

### Fall Armyworm | Spodoptera frugiperda

Las carpetas `Maize Fall Army Worm Pest` y `Maize Fall Army Worm Activity` contienen imágenes de cogollero tomadas en campo. Fueron integradas en `clean/fall_armyworm/real/`.

> **Contenido visual por carpeta:**
> - `Maize Fall Army Worm Activity` — imágenes de **hoja con daño** (sin insecto visible).
> - `Maize Fall Army Worm Pest` — imágenes de **hoja con daño y gusano** visible sobre la planta.

### Healthy | Sana

Muestras adicionales evaluadas según consistencia con el entorno real de campo.

---

## maize-diseases

### Identificador

`maize_desease`

### Common Rust (CR) | Puccinia sorghi

Todas las imágenes disponibles fueron tomadas en entornos controlados.

Se detectó que las imágenes presentes en la v1 y v1.1 del dataset son exactamente las mismas, por lo que se decidió eliminar la v1.1 del dataset limpio.

Las imágenes se añadieron a /clean/lab y renombraron para identificarlas.

### Gray Leaf Spot (GLS) | Cercospora zeae-maydis

No se incluyeron imágenes de esta enfermedad en este bloque debido a que el algoritmo PHash detectó que las 513 imágenes disponibles eran clones exactos de las imágenes ya integradas mediante el dataset `cropdg-unified-multidomain`.

### Northern Corn Leaf Blight (NCLB) | Exserohilum turcicum

Se reunieron alrededor de 1056 imagenes en un entorno de campo abierto. El lugar donde fueron rescatadas es en /maize-diseases/v1.1.
Adicionalmente, se detectaron imágenes duplicadas con una suma de 4223 imágenes, por lo que se decidió depurar.

### Healthy | Sana

Muestras originales validadas e integradas en el repositorio consolidado. -5,326 imágenes netas post-deduplicación.

---

## maize-in-field-dataset

### Identificador

`maize_field`

### Reorganización por enfermedad

#### Estado original

El dataset descargado desde Kaggle contenía una carpeta plana `leaf_images/` con **2355 imágenes** sin ninguna subdivisión, acompañada del archivo `Database.csv` con las etiquetas por imagen.

Estructura del CSV:

| Campo | Descripción |
|---|---|
| `imgID_id` | Identificador numérico de la imagen |
| `filePath` | Nombre del archivo |
| `GLS` | Gray Leaf Spot |
| `NCLB` | Northern Corn Leaf Blight |
| `PLS` | Phaeosphaeria Leaf Spot |
| `CR` | Common Rust |
| `SR` | Southern Rust |
| `NoFoliarSymptoms` | Sin síntomas foliares (sana) |
| `Other` | Otra condición |
| `UnidentifiedDisease` | Enfermedad no identificada |

Cada fila puede tener **una o más etiquetas activas** (valor `1`).

#### Criterio de organización

- Imagen con **una sola etiqueta** - carpeta con el nombre de la enfermedad.
- Imagen con **más de una etiqueta** - carpeta `multi_label/`.

#### Resultado

Las imágenes fueron movidas desde `leaf_images/` a subcarpetas dentro de `maize-in-field-dataset/`:

| Carpeta | Imágenes |
|---|---|
| `GLS/` | 630 |
| `multi_label/` | 891 |
| `NoFoliarSymptoms/` | 232 |
| `UnidentifiedDisease/` | 150 |
| `PLS/` | 149 |
| `NCLB/` | 140 |
| `CR/` | 107 |
| `Other/` | 49 |
| `SR/` | 7 |
| **Total** | **2355** |

### Common Rust (CR) | Puccinia sorghi

Todas las imágenes presentes fueron tomadas en entornos reales. Se movieron al dataset limpio.

### Gray Leaf Spot (GLS) | Cercospora zeae-maydis

Las imagenes con esta enfermedad fueron tomadas 607 en entornos reales. Se movieron al dataset limpio. De 629 imágenes, se encontraron 22 duplicados que fueron eliminados.

### Northern Corn Leaf Blight (NCLB) | Exserohilum turcicum

Se recopilaron alrededor de 140 imágenes con esta enfermedad. Todas fueron tomadas en entornos reales y se movieron al dataset limpio. No se encontraron imágenes duplicadas.

### Healthy | Sana

Las 232 imágenes de la carpeta `NoFoliarSymptoms` fueron validadas y enviadas al dataset limpio bajo la clase `healthy`. No se encontraron imágenes duplicadas.

---

## maize-nutrient-deficiency

### Identificador

`maize_nutrient`

### Hallazgos

No se presentan duplicados ni necesidad de limpieza específica para este dataset.

Fueron copiadas en su totalidad todas las imágenes referentes a:

| Clase | Carpeta | Imágenes |
|---|---|---|
| Sano | `Helathy` | 72 |
| Nitrógeno | `Nitrogen` | 99 |
| Fósforo | `Phosphorous` | 113 |
| Potasio | `Pottasium` | 56 |

Solamente `MagnesiumDeficiency` no se incluyó en el dataset limpio, ya que no es una clase objetivo para este proyecto, debido a los pocos ejemplos disponibles y su relevancia menor en comparación con las otras deficiencias.

---

## multicrop-disease-maiz-disease-pests-and-disease

### Identificador

`multi_desease`

### Common Rust (CR) | Puccinia sorghi

### Gray Leaf Spot (GLS) | Cercospora zeae-maydis

### Northern Corn Leaf Blight (NCLB) | Exserohilum turcicum

1. **Entropía y Desorden de Datos:** Las imágenes dentro de este repositorio se encuentran completamente desorganizadas y sin una estructura jerárquica clara, lo que impide una segmentación programática eficiente y automatizada para el entrenamiento de los modelos de Machine Learning.
2. **Mezcla Destructiva de Entornos:** Coexisten de manera aleatoria imágenes preprocesadas artificialmente (con filtros aplicados y recortes forzados), imágenes tomadas en campo abierto (`real`) y capturas controladas de laboratorio (`lab`) sin un orden específico ni metadatos de etiquetado que permitan su separación.
3. **Compromiso de Calidad:** La falta de homogeneidad en el origen e integridad de las imágenes introduce ruido masivo y disminuye significativamente el rendimiento y la capacidad de generalización del modelo de aprendizaje automático.

Por lo tanto, para salvaguardar la robustez del set consolidado, se decidió omitir este dataset del flujo final de limpieza.

### Lethal Necrosis (MLN) | Maize Lethal Necrosis

A diferencia de las otras clases de este dataset, las imágenes de Lethal Necrosis se encuentran correctamente aisladas en su propia carpeta y son identificables sin ambigüedad. Se integraron en su totalidad en `clean/lethal_necrosis/real/`. No se reportaron duplicados. Las imágenes son todas de entorno real.

### Fall Armyworm | Spodoptera frugiperda

> **Contenido visual:** las imágenes de cogollero de este dataset son una mezcla de **hoja con daño** y **hoja con daño + gusano** visible, sin separación entre ambos tipos dentro de la carpeta.

### Healthy | Sana

-