# Documentación del Proceso de Limpieza y Estructuración de Datos

## Objetivo
El objetivo principal de la fase de limpieza es consolidar un conjunto de datos robusto y de alta calidad en el directorio `data/clean/` (distribuido en clases como `healthy`, `common_rust`, `gray_leaf_spot`, etc.). Se busca garantizar que el modelo se entrene exclusivamente con **imágenes reales de campo** o **imágenes de laboratorio ideales** (por ejemplo, capturas controladas con fondo blanco), descartando cualquier imagen que haya sufrido alteraciones artificiales previas.

## Criterios de Selección y Filtrado
Durante la migración de datos desde `data/raw/` hacia `data/clean/`, se aplicaron reglas estrictas para evitar introducir ruido u homogeneidad artificial en los datos:

1. **Exclusión de Imágenes Tratadas:** Se evitó el uso de subconjuntos que contuvieran imágenes preprocesadas, con filtros aplicados o recortes (cropping) artificiales. Datasets como `multicrop-disease-maiz-disease-pests-and-disease` y `maize-in-field-dataset` requerían atención especial por contener este tipo de imágenes alteradas, por lo que se procedió a extraer únicamente aquellas que cumplían con el criterio de "imagen original".
2. **Selección en el Dataset CropDG:** Para el caso específico del dataset `cropdg-unified-multidomain`, el proceso de extracción se limitó estrictamente a la carpeta **`PV`**. La carpeta `CCMT` fue descartada por completo, ya que se identificó que las imágenes en su interior habían sido tratadas previamente.
3. **Eliminación de Versiones Redundantes:** El dataset `maize-diseases 1.0` fue descartado en su totalidad de la fase de extracción tras corroborar que el 100% de sus imágenes ya estaban contenidas e integradas en la versión actualizada `maize-diseases 1.1`.

## Trazabilidad y Renombrado
Para mantener el control del origen de los datos y asegurar la reproducibilidad, todas las imágenes fueron estandarizadas bajo un patrón de nomenclatura estricto al ser trasladadas a la carpeta limpia. 

El patrón de renombrado aplicado mediante script fue:
`[clase]_[origen]_[tipo]_[correlativo].[ext]`
*(Ejemplo: `healthy_cropdg_real_12345678.jpg`)*

* **clase:** Categoría de la enfermedad o estado de la hoja.
* **origen:** Abreviatura del dataset de procedencia.
* **tipo:** Especifica si la imagen es de campo (`real`) o de entorno controlado (`lab`).
* **correlativo:** Identificador único que se genera de forma secuencial o mediante un número aleatorio de 8 dígitos para evitar colisiones entre lotes.

## Detección y Eliminación de Duplicados
Para evitar el sobreajuste (overfitting) en el entrenamiento debido a datos redundantes, se implementó un flujo automatizado de limpieza a través del script `find_duplicates.py`.

* **Método Algorítmico:** Se utilizó la librería `imagededup` implementando **PHash (Perceptual Hashing)**. Este algoritmo extrae una huella visual de cada imagen (redimensionando a escala de grises y calculando frecuencias), lo que permite detectar imágenes idénticas o casi idénticas a pesar de posibles variaciones mínimas. Se configuró con un umbral de similitud de `0` para garantizar precisión estricta.
* **Resolución de Grupos (BFS):** El sistema reporta relaciones simétricas de duplicidad. Mediante búsqueda en amplitud (BFS), los pares se agruparon de manera determinista, garantizando que de cada clúster de duplicados se conserve solo una imagen original y el resto se marque para eliminación.
* **Auditoría y Registro:** Cada proceso de eliminación quedó registrado para fines de auditoría. Los reportes se almacenaron en `src/cleanup/results/` en formato CSV (ej. `duplicates_healthy_20260608_193638.csv`), documentando la ruta conservada junto con la estructura JSON de los archivos eliminados.

## Métricas de Volumen Finales
Tras aplicar los filtros de calidad y las rutinas de deduplicación mediante PHash, los volúmenes aproximados de imágenes útiles extraídas por dataset son:

* **Maize Diseases 1.1:** ~5,326 imágenes.
* **Maize Africa v1.2:** 1,872 imágenes.
* **CropDG (Solo PV):** ~1,162 imágenes.
* **Corn Leaf Roboflow:** ~1,000 imágenes.
* **Maize Africa v1:** 468 imágenes.
* **Maize Nutrient Deficiency:** ~72 imágenes.
* **Maize Diseases 1.0:** 0 imágenes (Descartado; absorbido por v1.1).