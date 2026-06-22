# Análisis Exploratorio de Datos

El EDA busca responder tres preguntas antes de diseñar el pipeline de entrenamiento: ¿qué hay en el dataset?, ¿qué problemas tiene?, y ¿qué decisiones impone?

El análisis completo y reproducible, con todo el código, está en la notebook [`notebooks/01_eda.ipynb`](https://github.com/daiv05/corn-leaf-desease-project/blob/master/notebooks/01_eda.ipynb). Esta página resume los hallazgos y las decisiones que se derivaron de ellos.

---

## Composición del dataset

El dataset consolidado en `data/clean/` contiene **31 622 imágenes** distribuidas en **9 clases**, procedentes de 6 fuentes públicas. Cada imagen pertenece a un entorno de captura: `lab` (fondo controlado, iluminación artificial) o `real` (campo abierto, iluminación solar).

| Clase | Lab | Real | Total |
|---|---:|---:|---:|
| `healthy` | 0 | 8 744 | **8 744** |
| `northern_corn_leaf_blight` | 888 | 5 942 | **6 830** |
| `lethal_necrosis` | 0 | 6 415 | **6 415** |
| `fall_armyworm` | 0 | 4 857 | **4 857** |
| `common_rust` | 2 150 | 106 | **2 256** |
| `gray_leaf_spot` | 513 | 606 | **1 119** |
| `phosphorus_deficiency` | 0 | 612 | **612** |
| `nitrogen_deficiency` | 0 | 523 | **523** |
| `potassium_deficiency` | 0 | 266 | **266** |

---

## 1. Distribución de clases

El dataset presenta un **desbalance severo**: `healthy` (8 744 imágenes) supera en **32.9×** a `potassium_deficiency` (266). Este cociente supera el umbral donde el modelo ignoraría sistemáticamente las clases minoritarias sin intervención explícita.

![Distribución de imágenes por clase](/eda/eda_01_distribucion_clases.png)

**Decisión:** se usa `WeightedRandomSampler` en el DataLoader para igualar la frecuencia efectiva de cada clase durante el entrenamiento, combinado con `class_weight='balanced'` en los baselines de sklearn. Para las clases más escasas (`potassium_deficiency`, `nitrogen_deficiency`, `phosphorus_deficiency`) se aplica un pipeline de augmentation extendido (`CornMinorityTransforms`).

- [Ver análisis completo: sección 1.2 de la notebook](https://github.com/daiv05/corn-leaf-desease-project/blob/master/notebooks/01_eda.ipynb)

---

## 2. Sesgo de entorno (lab vs real)

La proporción de imágenes de laboratorio vs campo varía drásticamente entre clases, lo que introduce riesgo de *domain shortcut*: una red puede aprender el fondo uniforme de PlantVillage en lugar de los síntomas foliares.

![Proporción lab vs real por clase](/eda/eda_02_lab_vs_real.png)

El caso más crítico es `common_rust`: **95.4 %** de sus 2 256 imágenes proviene de entorno controlado (fondo negro/blanco uniforme). En campo solo hay 106 imágenes.

**Decisión:** la división de validación prioriza imágenes de campo (`real/`) sobre las de laboratorio para medir la generalización real del modelo, no su capacidad de reconocer fondos de PlantVillage. Adicionalmente, el augmentation de `common_rust` incluye variaciones de fondo y color para mitigar el sesgo de dominio.

- [Ver análisis completo: sección 1.3 de la notebook](https://github.com/daiv05/corn-leaf-desease-project/blob/master/notebooks/01_eda.ipynb)

---

## 3. Resolución y dimensiones

Las imágenes tienen resoluciones muy dispares. Las de `corn_leaf_roboflow` ya vienen redimensionadas a **640 × 640 px** (Roboflow). El resto conserva su resolución original, que va desde imágenes pequeñas hasta varios megapíxeles.

![Distribución de resoluciones](/eda/eda_03_resoluciones.png)

**Decisión:** el pipeline aplica `Resize` a la resolución de entrada del modelo (224 × 224 px para MobileNetV3) como primera transformación, seguido de `CenterCrop` en validación y `RandomResizedCrop` en entrenamiento. El redimensionamiento desde 640 × 640 no genera pérdida significativa de información diagnóstica a esa escala.

- [Ver análisis completo: sección 1.4 de la notebook](https://github.com/daiv05/corn-leaf-desease-project/blob/master/notebooks/01_eda.ipynb)

---

## 4. Calidad de imagen

Se evaluaron tres métricas sobre una muestra estratificada de hasta 400 imágenes por clase: desenfoque (varianza del Laplaciano), subexposición (brillo medio < 40) y sobreexposición (brillo medio > 230).

![Problemas de calidad por clase](/eda/eda_04_calidad.png)

![Distribuciones de calidad por clase](/eda/eda_04b_calidad_boxplots.png)

El porcentaje de imágenes con problemas es bajo y distribuido uniformemente entre clases, sin concentración en ninguna categoría específica. Las imágenes oscuras en clases de laboratorio (fondo negro) son artefactos del umbral global, no defectos reales.

**Decisión:** no se aplica filtro automático de calidad. Las métricas no son lo suficientemente discriminativas para justificar eliminar imágenes sin revisión manual, y el volumen de clases pequeñas no admite pérdidas adicionales.

- [Ver análisis completo: sección 1.5 de la notebook](https://github.com/daiv05/corn-leaf-desease-project/blob/master/notebooks/01_eda.ipynb)

---

## 5. Duplicados y limpieza

Durante la etapa de construcción del dataset se detectaron duplicados entre fuentes distintas mediante **hash perceptual (pHash)** con `imagededup` (threshold = 0, solo copias exactas a nivel perceptual).

![Duplicados eliminados por clase y fuente](/eda/eda_05_duplicados.png)

Se eliminaron **8 538 imágenes** agrupadas en **8 050 grupos**. Las fuentes con mayor contaminación cruzada fueron `maize_desease` (~6 508 eliminadas) y `multi_desease` (~1 980). Sin esta limpieza, imágenes idénticas habrían aparecido tanto en train como en validación, inflando artificialmente las métricas.

El dataset en `data/clean/` es **post-deduplicación**. Los registros de cada ejecución se almacenan en `src/cleanup/results/`.

- [Ver análisis completo: sección 1.6 de la notebook](https://github.com/daiv05/corn-leaf-desease-project/blob/master/notebooks/01_eda.ipynb)

---

## 6. Sesgos identificados

A partir de los análisis anteriores se identifican cinco sesgos que afectan directa o indirectamente el entrenamiento:

![Composición lab/real y desbalance relativo](/eda/eda_06_sesgos.png)

| Sesgo | Clases afectadas | Riesgo |
|---|---|---|
| Desbalance de clases (32.9× entre extremos) | Todas | Alto |
| Dominio de imágenes de laboratorio | `common_rust` (95.4 % lab) | Alto |
| Heterogeneidad visual intraclase | `fall_armyworm` (daño vs. daño+gusano) | Medio-alto |
| Fuente única en clases pequeñas | `nitrogen`, `phosphorus`, `potassium` | Medio-alto |
| Concentración geográfica | `northern_corn_leaf_blight` | Medio |

El sesgo de `fall_armyworm` es especialmente relevante porque no es cuantitativo: mezcla dos señales diagnósticas distintas según el dataset de origen - hoja con daño sin insecto visible (`corn_leaf_roboflow`, `maize_africa/Activity`) y hoja con daño y gusano visible (`maize_africa/Pest`, `multicrop`). Esto puede hacer que el modelo aprenda el insecto como atajo en lugar del patrón de daño foliar.

- [Ver análisis completo: sección 1.7 de la notebook](https://github.com/daiv05/corn-leaf-desease-project/blob/master/notebooks/01_eda.ipynb)

---

## Conclusiones

1. **Muestreo ponderado es no negociable.** El ratio 32.9× entre `healthy` y `potassium_deficiency` hace que entrenar sin `WeightedRandomSampler` o `class_weight` produzca un clasificador trivial que ignora las clases más importantes clínicamente.

2. **La validación debe medir generalización de campo.** Incluir imágenes de laboratorio en validación daría una falsa sensación de buen rendimiento para clases con sesgo de dominio fuerte (`common_rust`).

3. **El dataset actual es suficiente para un baseline, no para producción.** Las deficiencias nutricionales tienen una sola fuente y pocos ejemplos. Un modelo entrenado con este dataset conoce condiciones de una región específica, no universales.

4. **Los duplicados estaban entre datasets, no dentro.** La deduplicación fue crítica para la integridad de los splits: sin ella, data leakage directo habría inflado las métricas de validación hasta ~10 puntos porcentuales en clases afectadas.
