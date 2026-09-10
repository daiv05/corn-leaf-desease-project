# Fase 0 — Auditoría de procedencia

Instrumentación del corpus y medición de los mecanismos de fuga presentes en la partición
`seed_42`. Fecha de ejecución: 2026-09-09.

Reproducible con:

```bash
python scripts/experiments/provenance_audit.py
```

Artefactos generados en `outputs/experiments/provenance/`: `manifest_with_source.csv`,
`source_by_class.csv`, `source_by_split.csv` y `provenance_audit.json`.

## Resumen

| Comprobación | Resultado | Estado |
|---|---|---|
| Imágenes con `source_id` asignado | 33 437 de 33 437 | Verificado |
| Tokens de fuente hallados | 14 | Verificado |
| Tokens sin dataset documentado | 0 | Verificado |
| Datasets documentados con imágenes | 11 de 12 | Verificado |
| Imágenes en `clean/` ausentes de los splits | 4 | Verificado, causa identificada |
| Duplicados exactos repartidos entre splits | 0 | Verificado |
| Casi-duplicados repartidos entre splits | entre 52 y 242 grupos | Cota, verificación parcial |
| Fuentes retenidas fuera de entrenamiento | 0 de 14 | Verificado |

## 1. Identificador de fuente

El corpus sigue el convenio `<clase>_<fuente>_<entorno>_<id>.<ext>`, así que `source_id` es
derivable sin trabajo de anotación. La extracción ancla el prefijo de clase y el sufijo de
entorno, porque los nombres de clase contienen guiones bajos.

**Las 33 437 imágenes reciben token; ninguna queda sin asignar.**

### Verificación contra la lista canónica

Los 14 tokens se contrastaron contra los 12 datasets de
[Recopilación de Datasets](/es/datasets/). Ningún token queda sin dataset documentado.

Tres tokens corresponden a versiones de un mismo origen y **deben tratarse como una sola
fuente** al agrupar, o la fuga persiste entre versiones:

| Origen documentado | Tokens |
|---|---|
| Maize, Beans & Tomatoes África | `maize_africa`, `maize_africa_v1`, `maize_africa_v1.2` |
| Maize Diseases | `maize_desease`, `maize_desease_v1.1` |

El dataset **Corn Leaf Diseases** (MIT, aumentado ×17) está documentado pero **no aporta
ninguna imagen** a `clean/`. No se integró.

## 2. Distribución fuente × clase

| Fuente | Total | Clases | Clases que aporta |
|---|---:|---:|---|
| `maize_africa` | 9 830 | 3 | fall_armyworm, lethal_necrosis, northern |
| `multi_desease` | 5 816 | 3 | common_rust, fall_armyworm, lethal_necrosis |
| `maize_desease_v1.1` | 4 741 | **1** | healthy |
| `corn_leaf_roboflow` | 2 919 | 6 | cogollero, healthy, N, P, K, northern |
| `cropdg` | 2 562 | 3 | gray_leaf_spot, healthy, northern |
| `maize_desease` | 2 248 | 2 | common_rust, northern |
| `maize_africa_v1.2` | 1 854 | **1** | healthy |
| `maize_field` | 852 | 3 | common_rust, gray_leaf_spot, northern |
| `maize_2_roboflow` | 846 | 3 | N, P, K |
| `corn_leaf_diseases_classification_roboflow` | 480 | **1** | gray_leaf_spot |
| `maize_africa_v1` | 460 | **1** | healthy |
| `maize_nutrient` | 340 | 4 | healthy, N, P, K |
| `maize_leaf_roboflow` | 331 | **1** | gray_leaf_spot |
| `maize_deficiency_scanner_roboflow` | 158 | 3 | N, P, K |

Cinco fuentes aportan **una sola clase**. En particular, `maize_desease_v1.1`,
`maize_africa_v1.2` y `maize_africa_v1` suman **7 055 imágenes que son todas `healthy`**, el
80,7 % de esa clase. Reconocer la fuente equivale a reconocer la clase.

### Fuentes por clase

| Clase | Fuentes | Clase | Fuentes |
|---|---:|---|---:|
| `lethal_necrosis` | **2** | `phosphorus_deficiency` | 4 |
| `common_rust` | 3 | `potassium_deficiency` | 4 |
| `fall_armyworm` | 3 | `northern_corn_leaf_blight` | 5 |
| `gray_leaf_spot` | 4 | `healthy` | 6 |
| `nitrogen_deficiency` | 4 | | |

Ocho de nueve clases tienen tres o más fuentes, así que **la validación dejando una fuente
fuera es viable**. `lethal_necrosis` admite sólo dos pliegues.

## 3. La partición actual no retiene ninguna fuente

`HierarchicalStratifiedSplitter` aplica `train_test_split` estratificando por
`label + "_" + environment`, sin agrupar por nada. El resultado medido: **las 14 fuentes
aparecen en train, val y test simultáneamente**, en proporciones prácticamente idénticas.

Estratificar por `label_environment` no es neutro respecto al problema: **garantiza** que la
mezcla de fuentes sea la misma en las tres particiones, que es la condición ideal para que un
atajo de procedencia transfiera de entrenamiento a prueba.

## 4. Cuatro imágenes fuera de los splits

`clean/` contiene 33 437 imágenes y los splits 33 433. Las cuatro ausentes son:

```
clean/healthy/real/healthy_maize_africa_v1.2_real_9.jpg
clean/healthy/real/healthy_maize_africa_v1.2_real_49.jpg
clean/healthy/real/healthy_maize_africa_v1.2_real_892.jpg
clean/healthy/real/healthy_maize_africa_v1.2_real_945.jpg
```

**Causa identificada:** cada una es un duplicado byte a byte de una imagen etiquetada
`fall_armyworm`.

| Archivo bajo `healthy/` | Duplicado exacto bajo `fall_armyworm/` |
|---|---|
| `..._real_9.jpg` | `fall_armyworm_maize_africa_real_18470217.jpg` |
| `..._real_49.jpg` | `fall_armyworm_maize_africa_real_60178451.jpg` |
| `..._real_892.jpg` | `fall_armyworm_maize_africa_real_30284126.jpg` |
| `..._real_945.jpg` | `fall_armyworm_maize_africa_real_56642353.jpg` |

La deduplicación por contenido las descartó y conservó la copia `fall_armyworm`. El
mecanismo funcionó, pero el hallazgo relevante es otro: **hay conflictos de etiqueta entre
clases en el corpus**. Cuatro archivos idénticos están catalogados a la vez como planta sana
y como daño por cogollero.

Se descartó la hipótesis de que fueran incorporaciones posteriores al congelado de los
splits: su fecha de modificación (2026-06-07) coincide con la de imágenes del mismo dataset
que sí están en los splits.

## 5. Duplicados exactos

Sobre las 33 437 imágenes hay **4 grupos de duplicados exactos**, los cuatro de la sección
anterior, y los cuatro cruzan clases distintas.

**Dentro de los splits no queda ningún duplicado exacto**, así que no hay fuga a nivel de
bytes entre entrenamiento y prueba.

## 6. Casi-duplicados: fuga confirmada, magnitud acotada

La ausencia de duplicados exactos no descarta reescalados ni recompresiones del mismo
original. Se midió con hash perceptual de diferencia de 64 bits y agrupación por distancia de
Hamming.

| Umbral | Grupos repartidos entre splits | De esos, con clases distintas | Imágenes implicadas |
|---:|---:|---:|---:|
| 0 | **52** | 10 | 291 |
| 1 | 104 | 20 | 582 |
| 2 | 166 | 47 | 1 115 |
| 3 | 216 | 59 | 1 872 |
| 4 | **242** | 71 | 2 923 |

**Con el criterio más estricto —hash perceptual idéntico— 52 grupos siguen repartidos entre
particiones distintas.** La fuga por casi-duplicados existe y está confirmada.

### Verificación parcial y su límite

Se inspeccionaron visualmente 8 grupos tomados al azar de los 242 del umbral 4. **Cuatro son
duplicados genuinos** (la misma larva de cogollero fotografiada una vez y presente en train y
val; la misma hoja de roya volteada horizontalmente entre val y test). **Cuatro son falsos
positivos**: primeros planos de nervadura, donde un hash de 64 bits sobre una imagen de bajo
contraste y estructura casi uniforme degenera y colisiona con otras distintas.

En consecuencia:

- **52 grupos (umbral 0) es una cota inferior fiable.**
- **242 grupos (umbral 4) es una cota superior**, con una tasa de falsos positivos estimada
  en torno al 50 % sobre una muestra de 8. Esa estimación tiene un intervalo muy ancho y **no
  debe citarse como tasa**.
- La cifra exacta **no está verificada** y requeriría, o bien revisión manual de los 242
  grupos, o bien un descriptor más discriminante que un dHash de 64 bits para los primeros
  planos de nervadura.

## 7. Dos mecanismos de fuga independientes

La auditoría deja establecidos dos, que se suman:

1. **Procedencia**: la fuente predice la clase y todas las fuentes están repartidas entre las
   tres particiones. Es el mecanismo dominante y el que mide el brazo del anillo de borde.
2. **Casi-duplicados**: al menos 52 grupos de imágenes casi idénticas repartidas entre
   particiones.

Ambos inflan el 0,9468 y ambos se corrigen en la Fase 1, el primero agrupando por
`source_id` y el segundo agrupando además por componente conexa de casi-duplicados.

## Qué queda sin verificar

| Cuestión | Por qué no se cerró |
|---|---|
| Número exacto de casi-duplicados entre splits | El dHash de 64 bits degenera en primeros planos de nervadura; se acota entre 52 y 242 |
| Si los conflictos de etiqueta van más allá de los 4 casos exactos | Los 10 grupos cross-clase del umbral 0 son candidatos, pero no se han revisado uno a uno |
| Si los tokens de versión son realmente el mismo origen | Se infiere del nombre y de la documentación; no se comprobó solapamiento de contenido entre versiones |
| Que el token de nombre refleje la descarga real | Se verificó contra la lista documentada, no contra los manifiestos de descarga originales |
