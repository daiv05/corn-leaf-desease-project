# maize-in-field-dataset

## Identificador

`maize_field`

## Reorganización por enfermedad

### Estado original

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

### Criterio de organización

- Imagen con **una sola etiqueta** - carpeta con el nombre de la enfermedad.
- Imagen con **más de una etiqueta** - carpeta `multi_label/`.

### Resultado

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

La carpeta `leaf_images/` quedó vacía y fue eliminada. Los archivos `Database.csv` y `ReadMe.txt` permanecen en la raíz del dataset.

### Estructura final

```
maize-in-field-dataset/
├── Database.csv
├── ReadMe.txt
├── CR/             (107 imágenes)
├── GLS/            (630 imágenes)
├── multi_label/    (891 imágenes)
├── NCLB/           (140 imágenes)
├── NoFoliarSymptoms/ (232 imágenes)
├── Other/          (49 imágenes)
├── PLS/            (149 imágenes)
├── SR/             (7 imágenes)
└── UnidentifiedDisease/ (150 imágenes)
```

## Common Rust (CR) | Puccinia sorghi

Todas las imágenes presentes fueron tomadas en entornos reales. Se movieron al dataset limpio.

## Gray Leaf Spot (GLS) | Cercospora zeae-maydis

Las imagenes con esta enfermedad fueron tomadas 607 en entornos reales. Se movieron al dataset limpio. De 629 imágenes, se encontraron 22 duplicados que fueron eliminados.

## Northern Corn Leaf Blight (NCLB) | Exserohilum turcicum

Se recopilaron alrededor de 140 imágenes con esta enfermedad. Todas fueron tomadas en entornos reales y se movieron al dataset limpio. No se encontraron imágenes duplicadas.

## Healthy | Sana

Las 232 imágenes de la carpeta `NoFoliarSymptoms` fueron validadas y enviadas al dataset limpio bajo la clase `healthy`. No se encontraron imágenes duplicadas.