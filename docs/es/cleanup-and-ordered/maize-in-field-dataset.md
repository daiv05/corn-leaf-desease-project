# maize-in-field-dataset

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

### Análisis previo al movimiento

Antes de reorganizar se verificaron los siguientes puntos:

- **Correspondencia CSV ↔ disco**: las 2355 entradas del CSV coinciden exactamente con los 2355 archivos en `leaf_images/`. Sin entradas faltantes ni archivos huérfanos.
- **Duplicados en CSV**: ninguno - cada nombre de archivo aparece exactamente una vez.
- **Colisiones de destino**: ninguna - no existen dos imágenes con el mismo nombre que vayan a la misma carpeta destino.
- **Nombres con espacios**: 254 imágenes contienen espacios en su nombre (ej. `2012_03_IMG_6386_maize_diverse JM-C-Ac.JPG`). No representan un problema - se manejan correctamente con `pathlib.Path`.

### Criterio de organización

- Imagen con **una sola etiqueta** → carpeta con el nombre de la enfermedad.
- Imagen con **más de una etiqueta** → carpeta `multi_label/`.

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

### Script utilizado

`scripts/sort_maize_by_disease.py` - lee `Database.csv`, determina la carpeta destino de cada imagen y la mueve. Soporta `--dry-run` para verificar sin ejecutar cambios.

## Common Rust (CR) | Puccinia sorghi

Todas las imágenes presentes fueron tomadas en entornos reales. Se movieron a /clean/common_rust/real para el dataset limpio.

## Gray Leaf Spot (GLS) | Cercospora zeae-maydis

## Northern Corn Leaf Blight (NCLB) | Exserohilum turcicum

## Healthy | Sana