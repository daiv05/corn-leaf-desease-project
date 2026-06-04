# Limpieza y ordenado de datasets

Aquí se resume el proceso para limpiar, clasificar y estandarizar imágenes antes de incorporarlas al dataset final. La prioridad es preservar la trazabilidad (origen del dataset) y evitar tocar el material crudo.

## Objetivo

- Consolidar imágenes útiles por clase (enfermedad) y contexto (lab/real).
- Eliminar duplicados, outliers y material procesado (recortes, filtros, etc.).
- Estandarizar nombres para facilitar auditorías y entrenamiento.

### DESCARTADO

**`corn-leaf-diseases-plant-village-augmented-data`**

### POR PROCESAR

Cada integrante debe revisar estos datasets en busca de imágenes de la clase asignada:

**DATASET                                       —————> IDENTIFICADOR**

`cropdg-unified-multidomain                        --> **cropdg**`

`maize-beans-and-tomatoes-image-dataset-for-africa --> **maize_africa**`

`maize-diseases                                    --> maize_desease`

`maize-in-field-dataset                            --> maize_field`

`multicrop-disease-maiz-disease-pests-and-disease  --> multi_desease`

## Enfermedades y división

1. `Roya común` | `Common Rust` | `Puccinia sorghi` |

2. `Tizón foliar del norte (NCLB)` | `Northern Corn Leaf Blight` | `Exserohilum turcicum` |

3. `Mancha gris de la hoja (GLS)` | `Gray Leaf Spot` | `Cercospora zeae-maydis` |

4. `Hoja sana` | `Healthy` | `—` |

## Flujo de trabajo

1. Clasificar por entorno en `/data/clean`:
    1. `lab`: fotos en entornos controlados (fondo negro/blanco/gris, estudio).
    2. `real`: fotos tomadas en campo.

    **COPIAR LAS FOTOS HACIA AHI, NO TOCAR LAS DE RAW**

    - Agregar el identificador del dataset en un paso intermedio para conservar el origen.
    - Excluir imágenes procesadas (recortes, filtros, augmentaciones). Si un dataset tiene este tipo de imágenes, documentarlo en /docs y omitirlas.

2. Eliminar duplicados con `imagededup`. Usar scripts en `src/cleanup` y **EJECUTAR SOLO PARA LA CARPETA QUE ESTÉ TRABAJANDO**.
3. Estandarizar nombres cuando ya estén ordenadas en `/clean`:
    - Anteponer el nombre de la clase en inglés:
        - common_rust
        - northern_corn_leaf_blight
        - gray_leaf_spot
        - healthy
    - Agregar el identificador del dataset.
    - Incluir el ambiente: `real` o `lab`.
    - Terminar con correlativo ascendente: `_1`, `_2`, `_3`, etc.
    - Ejemplo final: `common_rust_maize_africa_lab_1`.

    Este orden garantiza trazabilidad y evita colisiones entre datasets.

4. Revisar y eliminar outliers: imágenes que no correspondan a la enfermedad, tengan texto, tomas aéreas o detalles inconsistentes.

## Documentación

- Para cada dataset se registran decisiones y hallazgos (dataset descartado, criterios, problemas de calidad), además de anotar cualquier ajuste al proceso para mantener la trazabilidad del set final.