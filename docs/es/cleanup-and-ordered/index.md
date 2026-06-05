# Limpieza y ordenado de datasets

Aquí se resume el proceso y hallazgos encontrados al limpiar, clasificar y estandarizar las imágenes antes de incorporarlas al dataset final. La prioridad es preservar la trazabilidad (origen del dataset) y evitar modificar el material crudo (raw).

## Objetivo

- Consolidar imágenes útiles por clase (enfermedad) y contexto (lab/real).
- Eliminar duplicados, outliers y material procesado (recortes, filtros, etc.).
- Estandarizar nombres para facilitar auditorías y entrenamiento.

### Descartado

**`corn-leaf-diseases-plant-village-augmented-data`**

Debido a que contiene imágenes procesadas (recortes, filtros, augmentaciones) y no se dispone del material original, se decidió omitir este dataset para evitar confusiones. Se mantiene documentado en esta etapa de limpieza para referencia futura.

### Por procesar

Se exploran los siguientes datasets, y se definen identificadores para cada uno, que se incluirán en los nombres de las imágenes para mantener la trazabilidad:

**DATASET                                       -----> IDENTIFICADOR**

`cropdg-unified-multidomain                        --> cropdg`

`maize-beans-and-tomatoes-image-dataset-for-africa --> maize_africa`

`maize-diseases                                    --> maize_desease`

`maize-in-field-dataset                            --> maize_field`

`multicrop-disease-maiz-disease-pests-and-disease  --> multi_desease`

## Clases consideras (junio 2026) 

1. `Roya común` | `Common Rust` | `Puccinia sorghi` |

2. `Tizón foliar del norte (NCLB)` | `Northern Corn Leaf Blight` | `Exserohilum turcicum` |

3. `Mancha gris de la hoja (GLS)` | `Gray Leaf Spot` | `Cercospora zeae-maydis` |

4. `Hoja sana` | `Healthy` | `-` |

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

(Pendiente de completar tras la limpieza y ordenado de los datasets)