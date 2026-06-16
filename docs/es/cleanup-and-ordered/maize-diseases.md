# maize-diseases

## Identificador

`maize_desease`

## Common Rust (CR) | Puccinia sorghi

Todas las imágenes disponibles fueron tomadas en entornos controlados.

Se detectó que las imágenes presentes en la v1 y v1.1 del dataset son exactamente las mismas, por lo que se decidió eliminar la v1.1 del dataset limpio.

Las imágenes se añadieron a /clean/lab y renombraron para identificarlas.

## Gray Leaf Spot (GLS) | Cercospora zeae-maydis

No se incluyeron imágenes de esta enfermedad en este bloque debido a que el algoritmo PHash detectó que las 513 imágenes disponibles eran clones exactos de las imágenes ya integradas mediante el dataset `cropdg-unified-multidomain`.

## Northern Corn Leaf Blight (NCLB) | Exserohilum turcicum

Se reunieron alrededor de 1056 imagenes en un entorno de campo abierto. El lugar donde fueron rescatadas es en /maize-diseases/v1.1.
Adicionalmente, se detectaron imágenes duplicadas con una suma de 4223 imágenes, por lo que se decidió depurar.

## Healthy | Sana

Muestras originales validadas e integradas en el repositorio consolidado. ~5,326 imágenes netas post-deduplicación.