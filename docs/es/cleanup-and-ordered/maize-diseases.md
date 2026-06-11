# maize-diseases

## Control de Redundancia Intersite (v1.0 vs v1.1)
El dataset `maize-diseases 1.0` fue descartado en su totalidad del pipeline limpio tras corroborar que el 100% de sus imágenes ya estaban contenidas e integradas en la versión actualizada `maize-diseases 1.1`. Esto evita la duplicación de almacenamiento.

## Common Rust (CR) | Puccinia sorghi

Todas las imágenes disponibles fueron tomadas en entornos controlados.

Se detectó que las imágenes presentes en la v1 y v1.1 del dataset son exactamente las mismas, por lo que se decidió eliminar la v1.1 del dataset limpio.

Las imágenes se añadieron a /clean/lab y renombraron para identificarlas.

## Gray Leaf Spot (GLS) | Cercospora zeae-maydis
> **ALERTA DE DUPLICIDAD CRUZADA (DATA LEAKAGE):**
> No se incluyeron imágenes de esta enfermedad en este bloque debido a que el algoritmo PHash detectó que las 513 imágenes disponibles eran clones exactos de las imágenes ya integradas mediante el dataset `cropdg-unified-multidomain`. Se eliminaron para evitar sobreajuste. Volumen neto aportado: 0.

## Northern Corn Leaf Blight (NCLB) | Exserohilum turcicum
Se reunieron alrededor de 1056 imagenes en un entorno de campo abierto. El lugar donde fueron rescatadas es en /maize-diseases/v1.1.
Adicionalmente, se detectaron imágenes duplicadas con una suma de 4223 imágenes, por lo que se decidió depurar.
## Healthy | Sana
Muestras originales validadas e integradas en el repositorio consolidado.

---

### Métricas de Extracción
* **Volumen Útil Total (v1.1):** ~5,326 imágenes netas post-deduplicación.
* **Volumen Útil Total (v1.0):** 0 imágenes (Absorbido por v1.1).