# maize-beans-tomatoes-africa

## Common Rust (CR) | Puccinia sorghi

> **ALERTA DE CALIDAD / DESCARTE:** > Se detectó mediante inspección patológica que la enfermedad presente en este dataset **NO corresponde a la roya común** del maíz (*Puccinia sorghi*), sino que muestra características morfológicas de la roya del sur (*Puccinia polysora*). 
* **Acción:** No se incluye en la versión final del dataset limpio para evitar contaminar la firma visual de la clase, pero se mantiene documentado aquí para evitar futuras reintroducciones erróneas.

No se incluye en la versión final del dataset, pero se mantiene en esta etapa de limpieza y ordenado para evitar confusiones.

## Gray Leaf Spot (GLS) | Cercospora zeae-maydis
No se encontraron imagenes correspondientes a esta enfermedad en este dataset, por lo que no se incluye en la versión final del dataset.

## Northern Corn Leaf Blight (NCLB) | Exserohilum turcicum
Se reunieron un total de **4,223 imágenes** tomadas en un entorno de campo abierto (`real`). Las imágenes fueron extraídas de la ruta interna:
`Super_Image_Dataset_of_Maize_Beans_and_Tomatoes/Super_Image_Dataset_of_Maize_Beans_and_Tomatoes/train_data`
* **Acción:** Procesadas, renombradas bajo el patrón regulatorio y migradas a `data/clean/northern_corn_leaf_blight/real/`.

## Healthy | Sana
Muestras adicionales evaluadas según consistencia con el entorno real de campo.

---

### Métricas de Extracción
* **Volumen Útil Total:** 1,872 imágenes validadas (además del lote masivo de NCLB procesado en pasos de entrenamiento).