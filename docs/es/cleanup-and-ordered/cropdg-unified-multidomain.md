# cropdg-unified-multidomain

## Criterio de Selección Específico
El proceso de extracción para este dataset se limitó estrictamente a la carpeta **`PV`** (PlantVillage). La carpeta `CCMT` fue descartada por completo tras identificar que las imágenes en su interior habían sido tratadas y alteradas previamente.

---

## Common Rust (CR) | Puccinia sorghi

## Gray Leaf Spot (GLS) | Cercospora zeae-maydis
Se recopilaron un total de **513 imágenes** en un entorno de laboratorio (`lab`). El lugar original de donde fueron rescatadas es en `PV/Corn/Leaf_spot`. El análisis algorítmico determinó que **no se encontraron imágenes duplicadas** en este lote.
* **Destino:** `data/clean/gray_leaf_spot/lab/`

## Northern Corn Leaf Blight (NCLB) | Exserohilum turcicum
Se recopiló un total de **888 imágenes** en un entorno de laboratorio (`lab`). El lugar original de donde fueron rescatadas es en `PV/Corn/Leaf_blight`. No se detectaron duplicados.
* **Destino:** `data/clean/northern_corn_leaf_blight/lab/`

## Healthy | Sana
Muestras complementarias evaluadas bajo el mismo estándar de entorno controlado.

---

### Métricas de Extracción
* **Volumen Útil Total:** ~1,162 imágenes enviadas a la carpeta limpia.