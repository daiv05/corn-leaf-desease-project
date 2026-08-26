# Primera entrega - material congelado

Este directorio contiene el material de la **primera etapa del proyecto**. Su contenido es un
registro histórico y **no debe actualizarse** cuando cambie el dataset.

| Archivo | Qué es |
|---|---|
| `documentation_first_phase.tex` / `.pdf` | Paper de la primera entrega. **Se conserva sin cambios.** |
| `images/` | Figuras del paper. |
| `01_eda_first_phase.ipynb` | Copia exacta de `notebooks/01_eda.ipynb` tal como estaba al cierre de la primera etapa (byte a byte). |
| `01_eda_first_phase.html` | Salida estática renderizada de ese mismo notebook, con todas las gráficas embebidas. Se abre en cualquier navegador, sin necesidad de Jupyter ni del dataset. |

## Contexto

Las cifras de este material corresponden al corpus de la primera etapa: **31 622 imágenes**
en 9 clases, con un desbalance máximo de **32.9x** (`healthy` 8 744 vs. `potassium_deficiency` 266).

En **agosto de 2026**, posterior a esta entrega, el dataset se amplió con cuatro datasets
Roboflow adicionales hasta **33 438 imágenes** y el desbalance máximo bajó a **14.1x**. Ese
trabajo **no se refleja aquí a propósito**: el notebook vivo
([`notebooks/01_eda.ipynb`](../../notebooks/01_eda.ipynb)) sí está actualizado con las cifras
nuevas, y el detalle del procesamiento está en la documentación del proyecto
(`docs/es/cleanup-and-ordered/`, sección "Ampliación agosto 2026").

Para comparar ambas etapas, contrastar `01_eda_first_phase.html` (este directorio) contra la
ejecución actual del notebook vivo.

## Regeneración

La salida estática se generó con:

```bash
jupyter nbconvert --to html --output=01_eda_first_phase \
  --output-dir=reports/firts-phase notebooks/01_eda.ipynb
```

> Ejecutado **antes** de actualizar el notebook vivo. Reejecutar ese comando hoy produciría la
> versión nueva, no la de la primera entrega - no se debe regenerar salvo que se quiera
> reemplazar deliberadamente el registro histórico.
