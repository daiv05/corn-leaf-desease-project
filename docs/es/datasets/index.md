# Recopilación de Datasets

En esta sección se documentan los datasets evaluados para el proyecto. Se priorizan fuentes con imágenes en entornos de campo real; los datasets de laboratorio (PlantVillage y derivados) se consideran como auxiliares para el entrenamiento inicial.

## Criterios de Evaluación

Para cada dataset se registra:

- **Licencia y permisos** — si permite uso académico y redistribución
- **Dominio de captura** — laboratorio controlado, campo real, o mixto
- **Clases disponibles** — enfermedades etiquetadas y alineación con las 4 clases objetivo
- **Distribución por clase** — balance y posibles sesgos
- **Calidad y particularidades** — resolución, condiciones de iluminación, fondos, artefactos

## Resumen de Datasets

| Dataset | Dominio | Imágenes (maíz) | Clases objetivo disponibles | Licencia |
|---|---|---|---|---|
| [Maize in Field Dataset](/es/datasets/maize-in-field-dataset) | Campo real 🌿 | ~2 223 útiles (2 355 total) | Roya, NCLB, GLS, Sano | CC BY-NC-SA 4.0 |
| [Maize Diseases](/es/datasets/maize-diseases) | Laboratorio + campo 🔬🌿 | ~16 162 (v1.0 + v1.1) | Roya, NCLB, GLS, Sano | CC BY-NC-SA 4.0 |
| [Corn Leaf Diseases](/es/datasets/corn-leaf-diseases) | Laboratorio augmentado 🔬 | 52 360 (aug ×17) | Roya, NCLB, GLS, Sano | MIT |
| [CropDG Unified Multi-Domain](/es/datasets/cropdg-unified-multidomain) | Multi-dominio 🔬🌿 | ~13 275 (maíz + tomate) | NCLB, GLS, Sano (sin Roya) | CC BY-NC-SA 4.0 |
| [Maize, Beans & Tomatoes — África](/es/datasets/maize-beans-tomatoes-africa) | Campo real 🌿 | 23 286 (12 clases) | Roya (~99), Sano (2 340) | Apache 2.0 + CC |

## Inventario por Clase

Conteo consolidado de imágenes **originales únicas** (sin augmentación sintética), deduplicando fuentes compartidas (PlantVillage aparece en Maize Diseases, CropDG-PV y Corn Leaf Diseases).

| Clase | Lab original (únicos) | Campo real | **Total original** |
|---|---|---|---|
| Roya común | ~1 192 (PV) | ~300 (ZA) + ~99 (África) | **~1 591** |
| NCLB | ~985 (PV) + ~192 (PlantDoc) | ~554 (ZA) + ~5 029 (CCMT) | **~6 760** |
| GLS | ~513 (PV) + ~68 (PlantDoc) | ~1 084 (ZA) + ~4 285 (CCMT) | **~5 950** |
| Sano | ~1 162 (PV) + ~1 041 (CCMT) | ~285 (ZA) + ~2 340 (África) | **~4 828** |

> ZA = Maize in Field Dataset (Sudáfrica). CCMT = dominio campo de CropDG. PlantDoc = dominio PD de CropDG.

## Material de Campo Real

Las imágenes de campo real son el activo más valioso para la robustez del modelo. Su distribución actual por fuente:

| Clase | ZA (Maize in Field) | CCMT (CropDG) | PlantDoc (CropDG) | África | **Total campo real** |
|---|---|---|---|---|---|
| Roya común | 300 | — | — | ~99 | **~399** ⚠️ |
| NCLB | 554 | 5 029 | 192 | — | **~5 775** |
| GLS | 1 084 | 4 285 | 68 | — | **~5 437** |
| Sano | 285 | 1 041 | — | 2 340 | **~3 666** |

::: warning Desbalance crítico — Roya común
Con solo **~399 imágenes de campo real**, Roya común tiene **14× menos cobertura de campo** que NCLB (~5 775). Ningún dataset aporta imágenes de campo de Roya salvo Maize in Field (ZA) y el dataset africano. Esta clase requiere data augmentation prioritaria sobre sus imágenes de campo antes de cualquier evaluación final.
:::

## Potencial de Data Augmentation

El dataset **Corn Leaf Diseases** ya aplica 17 técnicas de augmentation sobre los originales de PlantVillage, con un ratio efectivo de ×17:

| Clase | Originales PV | Factor | Imágenes aug disponibles |
|---|---|---|---|
| Roya común | 953 | ×17 | 16 201 |
| NCLB | 788 | ×17 | 13 396 |
| GLS | 410 | ×17 | 6 970 |
| Sano | 929 | ×17 | 15 793 |
| **Total** | **3 080** | **×17** | **52 360** |

Aplicando el mismo ratio ×17 sobre las **~399 imágenes de campo real de Roya común** se obtendrían **~6 783 imágenes adicionales de campo**, equiparando su cobertura con NCLB y GLS. Técnicas adicionales recomendables para imágenes de campo: distorsión elástica, CutMix, perspectiva aleatoria y simulación de variaciones de iluminación natural.

## Estrategia de Uso

El objetivo de corpus es alcanzar **≥ 2 000 imágenes de campo real por clase** antes de la etapa de adaptación de dominio.

| Clase | Campo real disponible | Objetivo | Estado |
|---|---|---|---|
| Roya común | ~399 | ≥ 2 000 | Requiere augmentation ⚠️ |
| NCLB | ~5 775 | ≥ 2 000 | Cubierto ✓ |
| GLS | ~5 437 | ≥ 2 000 | Cubierto ✓ |
| Sano | ~3 666 | ≥ 2 000 | Cubierto ✓ |

**Roya común es la restricción activa**: antes de iniciar la etapa 2, se deben generar ~1 600 imágenes adicionales de campo para esta clase mediante augmentation (aplicar el mismo set de 17 técnicas ya usado en Corn Leaf Diseases sobre las ~399 originales de campo).

La estrategia de entrenamiento tiene tres etapas:

1. **Pre-entrenamiento de dominio**: ajuste fino inicial con imágenes de laboratorio (PlantVillage y derivados), que son más abundantes y uniformes.
2. **Adaptación de dominio**: segunda ronda de fine-tuning con imágenes de campo real, una vez que Roya común alcance el umbral de ≥ 2 000 mediante augmentation.
3. **Evaluación final**: conjunto de prueba completamente independiente, compuesto mayoritariamente por imágenes reales de campo.

::: warning Limitación conocida
Los modelos entrenados exclusivamente con imágenes de laboratorio suelen fallar en condiciones de campo (fondos variados, iluminación no controlada, oclusión parcial de la hoja). Por eso el conjunto de prueba prioriza imágenes reales aunque sean menos numerosas.
:::
