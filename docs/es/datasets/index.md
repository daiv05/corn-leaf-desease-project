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

| Dataset | Dominio | Imágenes | Clases (maíz) | Licencia |
|---|---|---|---|---|
| [Maize in Field Dataset](/es/datasets/maize-in-field-dataset) | Campo real 🌿 | ~2 355 | 5 + sano + otras | CC BY 4.0 |
| [Maize Diseases](/es/datasets/maize-diseases) | Laboratorio + campo 🔬🌿 | — | 4 (PlantVillage + PlantDoc) | Ver página |
| [Corn Leaf Diseases](/es/datasets/corn-leaf-diseases) | Laboratorio (augmentado) 🔬 | — | 4 (PlantVillage-AD) | Ver página |
| [CropDG Unified Multi-Domain](/es/datasets/cropdg-unified-multidomain) | Multi-dominio 🔬🌿 | — | Variable (generalizado) | Ver página |
| [Maize, Beans & Tomatoes — África](/es/datasets/maize-beans-tomatoes-africa) | Campo real 🌿 | — | Múltiples (10 fuentes) | Ver página |

## Estrategia de Uso

El dataset consolidado apunta a **≥ 3 200 imágenes en 4 clases**, con un mínimo aproximado de 700 imágenes por clase. La estrategia de entrenamiento tiene tres etapas:

1. **Pre-entrenamiento de dominio**: ajuste fino inicial con imágenes de laboratorio (PlantVillage y derivados), que son más abundantes y uniformes.
2. **Adaptación de dominio**: segunda ronda de fine-tuning con imágenes de campo real para reducir el sesgo hacia condiciones controladas.
3. **Evaluación final**: conjunto de prueba completamente independiente, compuesto mayoritariamente por imágenes reales de campo.

::: warning Limitación conocida
Los modelos entrenados exclusivamente con imágenes de laboratorio suelen fallar en condiciones de campo (fondos variados, iluminación no controlada, oclusión parcial de la hoja). Por eso el conjunto de prueba prioriza imágenes reales aunque sean menos numerosas.
:::
